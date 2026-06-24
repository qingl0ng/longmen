"""Discover → Plan → Execute orchestrator for complex multi-step tasks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from .agent_loop import ExecutionMode, run_agent_loop
from .compactor import Compactor
from .protocol import make_plan_revision, make_plan_status, make_stream_end
from .tools import get_known_tools, get_schemas
from .tools.plan_revision import RevisePlanTool

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from .config import GatewayConfig
    from .context_manager import ContextManager
    from .permissions import PermissionManager
    from .session import Session

__all__ = ["ExecutionMode", "Planner"]

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants: which requests trigger discovery
# ---------------------------------------------------------------------------

_COMPLEX_KEYWORDS = {
    "implement",
    "add feature",
    "refactor",
    "migrate",
    "set up",
    "create",
    "research",
    "analyze",
    "analyse",
    "compare",
    "investigate",
    "build",
    "write",
    "design",
    "architect",
    "restructure",
    "rewrite",
}

# ---------------------------------------------------------------------------
# Tool whitelists / blacklists for discovery phase
# ---------------------------------------------------------------------------

_DISCOVERY_ALLOWED_TOOLS = frozenset(
    {
        "tree",
        "grep",
        "symbols",
        "read_file",
        "list_dir",
        "detect_project",
        "git_status",
        "git_log",
        "git_diff",
        "shell",
        "web_search",
        "web_fetch",
        "rag_search",
        "delete_tool",
    }
)

_DISCOVERY_BLOCKED_TOOLS = frozenset(
    {
        "write_file",
        "search_replace",
        "build",
        "run_tests",
        "run_app",
        "sql_query",
        "git_add",
        "git_commit",
    }
)

_DISCOVERY_BLOCKED_MSG = (
    "Discovery phase — write tools and build/test commands are not available yet. "
    "Explore the codebase to understand the problem, then call create_plan when ready to act."
)

# Tools blocked during triage (re-run forbidden until a change is made)
_TRIAGE_BLOCKED_TOOLS = frozenset({"build", "run_tests"})
_TRIAGE_BLOCKED_MSG = (
    "You must fix the code before re-running. "
    "Apply a change based on your investigation using search_replace or write_file."
)

# Tools that signal a write was made during fix phase
_WRITE_TOOLS = frozenset({"write_file", "search_replace"})

# Read-only tools counted against the triage read budget
_TRIAGE_READ_TOOLS = frozenset(
    {
        "read_file",
        "symbols",
        "grep",
        "tree",
        "list_dir",
        "git_diff",
        "git_log",
        "git_status",
        "detect_project",
        "shell",
    }
)

# Tools that trigger self-correction when they fail
_BUILD_TEST_TOOLS = frozenset({"build", "run_tests"})

# ---------------------------------------------------------------------------
# Tool schema for create_plan
# ---------------------------------------------------------------------------

_CREATE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "create_plan",
        "description": (
            "Submit a plan for the complex task. Call this after you have explored the codebase "
            "and understand the problem well enough to create specific, actionable steps. "
            "This transitions from discovery to execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Ordered list of specific, actionable steps. Each step should be a "
                        "single verifiable action. Include verification steps (build, test) "
                        "after code changes. 3-10 steps."
                    ),
                },
                "context_summary": {
                    "type": "string",
                    "description": (
                        "Brief summary of what you discovered during exploration. "
                        "This is preserved across compaction."
                    ),
                },
            },
            "required": ["steps", "context_summary"],
        },
    },
}

# ---------------------------------------------------------------------------
# System prompt fragments
# ---------------------------------------------------------------------------

_DISCOVERY_SYSTEM_PROMPT = """
You are in the discovery phase for a complex task. Your goal is to understand
the codebase well enough to create a solid plan.

Available tools: tree, grep, symbols, read_file, list_dir, detect_project, git_status,
  git_log, git_diff, shell (safe/read-only commands only)
NOT available yet: write_file, search_replace, build, run_tests

Explore the codebase systematically:
1. Start with tree to see the project structure
2. Use symbols to understand key files before reading them
3. Use grep to find related code, tests, and patterns
4. Use read_file with line ranges for specific sections
5. Use shell for investigative commands (ls, find, cat on non-code files)

When you have enough understanding to create a concrete plan, call create_plan
with your steps. Each step should be specific enough that you could execute it
without further exploration.

Do NOT try to write files or run builds yet — that happens after planning.
""".strip()

_SELF_CORRECTION_SYSTEM_PROMPT = """
When a build or test fails, follow this debugging workflow:

TRIAGE (investigate first):
1. Read the error messages — they contain file paths and line numbers
2. Use read_file with the line range from the error to see the problematic code
3. Use grep or symbols to check related code (imports, callers, interfaces)
4. Use shell for investigative commands (grep logs, check configs, inspect environment)
5. Understand the ROOT CAUSE before attempting a fix

FIX (only after you understand the problem):
6. Apply a targeted fix using search_replace
7. If the fix requires changes in multiple files, make all changes before verifying

VERIFY:
8. Re-run the build or tests
9. If a new error appears, triage it the same way

Do NOT:
- Skip investigation and guess at a fix
- Re-run the same command without making a code change
- Make speculative changes without reading the relevant code first
""".strip()

_CLARIFICATION_INTERACTIVE = """
## Handling ambiguity

When a request is ambiguous and could apply to multiple files, modules, or approaches:
1. Use tree and grep to identify candidates
2. Present numbered options:
   "I found multiple matches:
    1) src/auth/login.py — handles password-based login
    2) src/auth/oauth.py — handles OAuth2 flow
    Which file should I modify?"

NEVER guess on destructive or irreversible actions.
ALWAYS ask when:
- Multiple files match a vague reference
- A destructive command could affect production data
- The request could mean fundamentally different things
""".strip()

_CLARIFICATION_WORKFLOW = """
## Handling ambiguity

When a request is ambiguous:
1. Use available tools to gather context (tree, grep, symbols)
2. Make the best inference based on file names, content, and project structure
3. State your assumption clearly: "Assuming X because Y"
4. Do NOT wait for user input — proceed with your best inference
5. For destructive actions, prefer the safest interpretation
""".strip()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    number: int
    description: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    summary: str = ""
    retries_used: int = 0


@dataclass
class PlanRevision:
    """Records a plan revision for audit trail."""

    revision_number: int
    action: str  # "add" | "remove" | "replace"
    description: str  # human-readable summary of what changed
    reason: str  # why the revision was needed
    previous_steps: list[str]  # snapshot of step descriptions before revision
    new_steps: list[str]  # snapshot of step descriptions after revision
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanExecution:
    steps: list[PlanStep]
    current_step: int = 0
    context_summary: str = ""
    execution_mode: str = "interactive"
    revisions: list[PlanRevision] = field(default_factory=list)
    max_revisions: int = 5

    @property
    def future_steps(self) -> list[PlanStep]:
        """Steps that haven't started yet — these are revisable."""
        return [s for s in self.steps if s.status == "pending"]

    @property
    def completed_steps(self) -> list[PlanStep]:
        """Steps that have finished — these are immutable."""
        return [s for s in self.steps if s.status in ("completed", "failed", "skipped")]

    @property
    def current_step_obj(self) -> PlanStep | None:
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def can_revise(self) -> tuple[bool, str]:
        """Check whether revision is allowed right now."""
        if len(self.revisions) >= self.max_revisions:
            return False, f"Revision limit reached ({self.max_revisions} per plan)"
        if self.current_step >= len(self.steps):
            return False, "Plan execution is already complete"
        return True, ""

    def _snapshot_descriptions(self) -> list[str]:
        return [s.description for s in self.steps]

    def renumber_steps(self) -> None:
        """Renumber all steps sequentially after mutation.
        Does NOT touch current_step — callers must handle that."""
        for i, step in enumerate(self.steps):
            step.number = i + 1

    def context_for_current_step(self) -> str:
        """Build context string to inject into the step system prompt."""
        lines: list[str] = []
        if self.context_summary:
            lines.append(f"Discovery summary: {self.context_summary}")
        lines.append(f"Executing step {self.current_step + 1} of {len(self.steps)}.")
        if self.current_step > 0:
            lines.append("Completed:")
            for step in self.steps[: self.current_step]:
                summary = step.summary or step.description
                lines.append(f"  \u2713 {step.number}. {summary}")
        lines.append(f"Current: {self.steps[self.current_step].description}")
        remaining = self.steps[self.current_step + 1 :]
        if remaining:
            lines.append("Remaining:")
            for step in remaining:
                lines.append(f"  \u25cb {step.number}. {step.description}")
        return "\n".join(lines)


@dataclass
class CorrectionState:
    tool_name: str
    original_error: str
    current_error: str
    attempt: int = 0
    max_retries: int = 3
    phase: str = "triage"  # triage | fix | verify
    triage_reads: int = 0
    changes_made: list[str] = field(default_factory=list)
    files_investigated: set[str] = field(default_factory=set)
    changes_made_this_cycle: int = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_complex_request(text: str, auto_plan: bool = True) -> bool:
    """Return True if this request should trigger discovery → plan → execute."""
    if not auto_plan:
        return False
    if len(text) > 100:
        return True
    text_lower = text.lower()
    return any(kw in text_lower for kw in _COMPLEX_KEYWORDS)


def build_clarification_prompt(
    execution_mode: ExecutionMode,
    verbosity: str = "ask_ambiguous",
) -> str:
    """Return the clarification section for the system prompt."""
    if execution_mode == ExecutionMode.WORKFLOW:
        return _CLARIFICATION_WORKFLOW
    # Interactive / Agent modes respect verbosity
    if verbosity == "best_guess":
        return _CLARIFICATION_WORKFLOW
    return _CLARIFICATION_INTERACTIVE


def _extract_text(content: list[Any]) -> str:
    """Extract plain text from user content blocks."""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        else:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(getattr(block, "text", ""))
    return " ".join(parts)


def _is_build_test_failure(result: dict[str, Any]) -> bool:
    """Return True if a tool result represents a build/test failure."""
    exit_code = result.get("exit_code", 0)
    if exit_code != 0:
        return True
    stderr = result.get("stderr", "")
    return bool(stderr and ("error" in stderr.lower() or "failed" in stderr.lower()))


def _extract_error_summary(result: dict[str, Any]) -> str:
    """Extract a concise error summary from a build/test result."""
    parts: list[str] = []
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if stderr:
        parts.append(stderr[:500])
    if stdout:
        # Take last part of stdout (typically where errors appear)
        lines = stdout.strip().splitlines()
        tail = "\n".join(lines[-30:]) if len(lines) > 30 else stdout
        parts.append(tail[:500])
    return "\n".join(parts) or "Unknown error"


# ---------------------------------------------------------------------------
# Planner class
# ---------------------------------------------------------------------------


class Planner:
    """Orchestrate discover → plan → execute for complex requests."""

    def __init__(
        self,
        session: Session,
        ws: Any,
        vllm_client: Any,
        config: GatewayConfig,
        ref_id: str,
        root_path: str,
        permission_manager: PermissionManager,
        context_manager: ContextManager | None,
        execution_mode: ExecutionMode,
        session_config: dict[str, Any],
        base_system_prompt: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        extra_tool_interceptors: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.ws = ws
        self.vllm_client = vllm_client
        self.config = config
        self.ref_id = ref_id
        self.root_path = root_path
        self.permission_manager = permission_manager
        self.context_manager = context_manager
        self.execution_mode = execution_mode
        self.session_config = session_config
        self.base_system_prompt = base_system_prompt
        self.plan_exec: PlanExecution | None = None
        self.max_revisions = self.config.planning.max_revisions

        # Resolve auto_compact from session_config, fall back to gateway config
        _auto_compact = self.session_config.get("auto_compact")
        if _auto_compact is None:
            _auto_compact = self.config.context.auto_compact
        _compact_target = (
            self.session_config.get("compact_target_tokens")
            or self.config.context.compact_target_tokens
        )
        self.compactor: Compactor | None = (
            Compactor(
                context_manager=self.context_manager,
                compact_target_tokens=_compact_target,
            )
            if (_auto_compact and self.context_manager is not None)
            else None
        )
        # When set, restrict available tools (e.g. agent-specific tool list)
        self._tool_schemas = tool_schemas
        self._extra_tool_interceptors = extra_tool_interceptors or {}

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    async def run_with_planning(self, user_content: list[Any]) -> tuple[int, int, int]:
        """Main entry: run discovery if complex, then execute, else direct loop.

        Returns (duration_ms, tokens_used, total_tool_calls) for the entire run.
        """
        tokens_before = self.session.tokens_used
        loop_start = time.time()
        self._accumulated_tool_calls: int = 0

        text = _extract_text(user_content)
        auto_plan = self.session_config.get("auto_plan", True)

        if not is_complex_request(text, auto_plan):
            # Simple request — go directly to agent loop
            system_prompt = self._build_base_system_prompt()
            _, _, tc = await run_agent_loop(
                session=self.session,
                user_content=user_content,
                ws=self.ws,
                vllm_client=self.vllm_client,
                config=self.config,
                ref_id=self.ref_id,
                root_path=self.root_path,
                permission_manager=self.permission_manager,
                system_prompt=system_prompt,
                tools=self._all_tools(),
                context_manager=self.context_manager,
                compactor=self.compactor,
                execution_mode=self.execution_mode,
                tool_interceptors=self._extra_tool_interceptors or None,
            )
            self._accumulated_tool_calls += tc
        else:
            # Complex request: discover → plan → execute
            plan_exec = await self._run_discovery(user_content)
            if plan_exec is not None:
                await self._run_execution(plan_exec)

        duration_ms = int((time.time() - loop_start) * 1000)
        tokens_used = self.session.tokens_used - tokens_before
        return duration_ms, tokens_used, self._accumulated_tool_calls

    # ------------------------------------------------------------------
    # Discovery phase
    # ------------------------------------------------------------------

    async def _run_discovery(self, user_content: list[Any]) -> PlanExecution | None:
        """Run the discovery phase.  Returns PlanExecution when create_plan called."""

        # Tell the client discovery has started
        await self.ws.send(
            json.dumps(
                make_plan_status(
                    session_id=self.session.session_id,
                    ref_id=self.ref_id,
                    step=0,
                    total_steps=0,
                    status="discovery",
                    description="Exploring codebase...",
                )
            )
        )

        plan_result: PlanExecution | None = None

        # --- Discovery token budget -------------------------------------------
        tokens_start = self.session.tokens_used
        discovery_budget_pct: float = self.session_config.get("discovery_budget_pct", 0.38)
        discovery_budget = self.config.model.context_limit * discovery_budget_pct
        budget_exceeded = False

        def _discovery_pct() -> float:
            if discovery_budget <= 0:
                return 0.0
            return (self.session.tokens_used - tokens_start) / discovery_budget

        def on_discovery_tool_result(
            tool_name: str, _args: dict[str, Any], _result: dict[str, Any]
        ) -> None:
            nonlocal budget_exceeded
            if _discovery_pct() >= 1.0:
                budget_exceeded = True
                log.info(
                    "planner.discovery_budget_exceeded",
                    pct=int(_discovery_pct() * 100),
                    session_id=self.session.session_id,
                )

        def get_discovery_blocked() -> tuple[frozenset[str], str] | None:
            if not budget_exceeded:
                return None  # fall back to static blocked_tools
            # Block every tool except create_plan so the model is forced to plan
            all_tool_names = frozenset(get_known_tools())
            return (
                all_tool_names,
                "Discovery budget exceeded. You must call create_plan now with "
                "your current understanding. No further exploration is available. "
                "Tip: Add exploration steps to your plan if needed. You can always "
                "use revise_plan during execution to add more exploration later.",
            )

        def get_discovery_system_prompt() -> str:
            pct = _discovery_pct()
            nudge = ""
            if pct >= 0.8:
                pct_int = int(pct * 100)
                nudge = (
                    f"\nYou've used {pct_int}% of your discovery budget. "
                    "Call create_plan soon with what you know — "
                    "you can read more files during execution if needed."
                )
            return self._join_prompt_parts(
                self._build_base_system_prompt(),
                _DISCOVERY_SYSTEM_PROMPT + nudge,
                build_clarification_prompt(
                    self.execution_mode,
                    self.session_config.get("verbosity", "ask_ambiguous"),
                ),
            )

        # -----------------------------------------------------------------------

        async def handle_create_plan(arguments: dict[str, Any]) -> dict[str, Any]:
            nonlocal plan_result
            steps_raw: list[str] = arguments.get("steps", [])
            summary: str = arguments.get("context_summary", "")
            steps = [PlanStep(number=i + 1, description=s) for i, s in enumerate(steps_raw)]
            plan_result = PlanExecution(
                steps=steps,
                context_summary=summary,
                execution_mode=self.execution_mode.value,
                max_revisions=self.max_revisions,
            )
            self.plan_exec = plan_result
            log.info(
                "planner.plan_created",
                steps=len(steps),
                session_id=self.session.session_id,
            )
            return {"__stop__": True, "status": "plan_created", "step_count": len(steps)}

        # Build restricted tool list for discovery.
        # If an agent has a restricted tool_schemas list, intersect with it so
        # the agent cannot use tools beyond its declared scope even during discovery.
        if self._tool_schemas is not None:
            agent_tool_names = {t["function"]["name"] for t in self._tool_schemas}
            allowed = _DISCOVERY_ALLOWED_TOOLS & agent_tool_names
        else:
            allowed = _DISCOVERY_ALLOWED_TOOLS & set(get_known_tools())
        discovery_tools = get_schemas(list(allowed)) + [_CREATE_PLAN_SCHEMA]

        _, _, tc = await run_agent_loop(
            session=self.session,
            user_content=user_content,
            ws=self.ws,
            vllm_client=self.vllm_client,
            config=self.config,
            ref_id=self.ref_id,
            root_path=self.root_path,
            permission_manager=self.permission_manager,
            get_system_prompt=get_discovery_system_prompt,
            tools=discovery_tools,
            context_manager=self.context_manager,
            compactor=self.compactor,
            execution_mode=self.execution_mode,
            blocked_tools=_DISCOVERY_BLOCKED_TOOLS,
            blocked_tool_message=_DISCOVERY_BLOCKED_MSG,
            tool_interceptors={"create_plan": handle_create_plan, **self._extra_tool_interceptors},
            on_tool_result=on_discovery_tool_result,
            get_blocked_tools=get_discovery_blocked,
            emit_stream_end=False,
        )
        self._accumulated_tool_calls = getattr(self, "_accumulated_tool_calls", 0) + tc

        if plan_result is not None:
            # Tell client a plan was created
            await self.ws.send(
                json.dumps(
                    make_plan_status(
                        session_id=self.session.session_id,
                        ref_id=self.ref_id,
                        step=0,
                        total_steps=len(plan_result.steps),
                        status="planned",
                        description="Plan created",
                    )
                )
            )
        else:
            # Discovery ran but create_plan was never called (timeout / error)
            # The agent loop did NOT send stream_end (emit_stream_end=False)
            # so we need to send it explicitly here
            await self.ws.send(
                json.dumps(
                    make_stream_end(
                        session_id=self.session.session_id,
                        ref_id=self.ref_id,
                        aborted=False,
                        usage=None,
                        context_limit=self.config.model.context_limit,
                        session_budget=self.session.context_budget,
                        finish_reason="stop",
                        tool_calls_made=0,
                    )
                )
            )

        return plan_result

    # ------------------------------------------------------------------
    # Execution phase
    # ------------------------------------------------------------------

    async def _run_execution(self, plan_exec: PlanExecution) -> None:
        """Execute each plan step sequentially."""
        i = 0
        while i < len(plan_exec.steps):
            plan_exec.current_step = i
            step = plan_exec.steps[i]
            step.status = "running"

            await self.ws.send(
                json.dumps(
                    make_plan_status(
                        session_id=self.session.session_id,
                        ref_id=self.ref_id,
                        step=i + 1,
                        total_steps=len(plan_exec.steps),
                        status="running",
                        description=step.description,
                    )
                )
            )

            step_context = plan_exec.context_for_current_step()
            step_system = self._join_prompt_parts(
                self._build_base_system_prompt(),
                step_context,
                _SELF_CORRECTION_SYSTEM_PROMPT,
                build_clarification_prompt(
                    self.execution_mode,
                    self.session_config.get("verbosity", "ask_ambiguous"),
                ),
            )

            success = await self._run_step(step, step_system)

            if success:
                step.status = "completed"
                if not step.summary:
                    step.summary = f"Completed: {step.description[:60]}"
                await self.ws.send(
                    json.dumps(
                        make_plan_status(
                            session_id=self.session.session_id,
                            ref_id=self.ref_id,
                            step=i + 1,
                            total_steps=len(plan_exec.steps),
                            status="completed",
                            description=step.description,
                            summary=step.summary,
                        )
                    )
                )
            else:
                step.status = "failed"
                await self.ws.send(
                    json.dumps(
                        make_plan_status(
                            session_id=self.session.session_id,
                            ref_id=self.ref_id,
                            step=i + 1,
                            total_steps=len(plan_exec.steps),
                            status="failed",
                            description=step.description,
                            summary=f"Failed after {step.retries_used} retries",
                        )
                    )
                )
                if self.execution_mode != ExecutionMode.WORKFLOW:
                    # Interactive / Agent: stop on first failure
                    break
                # Workflow: mark as failed and continue to next step

            i += 1

        # Final stream_end — the overall task is done
        await self.ws.send(
            json.dumps(
                make_stream_end(
                    session_id=self.session.session_id,
                    ref_id=self.ref_id,
                    aborted=False,
                    usage=None,
                    context_limit=self.config.model.context_limit,
                    session_budget=self.session.context_budget,
                    finish_reason="stop",
                    tool_calls_made=0,
                )
            )
        )

    # ------------------------------------------------------------------
    # Step execution with self-correction
    # ------------------------------------------------------------------

    async def _run_step(self, step: PlanStep, step_system: str) -> bool:
        """Run a single step.  Returns True on success, False if retries exhausted."""
        max_retries = self.session_config.get("max_retries", 3)
        triage_max_reads = self.session_config.get("triage_max_reads", 10)
        all_tools = self._all_tools()

        # Build tool list — include revise_plan when a plan is active
        tools: list[dict[str, Any]] | None = None
        tool_interceptors: (
            dict[str, Callable[..., Coroutine[Any, Any, dict[str, Any]]]]
        ) | None = None

        if self.plan_exec is not None:
            revise_tool = RevisePlanTool(self.plan_exec)
            # Add revise_plan schema to tools so the model can see and call it
            tools = all_tools + [revise_tool.schema()] if all_tools else [revise_tool.schema()]
            # Add revise_plan as an interceptor so it bypasses permission check
            tool_interceptors = {
                "revise_plan": self._handle_revise_plan,
                **self._extra_tool_interceptors,
            }
        else:
            tools = all_tools
            tool_interceptors = self._extra_tool_interceptors or None

        # Track messages so we can detect build/test failures afterwards
        msg_count_before = len(self.session.messages)

        # Build step user content (inject into session as a user turn)
        step_user_content: list[Any] = [
            {
                "type": "text",
                "text": (
                    f"Execute this step now: {step.description}\n\n"
                    "Important: focus only on this step. "
                    "Previous steps are already complete — do not redo their work."
                ),
            }
        ]

        _, _, tc = await run_agent_loop(
            session=self.session,
            user_content=step_user_content,
            ws=self.ws,
            vllm_client=self.vllm_client,
            config=self.config,
            ref_id=self.ref_id,
            root_path=self.root_path,
            permission_manager=self.permission_manager,
            system_prompt=step_system,
            tools=tools or None,
            context_manager=self.context_manager,
            compactor=self.compactor,
            execution_mode=self.execution_mode,
            blocked_tools=frozenset(),  # No blocked tools during plan execution
            tool_interceptors=tool_interceptors,
            emit_stream_end=False,
        )
        self._accumulated_tool_calls = getattr(self, "_accumulated_tool_calls", 0) + tc

        # Check for build/test failures in messages added during this step
        failure = self._detect_build_test_failure(msg_count_before)
        if failure is None:
            step.summary = f"Completed: {step.description[:60]}"
            return True

        # Enter correction loop
        tool_name, error_msg = failure
        correction = CorrectionState(
            tool_name=tool_name,
            original_error=error_msg,
            current_error=error_msg,
            max_retries=max_retries,
        )

        for attempt in range(max_retries):
            correction.attempt = attempt
            step.retries_used = attempt + 1

            success = await self._run_correction_cycle(
                step=step,
                correction=correction,
                step_system=step_system,
                triage_max_reads=triage_max_reads,
            )
            if success:
                step.summary = (
                    f"Completed after {attempt + 1} correction(s): {step.description[:50]}"
                )
                return True

            # If triage made no changes, further retries won't help
            if correction.changes_made_this_cycle == 0:
                log.warning(
                    "planner.correction_no_changes_skipping_retries",
                    step=step.number,
                    session_id=self.session.session_id,
                )
                break

            # Failure: update current error for next attempt
            msg_count_before_retry = len(self.session.messages)
            failure_retry = self._detect_build_test_failure(msg_count_before_retry)
            if failure_retry:
                correction.current_error = failure_retry[1]

        # Retries exhausted
        log.warning(
            "planner.correction_exhausted",
            step=step.number,
            retries=max_retries,
            session_id=self.session.session_id,
        )
        return False

    async def _run_correction_cycle(
        self,
        step: PlanStep,
        correction: CorrectionState,
        step_system: str,
        triage_max_reads: int,
    ) -> bool:
        """One triage → fix → verify cycle.  Returns True if verify passed."""
        all_tools = self._all_tools()

        # Reset changes counter for this cycle
        correction.changes_made_this_cycle = 0

        # --- Triage phase: block build/run_tests, count read-only tool calls ---

        def on_triage_tool_result(
            tool_name: str, _args: dict[str, Any], _result: dict[str, Any]
        ) -> None:
            if tool_name in _TRIAGE_READ_TOOLS:
                correction.triage_reads += 1

        def get_triage_system_prompt() -> str:
            nudge = ""
            if correction.triage_reads >= triage_max_reads:
                nudge = (
                    "\nYou've investigated extensively. "
                    "Apply your best fix using search_replace or write_file, "
                    "or skip this error if it cannot be resolved."
                )
            return self._join_prompt_parts(
                step_system,
                self._build_triage_prompt(correction, triage_max_reads) + nudge,
            )

        msg_count_before_triage = len(self.session.messages)

        triage_content: list[Any] = [
            {
                "type": "text",
                "text": (
                    f"A {correction.tool_name} command failed. "
                    f"Enter triage mode to investigate.\n\n"
                    f"Error summary:\n{correction.current_error}\n\n"
                    "Investigate the root cause using read-only tools, then apply a fix."
                ),
            }
        ]

        _, _, tc = await run_agent_loop(
            session=self.session,
            user_content=triage_content,
            ws=self.ws,
            vllm_client=self.vllm_client,
            config=self.config,
            ref_id=self.ref_id,
            root_path=self.root_path,
            permission_manager=self.permission_manager,
            get_system_prompt=get_triage_system_prompt,
            tools=all_tools or None,
            context_manager=self.context_manager,
            compactor=self.compactor,
            execution_mode=self.execution_mode,
            blocked_tools=_TRIAGE_BLOCKED_TOOLS,
            blocked_tool_message=_TRIAGE_BLOCKED_MSG,
            on_tool_result=on_triage_tool_result,
            tool_interceptors=self._extra_tool_interceptors or None,
            emit_stream_end=False,
        )
        self._accumulated_tool_calls = getattr(self, "_accumulated_tool_calls", 0) + tc

        # Check if any writes happened during triage
        messages_during_triage = self.session.messages[msg_count_before_triage:]
        changes_made_this_cycle = self._count_write_tools(messages_during_triage)
        correction.changes_made_this_cycle = changes_made_this_cycle

        if changes_made_this_cycle == 0:
            # No writes — correction failed without making any change
            log.warning(
                "planner.triage_no_changes",
                step=step.number,
                session_id=self.session.session_id,
            )
            return False

        # --- Verify phase: run without blocked tools ---
        verify_system = self._join_prompt_parts(
            step_system,
            f"Re-run {correction.tool_name} to verify your fix resolved the error.",
        )

        verify_content: list[Any] = [
            {
                "type": "text",
                "text": (
                    f"Your fix has been applied. "
                    f"Re-run {correction.tool_name} to verify the error is resolved."
                ),
            }
        ]

        msg_count_before_verify = len(self.session.messages)

        _, _, tc = await run_agent_loop(
            session=self.session,
            user_content=verify_content,
            ws=self.ws,
            vllm_client=self.vllm_client,
            config=self.config,
            ref_id=self.ref_id,
            root_path=self.root_path,
            permission_manager=self.permission_manager,
            system_prompt=verify_system,
            tools=all_tools or None,
            context_manager=self.context_manager,
            compactor=self.compactor,
            execution_mode=self.execution_mode,
            tool_interceptors=self._extra_tool_interceptors or None,
            emit_stream_end=False,
        )
        self._accumulated_tool_calls = getattr(self, "_accumulated_tool_calls", 0) + tc

        # Check if verify passed (no build/test failure in verify messages)
        failure = self._detect_build_test_failure(msg_count_before_verify)
        return failure is None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_base_system_prompt(self) -> str | None:
        return self.base_system_prompt

    def _all_tools(self) -> list[dict[str, Any]] | None:
        """Return the effective tool list (agent-specific if set, else all tools)."""
        if self._tool_schemas is not None:
            return self._tool_schemas
        return get_schemas(list(get_known_tools()))

    def _join_prompt_parts(self, *parts: str | None) -> str:
        return "\n\n".join(p for p in parts if p)

    def _build_triage_prompt(self, correction: CorrectionState, triage_max_reads: int) -> str:
        remaining = correction.max_retries - correction.attempt
        return (
            f"A {correction.tool_name} command failed. Enter triage mode to investigate.\n\n"
            f"Error summary:\n{correction.current_error}\n\n"
            "Investigate the root cause using read-only tools:\n"
            "1. Read the failing code at the error location\n"
            "2. Check related files (imports, callers, interfaces)\n"
            "3. Use shell for investigative commands "
            "(grep logs, check configs, inspect environment)\n"
            "4. Understand WHY the error occurred, not just WHERE\n\n"
            "When you understand the root cause, fix the code using search_replace or write_file.\n"
            "Do NOT guess at a fix without investigating first.\n\n"
            f"Triage attempt {correction.attempt + 1} of {correction.max_retries}. "
            f"Retry attempts remaining: {remaining}.\n"
            f"Triage read budget: {triage_max_reads} read-only tool calls."
        )

    def _detect_build_test_failure(self, since_message_idx: int) -> tuple[str, str] | None:
        """Scan session messages added since `since_message_idx` for build/test failures.

        Returns (tool_name, error_summary) or None.
        """
        messages = self.session.messages[since_message_idx:]
        for msg in messages:
            tool_name = getattr(msg, "tool_name", None)
            if tool_name not in _BUILD_TEST_TOOLS:
                continue
            content = getattr(msg, "content", "") or ""
            # Check for failure indicators
            if "exit_code: 1" in content or "exit_code: 2" in content:
                return tool_name, content[:600]
            if "error:" in content.lower() and "exit_code: 0" not in content:
                return tool_name, content[:600]
        return None

    async def _handle_revise_plan(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Interceptor for revise_plan tool — auto-approved, sends messages to client."""
        if self.plan_exec is None:
            return {
                "stdout": "",
                "stderr": "No plan is active",
                "exit_code": 1,
            }

        revise_tool = RevisePlanTool(self.plan_exec)
        result = await revise_tool.execute(root_path=None, **arguments)

        # Parse result to extract revision info
        # revision_number is computed AFTER execute, so it reflects the actual count
        revision_number_after = len(self.plan_exec.revisions)
        action = result.get("stderr", "").startswith("Revision limit") and "replace" or "add"
        reason = arguments.get("reason", "")
        description = result.get("stderr", "") or "Plan revised"

        if result.get("exit_code", 0) == 0 and revision_number_after > 0:
            # Get the last revision
            last_rev = self.plan_exec.revisions[-1]
            revision_number = last_rev.revision_number
            action = last_rev.action
            description = last_rev.description
            reason = last_rev.reason
        else:
            revision_number = revision_number_after

        # Format revised plan for message
        revised_plan = revise_tool._format_plan_summary()

        # Send plan_revision to client
        await self._send_plan_revision(
            session_id=self.session.session_id,
            ref_id=self.ref_id,
            revision_number=revision_number,
            action=action,
            reason=reason,
            description=description,
            revised_plan=revised_plan,
        )

        # Send updated plan_status for each step
        await self._send_full_plan_status()

        return result

    async def _send_plan_revision(
        self,
        session_id: str,
        ref_id: str,
        revision_number: int,
        action: str,
        reason: str,
        description: str,
        revised_plan: list[dict[str, str | object]],
    ) -> None:
        """Send plan_revision message to client."""
        await self.ws.send(
            json.dumps(
                make_plan_revision(
                    session_id=session_id,
                    ref_id=ref_id,
                    revision_number=revision_number,
                    action=action,
                    reason=reason,
                    description=description,
                    revised_plan=revised_plan,
                )
            )
        )

    async def _send_full_plan_status(self) -> None:
        """Send plan_status for each step in the current plan."""
        if self.plan_exec is None:
            return

        for step in self.plan_exec.steps:
            await self.ws.send(
                json.dumps(
                    make_plan_status(
                        session_id=self.session.session_id,
                        ref_id=self.ref_id,
                        step=step.number,
                        total_steps=len(self.plan_exec.steps),
                        status=step.status,
                        description=step.description,
                        summary=step.summary,
                    )
                )
            )

    def _count_write_tools(self, messages: list[Any]) -> int:
        """Count write tool calls in a list of session messages."""
        count = 0
        for msg in messages:
            tool_name = getattr(msg, "tool_name", None)
            if tool_name in _WRITE_TOOLS:
                count += 1
        return count
