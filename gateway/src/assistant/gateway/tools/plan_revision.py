"""Plan revision tool for modifying future steps during execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseTool

if TYPE_CHECKING:
    from ..planner import PlanExecution


class RevisePlanTool(BaseTool):
    """Tool for revising future steps in the current plan.

    Only future steps (status == 'pending') can be modified.
    Completed steps and the currently-executing step are immutable.
    """

    name = "revise_plan"

    def __init__(self, plan_execution: PlanExecution) -> None:
        self.plan_execution = plan_execution

    def schema(self) -> dict[str, Any]:
        """Return OpenAI-format function schema with wrapped structure."""
        return {
            "type": "function",
            "function": {
                "name": "revise_plan",
                "description": (
                    "Modify the remaining steps in the current plan. Use this when "
                    "you discover during execution that the plan needs to change. "
                    "You can add new steps after the current step, remove future "
                    "steps that are no longer needed, or replace all future steps "
                    "with a new list. Completed and currently-executing steps cannot "
                    "be changed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "remove", "replace"],
                            "description": (
                                "'add': insert new steps after a specified position. "
                                "'remove': delete specific future steps by their step numbers. "
                                "'replace': replace ALL future steps with a new list "
                                "(use for reordering or major revisions).",
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why this revision is needed. "
                                "Be specific — reference what you discovered.",
                            ),
                        },
                        # For "add" action:
                        "insert_after": {
                            "type": "integer",
                            "description": (
                                "Step number after which to insert new steps. "
                                "Must be >= current step number. Required when action='add'."
                            ),
                        },
                        "new_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Descriptions of steps to add (for action='add') or "
                                "the complete list of new future steps (for action='replace')."
                            ),
                        },
                        # For "remove" action:
                        "steps_to_remove": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Step numbers to remove. "
                                "Must all be future steps. "
                                "Required when action='remove'.",
                            ),
                        },
                    },
                    "required": ["action", "reason"],
                    # Note: Additional fields are conditionally required based on action.
                    # Validation for conditional requirements is performed in execute().
                },
            },
        }

    async def execute(self, root_path: str | None = None, **arguments: Any) -> dict[str, Any]:
        """Execute the plan revision. root_path is ignored — this is pure in-memory.

        Returns dict with stdout, stderr, exit_code following the standard tool pattern.
        """
        pe = self.plan_execution

        # Check revision limit
        can_revise, msg = pe.can_revise()
        if not can_revise:
            return {
                "stdout": "",
                "stderr": msg,
                "exit_code": 1,
            }

        action = arguments["action"]
        reason = arguments.get("reason", "")
        previous_snapshot = pe._snapshot_descriptions()

        # The current step number (1-indexed, for the model's perspective)
        current_step_num = pe.current_step + 1

        if action == "add":
            return await self._handle_add(arguments, current_step_num, reason, previous_snapshot)
        elif action == "remove":
            return await self._handle_remove(arguments, current_step_num, reason, previous_snapshot)
        elif action == "replace":
            return await self._handle_replace(
                arguments, current_step_num, reason, previous_snapshot
            )
        else:
            return {
                "stdout": "",
                "stderr": f"Unknown action: {action}",
                "exit_code": 1,
            }

    async def _handle_add(
        self, args: dict[str, Any], current_step_num: int, reason: str, prev_snapshot: list[str]
    ) -> dict[str, Any]:
        pe = self.plan_execution
        new_step_descriptions = args.get("new_steps")
        insert_after = args.get("insert_after")

        if not new_step_descriptions:
            return {
                "stdout": "",
                "stderr": "action='add' requires 'new_steps' (list of step descriptions)",
                "exit_code": 1,
            }
        if insert_after is None:
            return {
                "stdout": "",
                "stderr": "action='add' requires 'insert_after' (step number)",
                "exit_code": 1,
            }
        if insert_after < current_step_num:
            return {
                "stdout": "",
                "stderr": (
                    f"Cannot insert before step {insert_after} — "
                    f"current step is {current_step_num}. "
                    "Only future positions are allowed."
                ),
                "exit_code": 1,
            }

        # Convert to 0-indexed insert position (after the specified step)
        insert_idx = insert_after  # insert_after=3 means insert at index 3 (after step 3)
        if insert_idx > len(pe.steps):
            insert_idx = len(pe.steps)

        from ..planner import PlanStep

        for i, desc in enumerate(new_step_descriptions):
            pe.steps.insert(
                insert_idx + i,
                PlanStep(number=0, description=desc, status="pending"),
            )

        pe.renumber_steps()
        self._record_revision(
            "add",
            reason,
            prev_snapshot,
            f"Added {len(new_step_descriptions)} step(s) after step {insert_after}",
        )

        # Format result following standard tool pattern (stdout/stderr/exit_code)
        summary = (
            f"Plan revised: Added {len(new_step_descriptions)} step(s) "
            f"after step {insert_after}"
        )
        revised_plan_json = "\n".join(
            f"  Step {s['step']}: {s['description']}" for s in self._format_plan_summary()
        )
        return {
            "stdout": f"Reason: {reason}\n{summary}\n\nRevised plan:\n{revised_plan_json}",
            "stderr": "",
            "exit_code": 0,
        }

    async def _handle_remove(
        self, args: dict[str, Any], current_step_num: int, reason: str, prev_snapshot: list[str]
    ) -> dict[str, Any]:
        pe = self.plan_execution
        steps_to_remove = args.get("steps_to_remove")

        if not steps_to_remove:
            return {
                "stdout": "",
                "stderr": "action='remove' requires 'steps_to_remove' "
                "(list of step numbers to remove)",
                "exit_code": 1,
            }

        # Validate all targets are future steps
        errors = []
        for sn in steps_to_remove:
            if sn <= current_step_num:
                errors.append(
                    f"Step {sn} cannot be removed — "
                    "it is completed or currently executing"
                )
            elif sn < 1 or sn > len(pe.steps):
                errors.append(f"Step {sn} does not exist (plan has {len(pe.steps)} steps)")
        if errors:
            return {
                "stdout": "",
                "stderr": "; ".join(errors),
                "exit_code": 1,
            }

        # Remove in reverse order to preserve indices during removal
        for sn in sorted(steps_to_remove, reverse=True):
            idx = sn - 1
            pe.steps.pop(idx)

        pe.renumber_steps()
        # current_step index is still valid because we only removed steps after it
        self._record_revision(
            "remove", reason, prev_snapshot, f"Removed step(s) {steps_to_remove}"
        )

        # Format result following standard tool pattern (stdout/stderr/exit_code)
        summary = f"Plan revised: Removed {len(steps_to_remove)} step(s)"
        revised_plan_json = "\n".join(
            f"  Step {s['step']}: {s['description']}" for s in self._format_plan_summary()
        )
        return {
            "stdout": f"Reason: {reason}\n{summary}\n\nRevised plan:\n{revised_plan_json}",
            "stderr": "",
            "exit_code": 0,
        }

    async def _handle_replace(
        self, args: dict[str, Any], current_step_num: int, reason: str, prev_snapshot: list[str]
    ) -> dict[str, Any]:
        pe = self.plan_execution
        new_step_descriptions = args.get("new_steps")

        if new_step_descriptions is None:
            return {
                "stdout": "",
                "stderr": "action='replace' requires 'new_steps' "
                "(list of new future step descriptions)",
                "exit_code": 1,
            }

        from ..planner import PlanStep

        # Remove all future steps (everything after current_step)
        pe.steps = pe.steps[: pe.current_step + 1]

        # Append new future steps
        for desc in new_step_descriptions:
            pe.steps.append(PlanStep(number=0, description=desc, status="pending"))

        pe.renumber_steps()
        self._record_revision(
            "replace",
            reason,
            prev_snapshot,
            f"Replaced future steps with {len(new_step_descriptions)} new step(s)",
        )

        # Format result following standard tool pattern (stdout/stderr/exit_code)
        summary = (
            "Plan revised: Replaced all future steps with "
            f"{len(new_step_descriptions)} new step(s)"
        )
        revised_plan_json = "\n".join(
            f"  Step {s['step']}: {s['description']}" for s in self._format_plan_summary()
        )
        return {
            "stdout": f"Reason: {reason}\n{summary}\n\nRevised plan:\n{revised_plan_json}",
            "stderr": "",
            "exit_code": 0,
        }

    def _record_revision(
        self, action: str, reason: str, prev_snapshot: list[str], description: str
    ) -> None:
        from ..planner import PlanRevision

        pe = self.plan_execution
        pe.revisions.append(
            PlanRevision(
                revision_number=len(pe.revisions) + 1,
                action=action,
                description=description,
                reason=reason,
                previous_steps=prev_snapshot,
                new_steps=pe._snapshot_descriptions(),
            )
        )

    def _format_plan_summary(self) -> list[dict[str, str | object]]:
        """Return the full plan with status markers for the model and client."""
        result: list[dict[str, str | object]] = []
        for step in self.plan_execution.steps:
            markers = {
                "completed": "✓",
                "in_progress": "→",
                "pending": " ",
                "failed": "✗",
                "skipped": "–",
            }
            marker = markers.get(step.status, " ")
            result.append(
                {
                    "step": step.number,
                    "status": step.status,
                    "description": f"[{marker}] {step.description}",
                }
            )
        return result
