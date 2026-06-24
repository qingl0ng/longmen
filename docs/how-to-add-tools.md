# How to Add New Tools to the Gateway

This document provides a comprehensive guide for adding new tools to the assistant gateway. Tools can be either **permanent** (always available) or **dynamic** (conditionally available based on configuration).

> **⚠️ Tools are part of the gateway source — adding one means building your own image.**
> The published `longmen-gateway` image ships with the built-in tool set only.
> There is no plugin system or runtime tool loading: a new tool is new Python
> code in `src/assistant/gateway/tools/` plus registry/planner wiring, so it only
> takes effect after the gateway is **rebuilt from source**. To run a custom tool
> you must clone this repository, apply the changes below, and build your own
> image — typically `FROM longmen-gateway` (the documented extension pattern) or a
> full rebuild via `gateway/Dockerfile`. Then point your deployment at that image
> (`GATEWAY_IMAGE` in `deploy/.env`). The stock image cannot pick up new tools.

## Table of Contents

1. [Tool Types](#tool-types)
2. [Core Interface](#core-interface)
3. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
4. [Tool Registry](#tool-registry)
5. [Configuration and Dynamic Tools](#configuration-and-dynamic-tools)
6. [Security and Sandboxing](#security-and-sandboxing)
7. [Common Patterns and Best Practices](#common-patterns-and-best-practices)
8. [Places Where Code Changes Are Needed](#places-where-code-changes-are-needed)

---

## Tool Types

### Permanent Tools
- **Always available** in the tool registry
- Examples: `shell`, `read_file`, `write_file`, `git_status`, `build`, `run_tests`
- Defined in `_BASE_TOOL_REGISTRY` in `tools/__init__.py`
- Do not require external API keys or configuration

### Dynamic Tools
- **Conditionally available** based on configuration
- Examples: `web_search`, `web_fetch` (require Brave API key)
- Added/removed via `update_tool_registry()` when config changes
- Can be enabled/disabled at runtime without restart

---

## Core Interface

All tools inherit from `BaseTool` and must implement:

```python
# src/assistant/gateway/tools/base.py
class BaseTool(ABC):
    name: str  # Class attribute: unique tool identifier
    
    @abstractmethod
    async def execute(self, root_path: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            root_path: Project root directory (sandbox boundary)
            **kwargs: Tool-specific arguments from the model's function call
            
        Returns:
            dict with keys:
            - stdout: Success output (for model consumption)
            - stderr: Error output (for debugging)
            - exit_code: 0 for success, non-zero for failure
            - Optional: duration_ms, errors, failures, etc.
        """
    
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """
        Return OpenAI-format function schema.
        
        Returns:
            dict with structure:
            {
                "type": "function",
                "function": {
                    "name": str,
                    "description": str,
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...],
                    },
                },
            }
        """
```

### Return Value Convention

All tools follow a consistent return format:

```python
{
    "stdout": "Success output (formatted for model consumption)",
    "stderr": "Error messages (for debugging)",
    "exit_code": 0,  # 0 = success, non-zero = failure
    "duration_ms": 123,  # Optional: execution time
}
```

**Important**: The `stdout` field is what the model sees. Format it clearly with:
- Human-readable summaries
- Structured data when appropriate

---

## Step-by-Step Implementation Guide

### Step 1: Create the Tool File

Create a new file in `src/assistant/gateway/tools/`:

```python
# src/assistant/gateway/tools/my_new_tool.py
"""My new tool implementation."""

from __future__ import annotations

from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)


class MyNewTool(BaseTool):
    name = "my_new_tool"  # Must match the class name pattern
    
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "my_new_tool",
                "description": "Clear description of what this tool does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {
                            "type": "string",
                            "description": "Description of arg1",
                        },
                        "arg2": {
                            "type": "integer",
                            "description": "Description of arg2 with default",
                        },
                    },
                    "required": ["arg1"],  # Only required arguments
                },
            },
        }
    
    async def execute(
        self,
        root_path: str,
        arg1: str,
        arg2: int = 10,  # Default values for optional args
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the tool logic."""
        try:
            # Your implementation here
            result = await self._do_work(root_path, arg1, arg2)
            
            return {
                "stdout": f"Success! Result: {result}",
                "stderr": "",
                "exit_code": 0,
            }
            
        except Exception as e:
            log.error("my_new_tool.error", error=str(e))
            return {
                "stdout": f"Error: {e}",
                "stderr": str(e),
                "exit_code": 1,
            }
    
    async def _do_work(
        self,
        root_path: str,
        arg1: str,
        arg2: int,
    ) -> str:
        """Internal implementation."""
        # Your work here
        pass
```

### Step 2: Add Path Sandboxing

All tools must validate that paths stay within `root_path`:

```python
from .base import BaseTool

class MyTool(BaseTool):
    async def execute(
        self,
        root_path: str,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Use the inherited _safe_path method
        safe_path = self._safe_path(root_path, file_path)
        
        # Now you can safely use safe_path
        with open(safe_path, "r") as f:
            content = f.read()
```

The `_safe_path()` method (inherited from `BaseTool`):
- Resolves the path to absolute
- Validates it's within `root_path`
- Raises `ValueError` if path escape detected

### Step 3: Handle Errors Gracefully

Always catch and handle errors:

```python
async def execute(
    self,
    root_path: str,
    arg: str,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        # Try to do the work
        result = await self._work(arg)
        return {"stdout": result, "stderr": "", "exit_code": 0}
        
    except FileNotFoundError as e:
        return {
            "stdout": f"File not found: {arg}",
            "stderr": str(e),
            "exit_code": 1,
        }
    except PermissionError as e:
        return {
            "stdout": f"Permission denied: {arg}",
            "stderr": str(e),
            "exit_code": 1,
        }
    except Exception as e:
        log.error("tool.error", error=str(e))
        return {
            "stdout": f"Unexpected error: {e}",
            "stderr": str(e),
            "exit_code": 1,
        }
```

### Step 4: Add to Tool Registry

Edit `src/assistant/gateway/tools/__init__.py`:

```python
# At the top of the file, add import
from .my_new_tool import MyNewTool

# In _BASE_TOOL_REGISTRY (for permanent tools)
_BASE_TOOL_REGISTRY: dict[str, BaseTool] = {
    # ... existing tools ...
    "my_new_tool": MyNewTool(),
}

# For dynamic tools, add in update_tool_registry():
def update_tool_registry(..., my_dynamic_tool_enabled: bool = True) -> None:
    global _TOOL_REGISTRY
    
    if my_dynamic_tool_enabled:
        _TOOL_REGISTRY["my_new_tool"] = MyNewTool()
```

### Step 5: Update Schema Retrieval

The `get_schemas()` function in `__init__.py` automatically retrieves schemas from registered tools. No changes needed if you added the tool to the registry.

### Step 6: Rebuild the Gateway Image

Because tools are baked into the gateway source, your changes are only live once
you build and deploy a new image. The stock published image will not have your
tool.

```bash
# Option A — extend the published image (recommended for small additions)
#   Add your tool files on top of the released image, then rebuild.

# Option B — full rebuild from this repo
docker build -t my-longmen-gateway:dev -f gateway/Dockerfile gateway/
```

Then point your deployment at the rebuilt image — set `GATEWAY_IMAGE` in
`deploy/.env` (or your compose override) to the tag you just built and restart
the stack.

---

## Tool Registry

### Permanent Tools

Located in `src/assistant/gateway/tools/__init__.py`:

```python
# Build base TOOL_REGISTRY (tools that are always available)
_BASE_TOOL_REGISTRY: dict[str, BaseTool] = {
    "shell": ShellTool(),
    "read_file": ReadFileTool(),
    "list_dir": ListDirTool(),
    "grep": GrepTool(),
    "write_file": WriteFileTool(),
    "search_replace": SearchReplaceTool(),
    "tree": TreeTool(),
    "symbols": SymbolsTool(),
    "git_status": GitStatusTool(),
    "git_diff": GitDiffTool(),
    "git_log": GitLogTool(),
    "git_add": GitAddTool(),
    "git_commit": GitCommitTool(),
    "detect_project": ProjectDetectTool(),
    "build": BuildTool(),
    "run_tests": TestRunnerTool(),
    "run_app": AppRunnerTool(),
    "sql_query": SQLQueryTool(),
    "my_new_tool": MyNewTool(),  # Add your tool here
}
```

### Dynamic Tools

Added via `update_tool_registry()`:

```python
def update_tool_registry(
    web_config_brave_key: str = "",
    search_enabled: bool = True,
    fetch_enabled: bool = True,
    my_dynamic_tool_enabled: bool = True,
    ...
) -> None:
    """Update TOOL_REGISTRY based on configuration."""
    global _TOOL_REGISTRY, KNOWN_TOOLS, TOOL_REGISTRY
    
    _TOOL_REGISTRY = dict(_BASE_TOOL_REGISTRY)
    
    # Add permanent tools first
    # (already done by copying _BASE_TOOL_REGISTRY)
    
    # Add dynamic tools based on configuration
    if search_enabled and brave_key:
        _TOOL_REGISTRY["web_search"] = WebSearchTool(...)
    
    if fetch_enabled:
        _TOOL_REGISTRY["web_fetch"] = WebFetchTool(...)
    
    if my_dynamic_tool_enabled:
        _TOOL_REGISTRY["my_new_tool"] = MyDynamicTool(...)
    
    KNOWN_TOOLS = set(_TOOL_REGISTRY.keys())
    TOOL_REGISTRY = _TOOL_REGISTRY
```

### Exported Functions

```python
def get_tool_registry() -> dict[str, BaseTool]:
    """Return the current TOOL_REGISTRY."""
    return _TOOL_REGISTRY

def get_known_tools() -> set[str]:
    """Return the current KNOWN_TOOLS set."""
    return KNOWN_TOOLS

def get_schemas(tool_names: list[str]) -> list[dict[str, Any]]:
    """Return OpenAI-format function schemas for the requested tools."""
    return [TOOL_REGISTRY[n].schema() for n in tool_names if n in TOOL_REGISTRY]

async def execute_tool(name: str, root_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}
    return await tool.execute(root_path, **arguments)
```

---

## Configuration and Dynamic Tools

### Config File (gateway.toml)

Add configuration options for dynamic tools:

```toml
# gateway.toml
[web]
brave_api_key = ""
search_enabled = true
fetch_enabled = true
search_count = 5
fetch_timeout = 15
fetch_max_redirects = 5
fetch_blocked_domains = []
user_agent = "Mozilla/5.0 ..."

# For your new dynamic tool:
[my_tool]
enabled = true
api_key = ""  # Optional API key
timeout = 30
```

### Server Initialization

In `src/assistant/gateway/server.py`:

```python
from .tools import update_tool_registry

async def main():
    config = GatewayConfig.from_toml(config_path)
    
    # Initialize tool registry
    update_tool_registry(
        web_config_brave_key=config.web.brave_api_key,
        search_enabled=config.web.search_enabled,
        fetch_enabled=config.web.fetch_enabled,
        my_dynamic_tool_enabled=config.my_tool.enabled,
        my_tool_timeout=config.my_tool.timeout,
        ...
    )
```

### Hot Reload

When config changes, `config_watcher.py` triggers reload:

```python
async def on_config_reload(..., changed: list[str]) -> None:
    if "my_tool" in changed:
        await close_web_client()  # If needed
        update_tool_registry(
            my_dynamic_tool_enabled=new_config.my_tool.enabled,
            my_tool_timeout=new_config.my_tool.timeout,
            ...
        )
        log.info("tool_registry_updated", ...)
```

---

## Security and Sandboxing

### Path Validation

All file/directory operations must use `_safe_path()`:

```python
class MyTool(BaseTool):
    async def execute(
        self,
        root_path: str,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # This validates the path is within root_path
        safe_path = self._safe_path(root_path, file_path)
        
        # Now safe to use
        with open(safe_path, "r") as f:
            return {"stdout": f.read(), "stderr": "", "exit_code": 0}
```

### Command Injection Prevention

For shell commands, never interpolate user input directly:

```python
# ❌ BAD - vulnerable to command injection
await asyncio.create_subprocess_shell(
    f"rm -rf {user_input}"
)

# ✅ GOOD - use list form
await asyncio.create_subprocess_exec(
    "rm", "-rf", user_input  # Still needs validation!
)

# ✅ BEST - validate and sanitize first
safe_path = self._safe_path(root_path, user_input)
await asyncio.create_subprocess_exec("rm", "-rf", safe_path)
```

### Private Network Blocking

For web tools, use `WebClient` which provides:
- Private IP blocking (localhost, 192.168.x.x, etc.)
- Domain blacklist
- `.local` domain blocking

```python
from .web_client import get_web_client, PrivateNetworkError, BlockedDomainError

async def execute(self, root_path: str, url: str, **kwargs: Any):
    try:
        client = get_web_client(user_agent="...")
        response = await client.fetch_url(url)
        return {"stdout": response.content, "stderr": "", "exit_code": 0}
    except PrivateNetworkError as e:
        return {"stdout": f"Blocked: {e}", "stderr": str(e), "exit_code": 1}
    except BlockedDomainError as e:
        return {"stdout": f"Blocked: {e}", "stderr": str(e), "exit_code": 1}
```

---

## Common Patterns and Best Practices

### Pattern 1: Wrapper Tools

Wrap existing tools for specific use cases:

```python
# src/assistant/gateway/tools/build.py
from .shell import ShellTool

_shell = ShellTool()

class BuildTool(BaseTool):
    name = "build"
    
    async def execute(self, root_path: str, command: str | None = None, ...):
        # Get project type
        pt = get_cached_project_type(root_path)
        
        # Determine build command
        if command:
            build_cmd = command
        elif pt and pt.build_cmd:
            build_cmd = pt.build_cmd
        else:
            return {"stdout": "No build command", "stderr": "", "exit_code": 1}
        
        # Execute via ShellTool
        result = await _shell.execute(root_path, command=build_cmd)
        
        # Parse and format output
        errors, warnings = _parse_errors(pt.types, result["stdout"])
        
        return {
            "stdout": _format_result(result, errors, warnings),
            "stderr": "",
            "exit_code": result["exit_code"],
            "errors": errors,
            "warnings": warnings,
        }
```

### Pattern 2: Multi-Tool Operations

Coordinate multiple tool calls:

```python
class RefactorTool(BaseTool):
    name = "refactor"
    
    async def execute(
        self,
        root_path: str,
        file_path: str,
        transformation: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # 1. Read file
        read_tool = ReadFileTool()
        read_result = await read_tool.execute(root_path, path=file_path)
        
        if read_result["exit_code"] != 0:
            return read_result
        
        # 2. Apply transformation
        new_content = self._apply_transformation(read_result["stdout"], transformation)
        
        # 3. Write file
        write_tool = WriteFileTool()
        write_result = await write_tool.execute(
            root_path,
            path=file_path,
            content=new_content,
        )
        
        return write_result
```

### Pattern 3: Caching

Use caching for expensive operations:

```python
from functools import lru_cache

class ProjectDetectTool(BaseTool):
    name = "detect_project"
    
    @lru_cache(maxsize=100)
    def _detect_project(self, root_path: str) -> ProjectType | None:
        # Expensive detection logic
        pass
    
    async def execute(self, root_path: str, **kwargs: Any):
        result = self._detect_project(root_path)
        return {"stdout": str(result), "stderr": "", "exit_code": 0}
```

---

## Places Where Code Changes Are Needed

### 1. Tool Implementation File

**File**: `src/assistant/gateway/tools/my_new_tool.py`

**Changes**:
- Create new file with `BaseTool` subclass
- Implement `name`, `schema()`, `execute()`
- Add path sandboxing
- Add error handling

### 2. Tool Registry

**File**: `src/assistant/gateway/tools/__init__.py`

**Changes**:
- Add import: `from .my_new_tool import MyNewTool`
- Add to `_BASE_TOOL_REGISTRY` (permanent) or `update_tool_registry()` (dynamic)
- Example:
  ```python
  from .my_new_tool import MyNewTool
  
  _BASE_TOOL_REGISTRY = {
      ...
      "my_new_tool": MyNewTool(),
  }
  ```

### 3. Agent Registry Validation

**File**: `src/assistant/gateway/agent_registry.py`

**Changes**:
- No changes needed if using `get_known_tools()`
- Validation automatically checks against `known_tools` set
- Agents can only use tools in the registry

### 4. Planner Tool Selection

**File**: `src/assistant/gateway/planner.py`

**Changes**:
- If tool should be available in discovery phase, add to `_DISCOVERY_ALLOWED_TOOLS`:
  ```python
  _DISCOVERY_ALLOWED_TOOLS = frozenset({
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
      "my_new_tool",  # Add here if discovery-allowed
  })
  ```
- If tool should be blocked in discovery, add to `_DISCOVERY_BLOCKED_TOOLS`:
  ```python
  _DISCOVERY_BLOCKED_TOOLS = frozenset({
      "write_file",
      "search_replace",
      "build",
      "run_tests",
      "run_app",
      "sql_query",
      "git_add",
      "git_commit",
      "my_new_tool",  # Add here if blocked in discovery
  })
  ```

### 5. Triage Phase Controls

**File**: `src/assistant/gateway/planner.py`

**Changes**:
- If tool counts against read budget (triage phase), add to `_TRIAGE_READ_TOOLS`:
  ```python
  _TRIAGE_READ_TOOLS = frozenset({
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
      "my_new_tool",  # Add if read-only and should count against budget
  })
  ```
- If tool should be blocked during triage, add to `_TRIAGE_BLOCKED_TOOLS`:
  ```python
  _TRIAGE_BLOCKED_TOOLS = frozenset({"build", "run_tests", "my_new_tool"})
  ```

### 6. Server Initialization

**File**: `src/assistant/gateway/server.py`

**Changes**:
- If dynamic tool, add parameters to `update_tool_registry()` call:
  ```python
  update_tool_registry(
      web_config_brave_key=config.web.brave_api_key,
      search_enabled=config.web.search_enabled,
      fetch_enabled=config.web.fetch_enabled,
      my_dynamic_tool_enabled=config.my_tool.enabled,
      my_tool_timeout=config.my_tool.timeout,
      ...
  )
  ```
- If tool needs cleanup on shutdown, add to close functions:
  ```python
  # At shutdown
  if hasattr(tool, "close"):
      await tool.close()
  ```

### 7. Configuration

**File**: `gateway.toml`

**Changes**:
- Add configuration section for dynamic tools:
  ```toml
  [my_tool]
  enabled = true
  api_key = ""  # Optional
  timeout = 30
  ```

### 8. Permissions

**File**: `src/assistant/gateway/permissions.py`

**Changes**:
- If tool should be auto-approved (safe), add to `default_safe`:
  ```toml
  # permissions.toml or config
  [permissions]
  workflow_mode = "prompt"
  default_safe = ["read_file", "list_dir", "my_new_tool"]
  default_destructive = ["shell", "run_app"]
  ```

### 9. Tool Interceptors (Optional)

**File**: `src/assistant/gateway/agent_loop.py`

**Changes**:
- If you need to intercept tool calls for special handling:
  ```python
  # In run_agent_loop()
  tool_interceptors = {
      "my_new_tool": my_tool_interceptor,
  }
  
  async def my_tool_interceptor(arguments: dict[str, Any]) -> dict[str, Any]:
      # Custom logic before tool execution
      return {"__stop__": True, "result": "custom result"}
  ```

---

## Checklist for Adding a New Tool

- [ ] Create tool file in `src/assistant/gateway/tools/`
- [ ] Implement `BaseTool` subclass with `name`, `schema()`, `execute()`
- [ ] Add path sandboxing using `_safe_path()`
- [ ] Add error handling and logging
- [ ] Add to tool registry (`__init__.py`)
- [ ] Update `update_tool_registry()` if dynamic
- [ ] Add to `_DISCOVERY_ALLOWED_TOOLS` or `_DISCOVERY_BLOCKED_TOOLS` if needed
- [ ] Add to `_TRIAGE_READ_TOOLS` or `_TRIAGE_BLOCKED_TOOLS` if needed
- [ ] Add configuration to `gateway.toml` if dynamic
- [ ] Add to permissions config if needed
- [ ] **Rebuild the gateway image** and point your deployment at it (`GATEWAY_IMAGE`)

---

## Reference Existing Tools

For examples of how to implement tools, refer to existing implementations in the `tools/` directory:

| Tool File | Use Case |
|-----------|----------|
| `shell.py` | Execute shell commands with timeout handling |
| `web_search.py` | External API calls with error handling |
| `web_fetch.py` | URL fetching with private network blocking |
| `build.py` | Wrapper tool that parses build output |
| `test_runner.py` | Test execution with framework-specific parsing |
| `file_read.py` | File operations with path sandboxing |
| `git.py` | Git operations with structured output |
| `sql.py` | Database queries with result formatting |

---

## Summary

Adding a new tool requires changes in these places:

1. **Tool implementation** (`tools/my_new_tool.py`)
2. **Tool registry** (`tools/__init__.py`)
3. **Planner constraints** (`planner.py`):
   - `_DISCOVERY_ALLOWED_TOOLS`
   - `_DISCOVERY_BLOCKED_TOOLS`
   - `_TRIAGE_READ_TOOLS`
   - `_TRIAGE_BLOCKED_TOOLS`
4. **Server initialization** (`server.py`)
5. **Configuration** (`gateway.toml`)
6. **Permissions** (`permissions.py` or `permissions.toml`)

For **dynamic tools**, also update:
- `update_tool_registry()` parameters
- Config watcher reload logic

Remember:
- Always use `_safe_path()` for path validation
- Follow the return value convention (`stdout`, `stderr`, `exit_code`)
- Document your tool in the schema description
- Refer to existing tools for implementation patterns

