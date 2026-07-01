# longmen — Terminal Client

A thin async terminal client for the **Longmen** self-hosted coding assistant. It
connects to a Longmen **Gateway** over WebSocket, streams model output to the
terminal, and handles tool approval dialogs interactively.

> **Note:** This package is a client only. It does not run a language model or
> execute tools; it is a front-end that connects to a Longmen Gateway. The
> gateway communicates with a local vLLM model server and an optional RAG service.
> A running gateway is required. See the
> [main project](https://github.com/qingl0ng/longmen) for deployment instructions.

---

## Requirements

- **A running Longmen Gateway** — the server this client connects to. Deploy it
  from the [main project](https://github.com/qingl0ng/longmen) using Docker
  Compose. A reachable gateway is required for the client to function.
- **Python 3.12 or newer.**

---

## Install

`longmen` is a command-line application. Installation with
[pipx](https://pipx.pypa.io/) is recommended, as it isolates the application in a
dedicated environment and adds it to `PATH`:

```bash
# 1. Install pipx (once) — see https://pipx.pypa.io/stable/installation/
python3 -m pip install --user pipx
python3 -m pipx ensurepath
#    Restart the shell afterwards so the updated PATH takes effect.

# 2. Install the client
pipx install longmen
```

Alternatively, install with pip into the current environment:

```bash
pip install longmen
```

Both methods provide the `longmen` command.

### From source (development)

```bash
cd clients/terminal
poetry install
```

This creates a virtualenv at `clients/terminal/.venv` and installs all
dependencies including dev tools. Prefix the commands below with `poetry run`
when working from source.

---

## Run

```bash
# Interactive mode (uses ~/.longmen/terminal/config.toml)
longmen

# Override gateway URL
longmen --gateway ws://localhost:8420/ws

# Auto-select a project
longmen --project my-project

# Minimal theme (no colors)
longmen --theme minimal

# Use a specific config file
longmen --config /path/to/config.toml

# Print version
longmen --version
```

---

## Configuration

On first run, `~/.longmen/terminal/config.toml` is created with defaults:

```toml
[gateway]
url = "ws://localhost:8420/ws"

[gateway.auth]
mode = "open"      # authentication mode
token = ""         # auth token

[project]
id = ""            # set via /project or --project flag
root_path = ""     # informational only — tools run on the gateway host

[display]
theme = "default"           # "default" | "light" | "minimal"
show_thinking = false       # show model thinking/reasoning chunks
show_token_bar = true       # context usage bar after each response
max_output_lines = 200      # truncate long tool output in the display
auto_approve_safe = true    # auto-approve "safe"-risk tool calls

[keybindings]
abort = "c-c"           # abort in-progress generation
submit = "enter"        # submit prompt
multiline = "escape+enter"  # insert newline without submitting (Esc then Enter)

[logging]
level = "warning"
file = ""           # path for file logging (empty = disabled)
message_log = ""    # jsonl file for raw gateway messages
```

The config lives at `~/.longmen/terminal/config.toml` on every platform — it
resolves against your home directory: `/home/<you>/…` on Linux,
`/Users/<you>/…` on macOS, and `C:\Users\<you>\.longmen\terminal\config.toml`
on Windows. Point at a different file with `--config /path/to/config.toml`.

You can also edit config live with the `/config` slash command:

```
/config theme=minimal
/config display.show_thinking=true
/config display.max_output_lines=50
```

---

## Slash Commands

| Command | Where it runs | What it does |
|---------|---------------|--------------|
| `/help` | client | List all commands |
| `/new` | gateway | Start a new conversation |
| `/compact` | gateway | Compact conversation context |
| `/refresh` | gateway | Reload project context |
| `/permissions` | gateway | Show/manage stored tool permissions |
| `/config [key=val]` | client | Show or update config |
| `/project [id]` | gateway | List, switch, or add (`/project add <id> <root_path> [description]`) |
| `/status` | gateway | Request gateway status |
| `/prompt` | gateway | Show the current system prompt |
| `/quit` | client | Graceful disconnect and exit |

Unknown commands show fuzzy-matched suggestions.

---

## File Attachments

Prefix a file path with `@` anywhere in a prompt:

```
> Explain what this does @config.toml
> What does this diagram show? @architecture.png
> Compare these two files @old.py @new.py
> Check this @/absolute/path/to/file.cpp
> Review @~/notes/design.md
```

Rules:
- `@` must be at the start of the input or preceded by whitespace — prevents false matches in email addresses
- Text files are sent as `file` content blocks (base64)
- Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`) are sent as `image` content blocks
- Files over 1 MB trigger a warning; files over 10 MB are rejected

---

## Approval Dialog

> **Note:** the interactive approval dialog applies to the gateway's `prompt`
> workflow mode, which is **still under development**. The only supported mode
> today is `allow_all`, where tool calls execute automatically and no dialog is
> shown. The behaviour below describes the in-progress `prompt` mode.

When the model wants to run a tool, an approval dialog is shown:

```
┌─ Shell Command (moderate) ────────────────────────────────────┐
│ pytest tests/auth/ -v                                         │
│ Risk: moderate                                                │
│ Context: Running test suite for auth module                   │
├───────────────────────────────────────────────────────────────┤
│ [y] Yes — run this command                                    │
│ [n] No — skip                                                 │
│ [s] Yes, remember for this session                            │
│ [a] Always allow (persist across sessions)                    │
│ [e] Edit command before running                               │
└───────────────────────────────────────────────────────────────┘
```

Single keypress — no Enter needed. With `auto_approve_safe = true` (the default), `safe`-risk tool calls (read-only operations) are approved automatically with a brief notice.

---

## Troubleshooting

### "Connection refused" on startup

The Gateway isn't running or is on a different address. Check:
1. Start the gateway stack — see the
   [main project](https://github.com/qingl0ng/longmen) (Docker Compose). The
   gateway listens on `127.0.0.1:8420` by default.
2. Verify the URL in config matches: `grep url ~/.longmen/terminal/config.toml`
3. Override: `longmen --gateway ws://correct-host:8420/ws`

### AttributeError: 'ClientConnection' has no attribute 'closed'

websockets 15+ renamed the connection class and removed the `.closed` property. `connection.py` uses `ws.state in (WsState.CLOSING, WsState.CLOSED)` via `websockets.protocol.State`. If you upgrade websockets, verify this import still resolves:

```python
from websockets.protocol import State as WsState
```

### Crash logs

Unhandled exceptions are written to `~/.longmen/terminal/crash.log`.

### Config file errors

If the config file has a syntax error, `longmen` prints the parse error and exits. Delete or fix `~/.longmen/terminal/config.toml` — a new default is created on next start.

---

## Project Structure

```
clients/terminal/
├── pyproject.toml             # Poetry project — deps, scripts, tooling config
├── poetry.lock
└── src/termassist/
    ├── __init__.py            # __version__
    ├── cli.py                 # Entry point — argparse → App.run()
    ├── app.py                 # Main loop — two concurrent asyncio tasks
    ├── connection.py          # WebSocket client — connect, reconnect, send/recv
    ├── protocol.py            # Client-side message builders + parse_message()
    ├── state.py               # State machine — idle/streaming/awaiting_approval/etc.
    ├── config.py              # Config loading/saving (TOML)
    ├── renderer.py            # Rich-based output rendering
    ├── input_handler.py       # prompt-toolkit input with tab completion
    ├── approval.py            # Interactive approval dialog
    ├── slash_commands.py      # Slash command registry and parser
    ├── file_attach.py         # @path syntax → base64 content blocks
    ├── token_display.py       # Context budget bar
    ├── history.py             # In-memory conversation display cache
    └── themes.py              # Color theme definitions
```
