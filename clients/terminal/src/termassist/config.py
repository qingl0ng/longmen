"""Client configuration loaded from ~/.longmen/terminal/config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

DEFAULT_CONFIG_PATH = Path.home() / ".longmen" / "terminal" / "config.toml"

_DEFAULT_TOML = """\
# Terminal Assistant Client Configuration

[gateway]
url = "ws://localhost:8420/ws"

[gateway.auth]
mode = "open"                    # "open" | "paired"
token = ""                       # JWT token from pairing (auto-populated)

[project]
id = ""                          # active project ID (set via /project or --project)
root_path = ""                   # informational — tools run on gateway host

[display]
theme = "default"                # "default" | "light" | "minimal"
show_thinking = false            # show model thinking/reasoning chunks
show_token_bar = true            # show context budget bar after responses
max_output_lines = 200           # max lines of tool output to display
auto_approve_safe = true         # auto-approve "safe" risk tool calls

[keybindings]
abort = "c-c"                    # abort current generation
submit = "enter"                 # submit prompt
multiline = "escape+enter"       # newline in prompt (Esc then Enter)

[logging]
level = "warning"                # debug | info | warning | error
file = ""                        # log file path (empty = no file logging)
message_log = ""                 # jsonl file for raw gateway messages (--debug sets this)
"""


@dataclass
class AuthConfig:
    mode: str = "open"
    token: str = ""


@dataclass
class GatewayConfig:
    url: str = "ws://localhost:8420/ws"
    auth: AuthConfig = field(default_factory=AuthConfig)


@dataclass
class ProjectConfig:
    id: str = ""
    root_path: str = ""


@dataclass
class DisplayConfig:
    theme: str = "default"
    show_thinking: bool = False
    show_token_bar: bool = True
    max_output_lines: int = 200
    auto_approve_safe: bool = True


@dataclass
class KeybindingsConfig:
    abort: str = "c-c"
    submit: str = "enter"
    multiline: str = "escape+enter"


@dataclass
class LoggingConfig:
    level: str = "warning"
    file: str = ""
    message_log: str = ""  # path for raw gateway message log (jsonl)


@dataclass
class ClientConfig:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    keybindings: KeybindingsConfig = field(default_factory=KeybindingsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    _path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None = None) -> ClientConfig:
        config_path = path or DEFAULT_CONFIG_PATH
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(_DEFAULT_TOML)
        raw = tomllib.loads(config_path.read_text())
        return cls._from_dict(raw, config_path)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any], path: Path) -> ClientConfig:
        gw_raw = raw.get("gateway", {})
        auth_raw = gw_raw.get("auth", {})
        auth = AuthConfig(
            mode=auth_raw.get("mode", "open"),
            token=auth_raw.get("token", ""),
        )
        gateway = GatewayConfig(
            url=gw_raw.get("url", "ws://localhost:8420/ws"),
            auth=auth,
        )

        proj_raw = raw.get("project", {})
        project = ProjectConfig(
            id=proj_raw.get("id", ""),
            root_path=proj_raw.get("root_path", ""),
        )

        disp_raw = raw.get("display", {})
        display = DisplayConfig(
            theme=disp_raw.get("theme", "default"),
            show_thinking=disp_raw.get("show_thinking", False),
            show_token_bar=disp_raw.get("show_token_bar", True),
            max_output_lines=disp_raw.get("max_output_lines", 200),
            auto_approve_safe=disp_raw.get("auto_approve_safe", True),
        )

        kb_raw = raw.get("keybindings", {})
        keybindings = KeybindingsConfig(
            abort=kb_raw.get("abort", "c-c"),
            submit=kb_raw.get("submit", "enter"),
            multiline=kb_raw.get("multiline", "escape+enter"),
        )

        log_raw = raw.get("logging", {})
        logging_cfg = LoggingConfig(
            level=log_raw.get("level", "warning"),
            file=log_raw.get("file", ""),
            message_log=log_raw.get("message_log", ""),
        )

        obj = cls(
            gateway=gateway,
            project=project,
            display=display,
            keybindings=keybindings,
            logging=logging_cfg,
        )
        obj._path = path
        return obj

    def save(self) -> None:
        raw: dict[str, Any] = {
            "gateway": {
                "url": self.gateway.url,
                "auth": {
                    "mode": self.gateway.auth.mode,
                    "token": self.gateway.auth.token,
                },
            },
            "project": {
                "id": self.project.id,
                "root_path": self.project.root_path,
            },
            "display": {
                "theme": self.display.theme,
                "show_thinking": self.display.show_thinking,
                "show_token_bar": self.display.show_token_bar,
                "max_output_lines": self.display.max_output_lines,
                "auto_approve_safe": self.display.auto_approve_safe,
            },
            "keybindings": {
                "abort": self.keybindings.abort,
                "submit": self.keybindings.submit,
                "multiline": self.keybindings.multiline,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "message_log": self.logging.message_log,
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(tomli_w.dumps(raw).encode())

    def set_value(self, key: str, value: str) -> None:
        """Set a config value by dotted key (e.g. 'display.theme')."""
        parts = key.split(".")
        if len(parts) == 2:
            section, attr = parts
            obj = getattr(self, section, None)
            if obj is None:
                raise KeyError(f"Unknown config section: {section!r}")
            if not hasattr(obj, attr):
                raise KeyError(f"Unknown config key: {key!r}")
            field_type = type(getattr(obj, attr))
            if field_type is bool:
                setattr(obj, attr, value.lower() in ("true", "1", "yes"))
            elif field_type is int:
                setattr(obj, attr, int(value))
            else:
                setattr(obj, attr, value)
        elif len(parts) == 3:
            section, subsection, attr = parts
            obj = getattr(self, section, None)
            if obj is None:
                raise KeyError(f"Unknown config section: {section!r}")
            sub = getattr(obj, subsection, None)
            if sub is None:
                raise KeyError(f"Unknown config subsection: {subsection!r}")
            if not hasattr(sub, attr):
                raise KeyError(f"Unknown config key: {key!r}")
            field_type = type(getattr(sub, attr))
            if field_type is bool:
                setattr(sub, attr, value.lower() in ("true", "1", "yes"))
            elif field_type is int:
                setattr(sub, attr, int(value))
            else:
                setattr(sub, attr, value)
        else:
            raise KeyError(f"Invalid config key format: {key!r}")
