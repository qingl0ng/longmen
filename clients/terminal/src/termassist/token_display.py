"""Token budget bar rendering."""

from __future__ import annotations

from rich.text import Text

_BAR_WIDTH = 12
_FILLED = "\u2588"
_EMPTY = "\u2591"


def render_token_bar(used: int, limit: int, theme: dict[str, str]) -> Text:
    """Return a Rich Text object for the token budget bar."""
    if limit <= 0:
        return Text("")

    ratio = used / limit
    filled = round(ratio * _BAR_WIDTH)
    filled = max(0, min(filled, _BAR_WIDTH))
    empty = _BAR_WIDTH - filled

    bar_str = _FILLED * filled + _EMPTY * empty

    if ratio < 0.5:
        color = theme.get("token_bar_low", "green")
    elif ratio < 0.8:
        color = theme.get("token_bar_mid", "yellow")
    else:
        color = theme.get("token_bar_high", "red")

    used_k = f"{used // 1000}k" if used >= 1000 else str(used)
    limit_k = f"{limit // 1000}k" if limit >= 1000 else str(limit)

    text = Text()
    text.append("[ctx: ")
    text.append(bar_str, style=color)
    text.append(f" {used_k}/{limit_k}]")
    return text
