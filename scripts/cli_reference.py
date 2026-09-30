"""Write docs/reference/cli.md from the argument parser itself: every command, what it does, and
every option with its default, so the reference cannot drift from the CLI. The tables come from the
parser's actions, not from ``--help`` text, whose layout differs between Python versions.

    uv run python scripts/cli_reference.py --write    # after changing a command or an option
    uv run python scripts/cli_reference.py --check    # what the tests do
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from toolrank.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "reference" / "cli.md"

HEAD = """# CLI reference

Generated from `toolrank`'s argument parser by `scripts/cli_reference.py`; `toolrank <command>
--help` shows the same. Every command also takes `-h` / `--help`.
"""


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _help(parser: argparse.ArgumentParser, name: str, parent: argparse.ArgumentParser) -> str:
    for action in parent._actions:
        if isinstance(action, argparse._SubParsersAction):
            for choice in action._choices_actions:
                if choice.dest == name:
                    return choice.help or ""
    return ""


def _option(action: argparse.Action) -> tuple[str, str, str]:
    """(option, default, description) of one argument."""
    if action.option_strings:
        name = ", ".join(action.option_strings)
        if action.nargs != 0:
            metavar = action.metavar or (action.choices and "{" + ",".join(map(str, action.choices)) + "}")
            name += f" {metavar or action.dest.upper()}"
    else:
        name = action.metavar or action.dest
    name = f"`{name}`"
    if action.required or not action.option_strings:
        default = "required"
    elif action.nargs == 0 or action.default in (None, "", [], False, argparse.SUPPRESS):
        default = ""
    else:
        default = f"`{action.default}`"
    text = (action.help or "") % {"default": action.default, "prog": ""} if action.help else ""
    if isinstance(action, argparse._AppendAction):
        text = (text + " (repeatable)").strip()
    return name, default, _cell(text)


def _command(path: list[str], parser: argparse.ArgumentParser, about: str) -> list[str]:
    lines = [f"## `{' '.join(path)}`", ""]
    text = parser.description or about
    if text:
        lines += [_cell(text), ""]
    rows = [
        _option(a)
        for a in parser._actions
        if not isinstance(a, argparse._HelpAction | argparse._SubParsersAction)
        and a.help != argparse.SUPPRESS
    ]
    if rows:
        lines += ["| Argument | Default | Description |", "| --- | --- | --- |"]
        lines += [f"| {n} | {d} | {h} |" for n, d, h in rows]
        lines.append("")
    return lines


def render() -> str:
    root = build_parser()
    lines = [HEAD]

    def walk(path: list[str], parser: argparse.ArgumentParser, about: str) -> None:
        subs = _subcommands(parser)
        if not subs:  # a command, not a group of them
            lines.extend(_command(path, parser, about))
        for name, sub in subs.items():
            walk([*path, name], sub, _help(sub, name, parser))

    walk(["toolrank"], root, "")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    a = p.parse_args()
    text = render()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUT.relative_to(ROOT)} is stale: uv run python scripts/cli_reference.py --write")
        print(f"{OUT.relative_to(ROOT)} is current")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
