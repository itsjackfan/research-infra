"""EXPERIMENT_LOG.md parser. Format (research-infra/EXPERIMENT_LOG.example.md):

## <exp> (branch: <exp>)
QUESTION: <question>
HYPOTHESIS: <hypothesis>

- YYYY-MM-DD HH:MM EST `<ablation>`: <goal to be established, what will cause that, and what we are testing>
<!-- NOTES
Changes:
Expected:
Results:
-->
    - YYYY-MM-DD HH:MM EST `<sub-ablation or setup>`: <one-liner: subablation, proof-of-concept, smoke, or setup>

A run is allowed iff its tag is on a `- ` line under its `## <exp>` heading and that section has non-empty QUESTION
and HYPOTHESIS lines. Unindented lines are ablations; indented lines are sub-ablations of the ablation above them.
Several tags per line are allowed; tags end at the first "`: ". HTML comments (NOTES) and fenced code blocks are
ignored. The launcher is the only automated writer: it appends a job link after the tag."""
import re
from dataclasses import dataclass
from pathlib import Path

TAG = re.compile(r"`([^`]+)`")
LINKS = re.compile(r"(?: \[[^\]]*\]\([^)]*\))*")


@dataclass
class Entry:
    exp: str
    ablations: tuple[str, ...]
    parent: str | None  # first tag of the enclosing ablation for a sub-ablation; None for an ablation
    line_no: int


@dataclass
class Section:
    exp: str
    question: str
    hypothesis: str
    entries: list[Entry]


def visible_lines(text: str):
    """(1-based line number, line) outside HTML comments and fenced code blocks."""
    in_comment = in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if not in_comment and stripped.startswith("```"):
            in_fence = not in_fence
        elif in_comment:
            in_comment = "-->" not in line
        elif not in_fence and stripped.startswith("<!--"):
            in_comment = "-->" not in line
        elif not in_fence:
            yield i, line


def parse(text: str) -> dict[str, Section]:
    sections, section, parent = {}, None, None
    for i, line in visible_lines(text):
        if line.startswith("## "):
            exp = line.removeprefix("## ").split()[0]
            section = sections[exp] = Section(exp, "", "", [])
            parent = None
        elif line.startswith("# "):
            section = None
        elif section is None:
            continue
        elif line.startswith("QUESTION:"):
            section.question = line.removeprefix("QUESTION:").strip()
        elif line.startswith("HYPOTHESIS:"):
            section.hypothesis = line.removeprefix("HYPOTHESIS:").strip()
        elif line.lstrip().startswith("- ") and (tags := tuple(TAG.findall(line.split("`: ")[0] + "`"))):
            sub = line[0].isspace()
            section.entries.append(Entry(section.exp, tags, parent if sub else None, i))
            if not sub:
                parent = tags[0]
    return sections


def find(log: Path, target: str) -> tuple[str, str]:
    """`<ablation>` or `<exp>/<ablation>` -> (exp, ablation). Missing tag, ambiguous tag, or missing
    QUESTION/HYPOTHESIS -> SystemExit with what to add."""
    exp, _, ablation = target.rpartition("/")
    hits = [s for s in parse(log.read_text()).values()
            if exp in ("", s.exp) and any(ablation in e.ablations for e in s.entries)]
    if not hits:
        raise SystemExit(f"`{ablation}` is not tagged in {log}. Add it under its `## <exp> (branch: <exp>)` heading:\n"
                         f"- YYYY-MM-DD HH:MM EST `{ablation}`: <goal, what will cause it, and what we are testing>")
    if len(hits) > 1:
        raise SystemExit(f"`{ablation}` is tagged in several sections of {log}; launch one of: "
                         + ", ".join(f"{s.exp}/{ablation}" for s in hits))
    (section,) = hits
    if not (section.question and section.hypothesis):
        raise SystemExit(f"`## {section.exp}` in {log} needs non-empty QUESTION: and HYPOTHESIS: lines before any run.")
    return section.exp, ablation


def append_link(log: Path, exp: str, ablation: str, label: str, url: str):
    """`<ablation>` -> `<ablation>` [<label>](<url>) on the entry's line, after any links already there."""
    text = log.read_text()
    entry = next(e for e in parse(text)[exp].entries if ablation in e.ablations)
    lines = text.splitlines(keepends=True)
    line = lines[entry.line_no - 1]
    end = LINKS.match(line, line.index(f"`{ablation}`") + len(ablation) + 2).end()
    lines[entry.line_no - 1] = f"{line[:end]} [{label}]({url}){line[end:]}"
    log.write_text("".join(lines))
