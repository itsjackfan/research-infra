import pytest

from research_infra import run_guard
from conftest import LOG


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "EXPERIMENT_LOG.md"
    path.write_text(LOG)
    return path


def test_parse_sections_and_entries():
    sections = run_guard.parse(LOG)
    assert set(sections) == {"probe", "other", "dup"}  # fenced template and comments ignored
    probe = sections["probe"]
    assert probe.question == "Does attention reuse hold across frames?" and probe.hypothesis == "Most patches are reused."
    assert [(e.ablations, e.parent) for e in probe.entries] == [
        (("base",), None), (("smoke", "smoke-gpu"), "base"), (("wide",), None)]
    lines = LOG.splitlines()
    assert all(f"`{e.ablations[0]}`" in lines[e.line_no - 1] for e in probe.entries)
    assert sections["other"].hypothesis == ""


def test_find(log):
    assert run_guard.find(log, "smoke-gpu") == ("probe", "smoke-gpu")
    assert run_guard.find(log, "wide") == ("probe", "wide")
    assert run_guard.find(log, "probe/base") == ("probe", "base")
    assert run_guard.find(log, "dup/base") == ("dup", "base")


@pytest.mark.parametrize("target, match", [
    ("base", "probe/base"),  # ambiguous: names the qualified forms
    ("nohyp", "HYPOTHESIS"),
    ("missing", "EXPERIMENT_LOG.md"),
    ("ghost", "EXPERIMENT_LOG.md"),  # only inside an HTML comment
    ("notes-tag", "EXPERIMENT_LOG.md"),  # only inside NOTES
    ("<ablation>", "EXPERIMENT_LOG.md"),  # only inside the fenced template
    ("other/base", "EXPERIMENT_LOG.md"),
])
def test_find_refuses(log, target, match):
    with pytest.raises(SystemExit, match=match):
        run_guard.find(log, target)


def test_append_link_after_tag(log):
    run_guard.append_link(log, "probe", "smoke", "123", "https://x/123")
    run_guard.append_link(log, "probe", "smoke", "124", "https://x/124")
    run_guard.append_link(log, "dup", "base", "9", "https://x/9")
    new, old = log.read_text().splitlines(), LOG.splitlines()
    changed = [(o, n) for o, n in zip(old, new) if o != n]
    assert len(new) == len(old) and changed == [
        ("    - 2026-11-26 11:00 EST `smoke` `smoke-gpu`: smoke the pipeline",
         "    - 2026-11-26 11:00 EST `smoke` [123](https://x/123) [124](https://x/124) `smoke-gpu`: smoke the pipeline"),
        ("- 2026-11-26 10:00 EST `base`: same tag, other section",
         "- 2026-11-26 10:00 EST `base` [9](https://x/9): same tag, other section"),
    ]
    assert run_guard.find(log, "smoke") == ("probe", "smoke")
