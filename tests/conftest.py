import subprocess
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

LOG = """# Experiment log
Format:

```
## <exp> (branch: <exp>)
QUESTION: <question>
HYPOTHESIS: <hypothesis>

- YYYY-MM-DD HH:MM EST `<ablation>`: <goal>
```

<!-- old
- 2026-01-01 00:00 EST `ghost`: commented out
-->

## probe (branch: probe)
QUESTION: Does attention reuse hold across frames?
HYPOTHESIS: Most patches are reused.

- 2026-11-26 10:00 EST `base`: establish the baseline
<!-- NOTES
Changes: `notes-tag`
Expected:
Results:
-->
    - 2026-11-26 11:00 EST `smoke` `smoke-gpu`: smoke the pipeline
- 2026-11-26 12:00 EST `wide`: wider model

## other (branch: other)
QUESTION: Something?
HYPOTHESIS:

- 2026-11-26 10:00 EST `nohyp`: missing hypothesis

## dup (branch: dup)
QUESTION: Q?
HYPOTHESIS: H.

- 2026-11-26 10:00 EST `base`: same tag, other section
"""

CONFIG = """exp: ???
name: ???
run_dir: ???
git:
  branch: ???
  commit: ???
wandb: {project: proj, entity: null, mode: disabled}
lr: 0.1
"""


def git(cwd, *args):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    """Project repo on main (configs/, EXPERIMENT_LOG.md) + worktree `probe` at tmp_path/wt-probe."""
    repo = tmp_path / "proj"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs/config.yaml").write_text(CONFIG)
    (repo / "EXPERIMENT_LOG.md").write_text(LOG)
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")
    git(repo, "worktree", "add", "-q", str(tmp_path / "wt-probe"), "-b", "probe")
    return repo


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create({"exp": "probe", "name": "base", "run_dir": str(tmp_path / "run"),
                             "git": {"branch": "probe", "commit": "abc123"},
                             "wandb": {"project": "proj", "entity": None, "mode": "disabled"}})


@pytest.fixture
def wandb_mock(monkeypatch):
    from research_infra import monitoring
    mock = MagicMock()
    monkeypatch.setattr(monitoring, "wandb", mock)
    return mock
