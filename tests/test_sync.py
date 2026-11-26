import os
import stat
from pathlib import PurePosixPath

from research_infra import install_hooks, sync
from conftest import git


def test_write_gitinfo(repo, tmp_path):
    wt = tmp_path / "wt-probe"
    assert sync.write_gitinfo(wt) == "probe"
    assert (wt / ".gitinfo").read_text() == f"probe {git(wt, 'rev-parse', 'HEAD')}\n"


def test_rsync_cmd(tmp_path):
    dst = PurePosixPath("/n/lab/me/proj/code/probe")
    filters = ("--include=/third_party/x/keep.py", "--exclude=/third_party/x/*")
    cmd = sync.rsync_cmd(tmp_path, "fasrc", dst, filters)
    assert cmd[:4] == ["rsync", "-az", "--delete", "-e"] and cmd[4] == "ssh -o BatchMode=yes"
    assert f"--rsync-path=mkdir -p {dst} && rsync" in cmd
    rules = [a for a in cmd if a.startswith(("--include=", "--exclude="))]
    assert rules == [*filters, *[f"--exclude={e}" for e in sync.EXCLUDES]]  # project filters first: first match wins
    excludes = [a.removeprefix("--exclude=") for a in rules]
    assert {"/.env", ".git", "/.venv", "/outputs"} <= set(excludes) and "--delete-excluded" not in cmd
    assert cmd[-2:] == [f"{tmp_path}/", f"fasrc:{dst}/"]


def test_install_hooks(repo, tmp_path, monkeypatch):
    git(repo, "config", "core.hooksPath", "cluster/hooks")
    monkeypatch.chdir(tmp_path / "wt-probe")
    install_hooks.main()
    hooks = repo / ".git/hooks"
    for name, text in install_hooks.HOOKS.items():
        assert (hooks / name).read_text() == text
        assert os.stat(hooks / name).st_mode & stat.S_IXUSR
    assert "hookspath" not in git(repo, "config", "--list").lower()
