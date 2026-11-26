import sys
from pathlib import Path, PurePosixPath

import pytest
from omegaconf import OmegaConf

from research_infra import launch
from research_infra.launch import Cluster, expand
from research_infra.run_dir import LocalFS, mark_done
from conftest import CONFIG, git


class FakeCluster(LocalFS, Cluster):
    project, user = "proj", "me"

    def __init__(self, root):
        self.root, self.submitted, self.synced = PurePosixPath(root), [], []

    def check_synced(self, exp, commit):
        self.synced.append((exp, commit))

    def command(self, run):
        return f"submit {run.out_dir}/config.yaml"

    def submit(self, run):
        self.submitted.append(run)
        return str(len(self.submitted))

    def link(self, run, job_id):
        return f"https://jobs/{job_id}"


@pytest.fixture
def cluster(tmp_path):
    return FakeCluster(tmp_path / "cluster")


def test_expand():
    assert expand("base", "") == [("base", [])]
    assert expand("base", "a=1,2 b=x") == [("base_a=1", ["a=1", "b=x"]), ("base_a=2", ["a=2", "b=x"])]
    assert expand("base", "l=[1,2]") == [("base", ["l=[1,2]"])]
    assert [n for n, _ in expand("b", "a=1,2 c=3,4")] == ["b_a=1_c=3", "b_a=1_c=4", "b_a=2_c=3", "b_a=2_c=4"]


def test_paths(cluster, tmp_path):
    assert cluster.code_dir("probe") == PurePosixPath(tmp_path / "cluster/me/proj/code/probe")
    assert cluster.out_dir("probe", "base") == PurePosixPath(tmp_path / "cluster/me/proj/probe/outputs/base")


def test_git_info(repo, tmp_path):
    wt = tmp_path / "wt-probe"
    branch, commit, dirty = launch.git_info(wt)
    assert (branch, commit, dirty) == ("probe", git(repo, "rev-parse", "HEAD"), False)
    (wt / "untracked.txt").write_text("x")
    assert not launch.git_info(wt)[2]
    (wt / "configs/config.yaml").write_text("changed: 1\n")
    assert launch.git_info(wt)[2]


def test_find_worktree(repo, tmp_path):
    assert launch.find_worktree(repo, "probe").resolve() == (tmp_path / "wt-probe").resolve()
    assert launch.find_worktree(repo, "main").resolve() == repo.resolve()
    with pytest.raises(SystemExit, match="git worktree add"):
        launch.find_worktree(repo, "nope")


def test_check_clean(repo):
    launch.check_clean(repo)
    (repo / "src/scripts/tmp").mkdir(parents=True)
    (repo / "src/scripts/tmp/x.py").write_text("")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "scratch")
    with pytest.raises(SystemExit, match="src/scripts"):
        launch.check_clean(repo)


def test_launch_submits_and_links(repo, cluster, tmp_path):
    cluster.launch(repo, "probe/base", "lr=0.2")
    (run,) = cluster.submitted
    commit = git(tmp_path / "wt-probe", "rev-parse", "HEAD")
    assert run.exp == "probe" and run.name == "base" and run.worktree.resolve() == (tmp_path / "wt-probe").resolve()
    assert run.code_dir == cluster.code_dir("probe") and run.out_dir == cluster.out_dir("probe", "base")
    cfg = OmegaConf.load(Path(run.out_dir) / "config.yaml")
    assert cfg.lr == 0.2 and cfg.exp == "probe" and cfg.name == "base" and cfg.run_dir == str(run.out_dir)
    assert cfg.git.branch == "probe" and cfg.git.commit == commit
    assert cluster.synced == [("probe", commit)]
    assert "`base` [1](https://jobs/1): establish the baseline" in (repo / "EXPERIMENT_LOG.md").read_text()


def test_launch_sweep(repo, cluster):
    cluster.launch(repo, "smoke", "lr=0.2,0.3")
    assert [(r.name, r.cfg.lr) for r in cluster.submitted] == [("smoke_lr=0.2", 0.2), ("smoke_lr=0.3", 0.3)]
    assert "`smoke` [1](https://jobs/1) [2](https://jobs/2)" in (repo / "EXPERIMENT_LOG.md").read_text()


def test_launch_is_idempotent(repo, cluster):
    cluster.launch(repo, "wide")
    cluster.launch(repo, "wide")  # same config: resume
    assert len(cluster.submitted) == 2
    mark_done(Path(cluster.submitted[0].out_dir))
    cluster.launch(repo, "wide")  # DONE: no-op
    assert len(cluster.submitted) == 2
    with pytest.raises(SystemExit, match="--force-override"):
        cluster.launch(repo, "wide", "lr=0.5")
    cluster.launch(repo, "wide", "lr=0.5", force_override=True)
    assert len(cluster.submitted) == 3 and cluster.submitted[-1].cfg.lr == 0.5


def test_launch_refuses_dirty_worktree(repo, cluster, tmp_path):
    (tmp_path / "wt-probe/configs/config.yaml").write_text(CONFIG + "x: 1\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        cluster.launch(repo, "wide")
    assert cluster.submitted == []


def test_launch_refuses_untagged(repo, cluster):
    with pytest.raises(SystemExit):
        cluster.launch(repo, "untagged")
    assert cluster.submitted == []


def test_dry_run_writes_nothing(repo, cluster, capsys):
    cluster.launch(repo, "wide", dry_run=True)
    assert cluster.submitted == [] and not Path(cluster.out_dir("probe", "wide")).exists()
    out = capsys.readouterr().out
    assert str(cluster.code_dir("probe")) in out and "submit" in out


def test_load_cluster(repo, monkeypatch):
    (repo / "compute").mkdir()
    (repo / "compute/launch.py").write_text("CLUSTER = 'fasrc'\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for m in ("compute", "compute.launch"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    assert launch.load_cluster(repo) == "fasrc"
