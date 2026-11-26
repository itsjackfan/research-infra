"""Shared launcher. Humans only, from any checkout of the project:
`uv run python -m research_infra.launch <ablation|exp/ablation> "<hydra overrides>" [--force-override] [--dry-run]`

The ablation's `## <exp>` heading in EXPERIMENT_LOG.md gives the branch; the launcher composes the config in that
branch's local worktree and runs the cluster commands from the synced copy of it, so no `cd` is needed.
Comma values (`train.lr=1e-3,1e-4`) expand into a sweep; each config is named `<ablation>_<key>=<val>`.

Each project defines `CLUSTER = <its Cluster subclass>()` in compute/launch.py.
Paths: code    <root>/<user>/<project>/code/<exp>/
       outputs <root>/<user>/<project>/<exp>/outputs/<ablation>/"""
import argparse
import importlib
import itertools
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from omegaconf import DictConfig, OmegaConf

from research_infra import run_guard
from research_infra.hydra_compose import compose
from research_infra.run_dir import prepare_run_dir


@dataclass
class Run:
    exp: str
    name: str  # <ablation>[_<key>=<val>...]
    overrides: list[str]
    worktree: Path
    code_dir: PurePosixPath
    out_dir: PurePosixPath
    cfg: DictConfig | None = None
    args: argparse.Namespace = field(default_factory=argparse.Namespace)


class Cluster(ABC):
    project: str
    root: PurePosixPath
    user: str
    host: str | None = None  # ssh alias for sync; None = nothing to sync (code ships from this machine)
    rsync_filters: tuple[str, ...] = ()  # raw rsync filter args, placed before sync.EXCLUDES (first match wins)
    config_name = "config"

    def code_dir(self, exp) -> PurePosixPath:
        return self.root / self.user / self.project / "code" / exp

    def out_dir(self, exp, name) -> PurePosixPath:
        return self.root / self.user / self.project / exp / "outputs" / name

    def add_args(self, parser: argparse.ArgumentParser):
        pass

    def compose(self, run: Run) -> DictConfig:
        return compose(run.worktree / "configs", self.config_name, run.overrides)

    def launch(self, repo: Path, target: str, overrides="", force_override=False, dry_run=False, args=None):
        log = repo / "EXPERIMENT_LOG.md"
        exp, ablation = run_guard.find(log, target)
        check_clean(repo)
        worktree = find_worktree(repo, exp)
        branch, commit, dirty = git_info(worktree)
        if dirty:
            raise SystemExit(f"{worktree} has uncommitted changes; commit (and sync) before launching.")
        self.check_synced(exp, commit)
        for name, combo in expand(ablation, overrides):
            run = Run(exp, name, combo, worktree, self.code_dir(exp), self.out_dir(exp, name), args=args)
            run.cfg = self.compose(run)
            run.cfg.exp, run.cfg.name, run.cfg.run_dir = exp, name, str(run.out_dir)
            run.cfg.git.branch, run.cfg.git.commit = branch, commit
            if dry_run:
                print(f"[dry-run] {name}: cwd={run.code_dir} out={run.out_dir}\n{self.command(run)}")
                continue
            config_yaml = OmegaConf.to_yaml(OmegaConf.to_container(run.cfg, resolve=True, throw_on_missing=True))
            if prepare_run_dir(run.out_dir, config_yaml, force_override, fs=self) == "done":
                print(f"{name}: already DONE in {run.out_dir}; nothing to do.")
                continue
            job_id = self.submit(run)
            link = self.link(run, job_id)
            run_guard.append_link(log, exp, ablation, job_id, link)
            print(f"{name}: submitted {job_id} -> {link}")

    @abstractmethod
    def check_synced(self, exp: str, commit: str):
        """SystemExit unless the cluster's copy of <exp> is at `commit`."""

    @abstractmethod
    def command(self, run: Run) -> str:
        """The exact submit command, run from run.code_dir (printed by --dry-run, executed by submit)."""

    @abstractmethod
    def submit(self, run: Run) -> str:
        """Submit one run from run.code_dir; return its job / call id."""

    @abstractmethod
    def link(self, run: Run, job_id: str) -> str:
        """Where to look at this job (URL or command); appended to EXPERIMENT_LOG.md after the ablation's tag."""

    # research_infra.run_dir.FS on the cluster's filesystem
    @abstractmethod
    def read_text(self, path) -> str | None: ...

    @abstractmethod
    def write_text(self, path, text: str): ...

    @abstractmethod
    def rmtree(self, path): ...


def expand(ablation: str, overrides: str) -> list[tuple[str, list[str]]]:
    """"a=1,2 b=x" -> [("<abl>_a=1", ["a=1", "b=x"]), ("<abl>_a=2", ["a=2", "b=x"])]; list values "[..]" are not split."""
    axes = [[f"{k}={v}" for v in ([vs] if vs.startswith("[") else vs.split(","))]
            for k, vs in (o.split("=", 1) for o in overrides.split())]
    return [("_".join([ablation] + [o for o, axis in zip(combo, axes) if len(axis) > 1]), list(combo))
            for combo in itertools.product(*axes)]


def git(path, *args) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def git_info(path: Path) -> tuple[str, str, bool]:
    """(branch, commit, dirty) of the checkout at `path`; untracked files do not count as dirty."""
    return (git(path, "rev-parse", "--abbrev-ref", "HEAD"), git(path, "rev-parse", "HEAD"),
            bool(git(path, "status", "--porcelain", "--untracked-files=no")))


def find_worktree(repo: Path, branch: str) -> Path:
    """Local checkout of `branch` (`git worktree list --porcelain`); SystemExit if none."""
    for block in git(repo, "worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        if fields.get("branch") == f"refs/heads/{branch}":
            return Path(fields["worktree"])
    raise SystemExit(f"No local checkout of branch {branch}; create it with "
                     f"`git worktree add .claude/worktrees/{branch} -b {branch}`.")


def check_clean(repo: Path):
    """Protocol gate: main must hold no scratch code (src/scripts/) before any new run."""
    if git(repo, "ls-tree", "-r", "--name-only", "main", "--", "src/scripts"):
        raise SystemExit("main still holds scratch code in src/scripts/; document its results and delete it before a new run.")


def load_cluster(repo: Path) -> Cluster:
    sys.path.insert(0, str(repo))
    return importlib.import_module("compute.launch").CLUSTER


def main(argv=None):
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    cluster = load_cluster(repo)
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="<ablation> or <exp>/<ablation>, as tagged in EXPERIMENT_LOG.md")
    parser.add_argument("overrides", nargs="?", default="")
    parser.add_argument("--force-override", action="store_true", help="delete an existing run dir with a different config")
    parser.add_argument("--dry-run", action="store_true")
    cluster.add_args(parser)
    args = parser.parse_args(argv)
    cluster.launch(repo, args.target, args.overrides, args.force_override, args.dry_run, args)


if __name__ == "__main__":
    main()
