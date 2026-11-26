"""Idempotent run dirs. A run is (exp, ablation, resolved config incl. git commit); config.yaml is its identity.

no config.yaml                     -> "new"     (write config.yaml)
same config.yaml, no DONE          -> "resume"  (trainer loads its latest checkpoint if any)
same config.yaml, DONE             -> "done"    (no-op)
different config.yaml              -> SystemExit, unless force_override: delete the dir, then "new"

Called by the launcher (on the cluster's filesystem, before submit) and by every runner at job start (local
filesystem), so a SLURM requeue resumes and a relaunch of a finished run does nothing."""
import shutil
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

State = Literal["new", "resume", "done"]


class FS(Protocol):
    def read_text(self, path) -> str | None: ...
    def write_text(self, path, text: str): ...
    def rmtree(self, path): ...
    def symlink(self, target, link): ...


class LocalFS:
    def read_text(self, path):
        path = Path(path)
        return path.read_text() if path.exists() else None

    def write_text(self, path, text):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)

    def rmtree(self, path):
        shutil.rmtree(path)

    def symlink(self, target, link):
        """mkdir target; link -> target unless link already exists."""
        Path(target).mkdir(parents=True, exist_ok=True)
        if not Path(link).exists():
            Path(link).parent.mkdir(parents=True, exist_ok=True)
            Path(link).symlink_to(target)


def prepare_run_dir(run_dir, config_yaml: str, force_override=False, fs: FS = LocalFS()) -> State:
    run_dir = PurePosixPath(run_dir)
    existing = fs.read_text(run_dir / "config.yaml")
    if existing == config_yaml:
        return "done" if fs.read_text(run_dir / "DONE") is not None else "resume"
    if existing is not None:
        if not force_override:
            raise SystemExit(f"{run_dir} holds a run with a different config or commit. Use a new ablation name, or "
                             "pass --force-override to delete it and start over.")
        print(f"--force-override: deleting {run_dir}")
        fs.rmtree(run_dir)
    fs.write_text(run_dir / "config.yaml", config_yaml)
    return "new"


def mark_done(run_dir):
    Path(run_dir, "DONE").write_text("")
