"""Post-commit sync: mirror this worktree to <root>/<user>/<project>/code/<branch>/ on the project's cluster.
Run by the hook from research_infra.install_hooks. Needs an open `ssh <host>` ControlMaster connection (BatchMode
fails fast instead of prompting for 2FA). Clusters with host=None skip it (code ships from this machine)."""
import subprocess
from pathlib import Path

from research_infra.launch import git_info, load_cluster

# leading / anchors to the repo root; excluded paths are also protected from --delete (cluster .env, .venv, outputs).
# Unanchored .git also covers submodules' .git files.
EXCLUDES = (".git", "/.venv", "/.claude", "/.env", "/outputs", "/wandb", "__pycache__", ".pytest_cache")


def write_gitinfo(repo: Path) -> str:
    """`<branch> <commit>` -> repo/.gitinfo (the cluster copy has no .git); returns the branch."""
    branch, commit, _ = git_info(repo)
    (repo / ".gitinfo").write_text(f"{branch} {commit}\n")
    return branch


def rsync_cmd(repo: Path, host: str, dst, filters=()) -> list[str]:
    return ["rsync", "-az", "--delete", "-e", "ssh -o BatchMode=yes", f"--rsync-path=mkdir -p {dst} && rsync",
            *filters, *[f"--exclude={e}" for e in EXCLUDES], f"{repo}/", f"{host}:{dst}/"]


def main():
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    cluster = load_cluster(repo)
    if cluster.host is None:
        return
    branch = write_gitinfo(repo)
    cmd = rsync_cmd(repo, cluster.host, cluster.code_dir(branch), cluster.rsync_filters)
    if subprocess.run(cmd).returncode:
        print(f"sync: failed; run `ssh {cluster.host}` once, then `uv run python -m research_infra.sync`")


if __name__ == "__main__":
    main()
