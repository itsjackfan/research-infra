"""Once per clone: `uv run python -m research_infra.install_hooks`.
Writes shims into the common git dir's hooks/ (shared by every worktree) and unsets core.hooksPath:
  post-commit    sync this worktree to the cluster
  post-checkout  init the research-infra submodule in new worktrees (so the path dependency resolves)"""
import subprocess
from pathlib import Path

HOOKS = {
    "post-commit": "#!/bin/sh\nexec uv run --no-sync python -m research_infra.sync\n",
    "post-checkout": "#!/bin/sh\ngit submodule update --init research-infra\n",
}


def main():
    hooks = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()).resolve() / "hooks"
    hooks.mkdir(exist_ok=True)
    for name, text in HOOKS.items():
        (hooks / name).write_text(text)
        (hooks / name).chmod(0o755)
    subprocess.run(["git", "config", "--unset", "core.hooksPath"])


if __name__ == "__main__":
    main()
