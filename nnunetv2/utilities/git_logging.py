from __future__ import annotations

from pathlib import Path
import inspect

from git import Repo, InvalidGitRepositoryError, NoSuchPathError  # type: ignore


def _guess_repo_start_dir() -> Path:
    """Heuristic to locate a starting directory for git discovery."""

    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = inspect.getframeinfo(frame.f_back).filename
            if caller_file and caller_file != "<stdin>":
                return Path(caller_file).resolve().parent
    except Exception:
        pass

    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def log_git_context(trainer, repo_dir=None):
    """Log basic Git context information to the trainer's log."""

    start_dir = Path(repo_dir).resolve() if repo_dir else _guess_repo_start_dir()

    try:
        repo = Repo(start_dir, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        trainer.print_to_log_file(
            "Git:", "Not a git repository:", str(exc), f"(start={start_dir})"
        )
        return

    root = Path(repo.working_tree_dir).resolve()

    try:
        sha = repo.head.commit.hexsha
        short = sha[:12]
    except Exception:
        sha, short = None, None

    try:
        branch = repo.active_branch.name
    except Exception:
        branch = "(detached)"

    dirty = repo.is_dirty(untracked_files=True, working_tree=True)

    trainer.print_to_log_file(
        "Git:",
        f"commit={short or 'UNKNOWN'}",
        f"branch={branch}",
        f"dirty={dirty}",
        f"root={root}",
        f"start={start_dir}",
    )

    unstaged = list(repo.index.diff(None))
    if unstaged:
        trainer.print_to_log_file(f"Git: Unstaged changes ({len(unstaged)}):")
        for diff in unstaged:
            change_type = diff.change_type
            if change_type == "R":
                pretty = f"{diff.rename_from} -> {diff.rename_to}"
            else:
                path = diff.b_path or diff.a_path
                pretty = str(root / path) if path else "(unknown)"
            trainer.print_to_log_file("   ", change_type, pretty)
    else:
        trainer.print_to_log_file("Git: No unstaged changes to tracked files.")

    untracked = repo.untracked_files
    if untracked:
        trainer.print_to_log_file(f"Git: Untracked files ({len(untracked)}):")
        for path in untracked:
            trainer.print_to_log_file("   ?", str(root / path))
    else:
        trainer.print_to_log_file("Git: No untracked files.")


__all__ = ["log_git_context"]
