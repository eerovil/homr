import pathlib
import subprocess

TEST_FILE_PATH = pathlib.Path(__file__).parent.resolve()
PYPROJECT_TOML_PATH = list(TEST_FILE_PATH.glob("../pyproject.toml"))
MAKEFILE_PATH = list(TEST_FILE_PATH.glob("../Makefile"))


def _hooks_directory() -> pathlib.Path:
    """Where git keeps this checkout's hooks.

    In a git worktree `.git` is a file pointing at the main repository, and the
    hooks live in the main repository's git directory, shared by every worktree;
    `../.git/hooks` does not exist there however the hooks are set up.
    """
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],  # noqa: S607
            cwd=TEST_FILE_PATH,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return TEST_FILE_PATH.parent / ".git" / "hooks"
    return (TEST_FILE_PATH / common).resolve() / "hooks"


PRECOMMIT_HOOKS_PATH = [path for path in [_hooks_directory()] if path.is_dir()]


def test_file_uniqueness() -> None:
    # File uniqueness
    if len(PYPROJECT_TOML_PATH) != 1:
        raise ValueError(
            "Found more than one 'pyproject.toml':"
            f" {', '.join(str(p) for p in PYPROJECT_TOML_PATH) }"
        )

    if len(MAKEFILE_PATH) != 1:
        raise ValueError(
            f"Found more than one 'Makefile': {', '.join(str(p) for p in MAKEFILE_PATH) }"
        )


def test_isset_precommit_hooks() -> None:
    if len(PRECOMMIT_HOOKS_PATH) == 0:
        raise ValueError("Pre-commit hooks are not set, run `make pre-commit` in `bash`")
