from pathlib import Path
import subprocess


def sdownload(id: int, local_folder: str, blinking: bool = False):
    """
    Shallow-clones the given folder from DJ-Nieves's GitHub repo using Git's sparse-checkout.
    """
    owner = "DJ-Nieves"
    repo = "ARI-and-IoU-cluster-analysis-evaluation"
    url = f"https://github.com/{owner}/{repo}.git"

    base_dir = Path.cwd() / local_folder
    basename = "Multiple Blinking" if blinking else "Ground Truth"
    folder_name = f"{basename} - Scenario {id}"
    target_dir = base_dir / folder_name
    git_dir = base_dir / ".git"
    sparse_file = git_dir / "info" / "sparse-checkout"

    # Ensure base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Early exit if folder already present
    if target_dir.is_dir():
        print(f"Folder '{folder_name}' already exists in {base_dir.resolve()}.")
        return target_dir.resolve()

    first_clone = False
    # Initialize git in existing directory if needed
    if not git_dir.exists():
        first_clone = True
        # init repo and add remote
        subprocess.run(["git", "-C", str(base_dir), "init"], check=True)
        subprocess.run(
            ["git", "-C", str(base_dir), "remote", "add", "origin", url], check=True
        )
        # bootstrap sparse-checkout in cone mode
        subprocess.run(
            ["git", "-C", str(base_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )

    elif not git_dir.is_dir():
        raise RuntimeError(f"{base_dir!r} exists but is not a Git repo.")

    # Read existing patterns (if any)
    patterns = []
    if sparse_file.exists():
        for line in sparse_file.read_text().splitlines():
            entry = line.strip().rstrip("/")
            if entry:
                patterns.append(entry)

    # Append the requested scenario folder
    if folder_name not in patterns:
        patterns.append(folder_name)

    # Apply sparse list (fetches only missing content)
    subprocess.run(
        ["git", "-C", str(base_dir), "sparse-checkout", "set", *patterns], check=True
    )

    # Fetch latest and reset main branch
    subprocess.run(
        [
            "git",
            "-C",
            str(base_dir),
            "fetch",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            "main",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(base_dir), "checkout", "-B", "main", "origin/main"],
        check=True,
    )

    print(f"Retrieved '{folder_name}' into {base_dir.resolve()}")
    return target_dir.resolve()
