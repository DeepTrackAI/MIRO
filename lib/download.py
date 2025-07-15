import requests
from pathlib import Path
from typing import Optional


def sdownload(
    id: int,
    local_folder: str,
    blinking: bool = False,
    branch: str = "main",
    token: Optional[str] = None,
) -> Path:
    """
    Downloads only the specified scenario folder from GitHub by recursively
    walking the GitHub Contents API and saving each file.
    """
    owner = "DJ-Nieves"
    repo = "ARI-and-IoU-cluster-analysis-evaluation"
    basename = "Multiple Blinking" if blinking else "Ground Truth"
    remote_path = f"{basename} - Scenario {id}"
    local_root = Path(local_folder)
    target_dir = local_root / remote_path

    if target_dir.is_dir():
        print(f"Folder '{remote_path}' already exists at {target_dir.resolve()}")
        return target_dir.resolve()

    local_root.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    session.headers.update(headers)

    def _download_dir(path: str, dest: Path):
        """
        Recursively download the contents of `path` in the repo into `dest`.
        """
        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        )
        resp = session.get(api_url)
        resp.raise_for_status()
        for item in resp.json():
            item_type = item["type"]
            name = item["name"]
            if item_type == "file":
                # download and write
                download_url = item["download_url"]
                data = session.get(download_url).content
                out_file = dest / name
                out_file.write_bytes(data)
            elif item_type == "dir":
                # recurse into subdirectory
                sub_dest = dest / name
                sub_dest.mkdir(exist_ok=True)
                _download_dir(item["path"], sub_dest)

    _download_dir(remote_path, target_dir)
    print(f"Downloaded '{remote_path}' into {target_dir.resolve()}")
    return target_dir.resolve()
