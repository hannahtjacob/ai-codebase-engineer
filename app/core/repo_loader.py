from pathlib import Path
from git import Repo
import shutil
import hashlib


class RepoLoader:
    def __init__(self, base_path: str = "data/repos"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def repo_id_from_url(self, repo_url: str) -> str:
        return hashlib.sha256(repo_url.encode()).hexdigest()[:12]

    def clone_repo(self, repo_url: str) -> Path:
        repo_id = self.repo_id_from_url(repo_url)
        target_path = self.base_path / repo_id

        if target_path.exists():
            shutil.rmtree(target_path)

        Repo.clone_from(repo_url, target_path)
        return target_path