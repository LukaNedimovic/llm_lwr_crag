import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

from utils.logging import logger, toggle_logger

__all__ = ["download_repo"]


def download_repo(
    repo_url: Path,
    repo_dir: Path,
    force_download: bool = False,
    log: Union[bool, int] = True,
    ui: bool = False,
):
    """
    Clone a GitHub repository to a target directory.

    Args:
        repo_url (str): URL of the GitHub repository.
        target_dir (Path): Path to the directory where the repo will be cloned.
        force_download (bool): If True, remove the existing directory before cloning.
        log (bool): If True, will log the cloning process
    """
    toggle_logger(log)

    try:
        if repo_dir.exists():
            if force_download:
                logger.info(f"Removing existing directory: {repo_dir}.")
                shutil.rmtree(repo_dir)
            else:
                logger.info(f"Repository already cloned at: {repo_dir}.")
                return f"ℹ️ Repository already cloned at: {repo_dir}."

        os.makedirs(repo_dir, exist_ok=True)

        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

        logger.info(f"Repository cloned successfully to {repo_dir}")
        return f"✅ Repository cloned successfully to: {repo_dir}!"
    except subprocess.CalledProcessError as e:
        logger.fatal(f"Failed to clone repository: {e}")
