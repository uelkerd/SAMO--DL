import logging
import os
import shutil
import subprocess
import time
from typing import Optional

from huggingface_hub import HfApi, create_repo, login

from .discovery import is_interactive_environment


def setup_huggingface_auth() -> bool:
    logging.info("Authenticating with HuggingFace")
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        if is_interactive_environment():
            logging.error(
                "HuggingFace token not found in env (HUGGINGFACE_TOKEN/HF_TOKEN)"
            )
        else:
            logging.error(
                "Non-interactive environment: set HUGGINGFACE_TOKEN or HF_TOKEN"
            )
        return False
    try:
        login(token=hf_token)
        logging.info("Authenticated with token")
        return True
    except Exception as e:
        logging.error("Token authentication failed: %s", e)
        return False


def choose_repository_privacy(cli_private: Optional[bool] = None) -> bool:
    if cli_private is not None:
        logging.info(
            "Repository privacy from CLI: %s", "private" if cli_private else "public"
        )
        return cli_private
    hf_repo_private = os.environ.get("HF_REPO_PRIVATE")
    if hf_repo_private:
        if hf_repo_private.lower() == "true":
            logging.info("Using PRIVATE repository (HF_REPO_PRIVATE=true)")
            return True
        if hf_repo_private.lower() == "false":
            logging.info("Using PUBLIC repository (HF_REPO_PRIVATE=false)")
            return False
        logging.warning("Invalid HF_REPO_PRIVATE value: %s", hf_repo_private)
    if not is_interactive_environment():
        logging.info("Non-interactive environment: defaulting to PRIVATE repository")
        return True
    # Default interactive: public
    return False


def setup_git_lfs() -> bool:
    logging.info("Setting up Git LFS for large files")
    if shutil.which("git") is None:
        logging.warning("Git not installed. Skipping Git LFS setup")
        return False
    try:
        result = subprocess.run(
            ["git", "lfs", "version"], capture_output=True, text=True, check=True
        )
        if result.returncode != 0:
            logging.warning("Git LFS not available. Install with: git lfs install")
            return False
        lfs_patterns = [
            "*.bin",
            "*.safetensors",
            "*.onnx",
            "*.pkl",
            "*.pth",
            "*.pt",
            "*.h5",
        ]
        for pattern in lfs_patterns:
            subprocess.run(
                ["git", "lfs", "track", pattern],
                capture_output=True,
                text=True,
                check=True,
            )
        # Update .gitattributes if exists
        gitattributes_path = ".gitattributes"
        if os.path.exists(gitattributes_path):
            with open(gitattributes_path) as f:
                content = f.read()
            for pattern in lfs_patterns:
                lfs_line = f"{pattern} filter=lfs diff=lfs merge=lfs -text"
                if lfs_line not in content:
                    content += f"\n{lfs_line}"
            with open(gitattributes_path, "w") as f:
                f.write(content)
        return True
    except Exception as e:
        logging.warning("Git LFS setup failed: %s", e)
        return False


def resolve_repo_id(repo_id: Optional[str], repo_name: Optional[str]) -> str:
    if repo_id:
        return repo_id
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    name = repo_name or "samo-dl-emotion-model"
    return f"{username}/{name}"


def upload_to_huggingface(
    temp_dir: str,
    repo_id: str,
    is_private: bool,
    commit_message: str,
    max_retries: int = 5,
    backoff_factor: int = 2,
    initial_delay: int = 2,
) -> Optional[str]:
    logging.info("Uploading to HuggingFace Hub: %s", repo_id)
    api = HfApi()
    for attempt in range(1, max_retries + 1):
        try:
            create_repo(repo_id, exist_ok=True, private=is_private, repo_type="model")
            if attempt == 1:
                logging.info(
                    "Repository created/confirmed (%s)",
                    "private" if is_private else "public",
                )
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            logging.info(
                "Model uploaded successfully: https://huggingface.co/%s", repo_id
            )
            return repo_id
        except Exception as e:
            logging.warning("Upload attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                logging.error("Maximum upload attempts reached")
                return None
            sleep_time = initial_delay * (backoff_factor ** (attempt - 1))
            logging.info("Retrying in %s seconds...", sleep_time)
            time.sleep(sleep_time)
    return None
