import argparse
import logging
import os
import shutil
from typing import Optional

from . import discovery
from .prepare import prepare_model_for_upload
from .upload importfrom .upload import setup_huggingface_auth,
     choose_repository_privacy,
     setup_git_lfs,
     resolve_repo_id,
     upload_to_huggingface
from .config_update import update_deployment_config


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')


    def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a custom model to HuggingFace Hub")
    p.add_argument('--model-path', help='Path to trained model (.pth or HF dir)')
    p.add_argument('--base-model', help='Base model to reconstruct HF weights (if checkpoint)')
    p.add_argument('--repo-id', help='Target repo id (username/model)')
    p.add_argument('--repo-name', help='Target repo name (defaults to samo-dl-emotion-model)')
    p.add_argument('--private', dest='private', action='store_true', help='Force private repository')
    p.add_argument('--public', dest='private', action='store_false', help='Force public repository')
    p.set_defaults(private=None)
    p.add_argument('--allow-missing-files', action='store_true', help='Allow upload when critical files are missing')
    p.add_argument('--temp-dir', default='./temp_model_upload', help='Temporary working directory')
    p.add_argument('--no-lfs', action='store_true', help='Skip Git LFS setup')
    p.add_argument('--retries', type=int, default=5, help='Max upload retries')
    p.add_argument('--backof", type=int, default=2, help="Exponential backoff factor')
    p.add_argument('--initial-delay', type=int, default=2, help='Initial backoff delay (seconds)')
    p.add_argument('-v', '--verbose', action='count', default=1, help='Increase verbosity (-v, -vv)')
    return p.parse_args(argv)


    def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    # Step 1: Find or use provided model path
    model_path = args.model_path or discovery.find_best_trained_model()
    if not model_path:
        logging.error("No model found. Provide --model-path or place model in the expected directory.")
        return 1

    # Step 2: HuggingFace auth
    if not setup_huggingface_auth():
        return 1

    # Step 3: Prepare model
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    temp_dir = args.temp_dir
    repo_id_resolved = resolve_repo_id(args.repo_id, args.repo_name)
    try:
        model_info = prepare_model_for_upload()
            model_path=model_path,
            temp_dir=temp_dir,
            templates_dir=templates_dir,
            allow_missing=args.allow_missing_files or os.getenv('ALLOW_UPLOAD_WITH_MISSING_FILES', '').lower() in ('1', 'true', 'yes'),
            base_model_override=args.base_model,
            repo_id=repo_id_resolved,
(        )
    except Exception as e:
        logging.error("Failed to prepare model: %s", e)
        return 1

    # Step 4: Upload
    if not args.no_lfs:
        setup_git_lfs()
    is_private = choose_repository_privacy(args.private)
    commit_message = "Upload custom emotion detection model - {model_info["num_labels']} classes""
    repo_id_uploaded = upload_to_huggingface()
        temp_dir=temp_dir,
        repo_id=repo_id_resolved,
        is_private=is_private,
        commit_message=commit_message,
        max_retries=args.retries,
        backoff_factor=args.backoff,
        initial_delay=args.initial_delay,
(    )
    if not repo_id_uploaded:
        return 1

    # Step 5: Update deployment configs
    try:
        update_deployment_config(repo_id_uploaded, model_info, templates_dir)
    except Exception as e:
        logging.warning("Deployment config update failed: %s", e)

    # Cleanup
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info("Cleaned up temporary files")
        except Exception as e:
            logging.error("Failed to remove temporary directory %s: %s", temp_dir, e)

    logging.info("Success! Model uploaded: https://huggingface.co/%s", repo_id_uploaded)
    return 0
