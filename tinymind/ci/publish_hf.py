"""Optional: upload an inference package to a Hugging Face model repo. Runs only when the workflow has an
``HF_TOKEN`` secret; nothing in normal CI depends on it, and the workflow treats any failure here as a
warning. NOT exercised by this repository's tests (it needs network access and a token)."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; nothing to do")
        return 0
    from huggingface_hub import HfApi  # imported lazily: only the publish job installs it
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/{args.name}"
    api.create_repo(repo, exist_ok=True, private=True)
    api.upload_folder(folder_path=args.folder, repo_id=repo)
    print(f"uploaded to {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
