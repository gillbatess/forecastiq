"""Publish the app to a Hugging Face Space using the Hub's HTTP upload API.

Why not `git push`?  Hugging Face rejects git pushes that contain binary files
(our models are .gz, the font is .woff2) unless they go through Git-LFS/Xet.
`upload_folder` handles that automatically, so this works from a laptop or CI
without any LFS setup.

Usage
-----
    pip install -U huggingface_hub
    export HF_TOKEN=hf_xxx            # a token with WRITE access (never commit it)
    export HF_SPACE=gillbatess/forecastiq
    python scripts/deploy_hf.py            # upload
    python scripts/deploy_hf.py --dry-run  # only list the files that would be uploaded
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects

ROOT = Path(__file__).resolve().parents[1]

# Only what the container needs - nothing else is sent to the Space.
ALLOW = ["Dockerfile", "README.md", "requirements.txt", "backend/**", "src/**", "web/**", "artifacts/**"]
IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]


def files_to_upload() -> list[str]:
    everything = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
                  and ".git/" not in p.as_posix() and ".venv/" not in p.as_posix()]
    return sorted(filter_repo_objects(everything, allow_patterns=ALLOW, ignore_patterns=IGNORE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("HF_SPACE"), help="<username>/<space-name>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = files_to_upload()
    if args.dry_run:
        print("\n".join(files))
        print(f"\n{len(files)} files would be uploaded")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Create a token with write access and export it (do not paste it into files).")
    if not args.space or "/" not in args.space:
        sys.exit("Set HF_SPACE=<username>/<space-name> (or pass --space).")

    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"Authenticated as {who}; uploading {len(files)} files to space {args.space} ...")
    info = api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.space,
        repo_type="space",
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        commit_message="Deploy ForecastIQ",
    )
    print("Done:", info)
    print(f"Watch the build: https://huggingface.co/spaces/{args.space}  (Logs tab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
