#!/usr/bin/env python
"""
sync_to_hf.py — back up / restore runs/ to a PRIVATE Hugging Face repo, so a run can continue
on a fresh machine after the previous one was stopped.

Backed up: markers, metrics, configs, LoRA adapters, training checkpoints (resume), model
responses and judge cache (so evals resume without regenerating or re-paying the judge).
Responses include outputs on harmful prompts: the repo MUST stay private (checked every sync).

Never uploaded: _merged/ (6 GB merged models, rebuildable), *.tmp, smoke-run checkpoints.

Usage:
  python sync_to_hf.py --repo Trilok6767/rim-sdp-runs            # back up
  python sync_to_hf.py --repo Trilok6767/rim-sdp-runs --restore  # pull onto a fresh machine
Needs a Hugging Face token with WRITE access (HF_TOKEN env or `hf auth login`).
"""
import argparse
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

IGNORE_UP = ["_merged/**", "**/_merged/**", "**/*.tmp", "**/*.tmp/**", "runs/train/smoke*/ckpt/**",
             "**/smoke*/ckpt/**"]
# per-machine markers and the live log are never restored
IGNORE_DOWN = ["runs/phase1/env.done", "runs/phase1/data.done", "runs/phase1/phase1.log",
               "**/*.tmp"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folder", default="runs")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    api = HfApi()

    if a.restore:
        try:
            info = api.repo_info(a.repo, repo_type="model")
        except RepositoryNotFoundError:
            print(f"no backup at {a.repo} yet: starting fresh")
            return 0
        if not info.private:
            print(f"REFUSING: {a.repo} is public."); return 1
        snapshot_download(a.repo, repo_type="model", local_dir=".",
                          allow_patterns=[f"{a.folder}/**"], ignore_patterns=IGNORE_DOWN)
        print(f"restored {a.folder}/ from {a.repo}")
        return 0

    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(a.repo, repo_type="model").private:
        print(f"REFUSING: {a.repo} is public. Make it private on huggingface.co first.")
        return 1
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=a.folder,
                      path_in_repo=a.folder, ignore_patterns=IGNORE_UP,
                      commit_message="sync runs")
    print(f"synced {a.folder}/ -> https://huggingface.co/{a.repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
