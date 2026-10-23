#!/usr/bin/env python
"""
sync_to_hf.py — back up runs/ (metrics, configs, LoRA adapters) to a PRIVATE Hugging Face repo.

Never uploaded:
  * generations.jsonl     raw model outputs (include responses to harmful prompts)
  * judge_cache.jsonl     judge reasons can paraphrase harmful content
  * _merged/              full merged model copies (6 GB each; rebuildable from adapter)
  * ckpt/                 optimizer state (only needed to resume on the same machine)

Usage: python sync_to_hf.py --repo Trilok6767/rim-sdp-runs
Needs a Hugging Face token with WRITE access (HF_TOKEN env or `hf auth login`).
"""
import argparse
import sys

from huggingface_hub import HfApi

IGNORE = ["**/generations.jsonl", "**/judge_cache.jsonl", "_merged/**", "**/_merged/**",
          "**/ckpt/**", "**/*.tmp"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--folder", default="runs")
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(a.repo, repo_type="model").private:
        print(f"REFUSING: {a.repo} is public. Make it private on huggingface.co first.")
        return 1
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=a.folder,
                      path_in_repo=a.folder, ignore_patterns=IGNORE,
                      commit_message="sync runs")
    print(f"synced {a.folder}/ -> https://huggingface.co/{a.repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
