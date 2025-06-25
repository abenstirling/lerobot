#!/usr/bin/env python
import os
import shutil
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def backup_dir(src_dir: Path):
    backup_path = src_dir.parent / (src_dir.name + "_backup")
    if backup_path.exists():
        print(f"Backup already exists at {backup_path}, skipping backup.")
        return backup_path
    shutil.copytree(src_dir, backup_path)
    print(f"Backup created at {backup_path}")
    return backup_path


def remove_and_reindex(dataset_root: Path, episodes_to_remove: list[int]):
    print(f"Removing episodes: {episodes_to_remove}")
    episodes_to_remove = sorted(set(episodes_to_remove))
    # 1. Remove parquet and video files for each episode
    data_dir = dataset_root / "data/chunk-000"
    video_dirs = [
        dataset_root / "videos/chunk-000/observation.images.laptop",
        dataset_root / "videos/chunk-000/observation.images.phone",
    ]
    # Remove files
    for ep in episodes_to_remove:
        ep_file = data_dir / f"episode_{ep:06d}.parquet"
        if ep_file.exists():
            ep_file.unlink()
            print(f"Deleted {ep_file}")
        for vdir in video_dirs:
            vfile = vdir / f"episode_{ep:06d}.mp4"
            if vfile.exists():
                vfile.unlink()
                print(f"Deleted {vfile}")
    # 2. Rename all later episodes down by the number of removed episodes before them
    # Build a map: old_index -> new_index (skip removed)
    all_indices = [int(f.name.split("_")[1].split(".")[0]) for f in data_dir.glob("episode_*.parquet")]
    all_indices = sorted(all_indices)
    index_map = {}
    removed_set = set(episodes_to_remove)
    decrement = 0
    for idx in all_indices:
        decrement = sum(1 for r in episodes_to_remove if r < idx)
        new_idx = idx - decrement
        if new_idx != idx:
            # Rename parquet
            src = data_dir / f"episode_{idx:06d}.parquet"
            dst = data_dir / f"episode_{new_idx:06d}.parquet"
            src.rename(dst)
            print(f"Renamed {src} -> {dst}")
            # Rename videos
            for vdir in video_dirs:
                vsrc = vdir / f"episode_{idx:06d}.mp4"
                vdst = vdir / f"episode_{new_idx:06d}.mp4"
                if vsrc.exists():
                    vsrc.rename(vdst)
                    print(f"Renamed {vsrc} -> {vdst}")
        index_map[idx] = new_idx
    # 3. Update metadata files
    meta_dir = dataset_root / "meta"
    for meta_file in ["episodes.jsonl", "episodes_stats.jsonl"]:
        lines = []
        with open(meta_dir / meta_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                ep_idx = entry["episode_index"]
                if ep_idx in removed_set:
                    continue
                # Update index
                decrement = sum(1 for r in episodes_to_remove if r < ep_idx)
                entry["episode_index"] = ep_idx - decrement
                lines.append(entry)
        with open(meta_dir / meta_file, "w") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")
        print(f"Updated {meta_file}")
    # Update info.json
    info_path = meta_dir / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info["total_episodes"] = info["total_episodes"] - len(episodes_to_remove)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print("Updated info.json")


def upload_to_hub(dataset_root: Path, new_repo_id: str):
    api = HfApi()
    print(f"Creating repo {new_repo_id} on Hugging Face Hub...")
    create_repo(new_repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading {dataset_root} to {new_repo_id}...")
    upload_folder(
        repo_id=new_repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        allow_patterns=["data/**", "videos/**", "meta/**", "README.md"]
    )
    print(f"Upload complete!")


def main():
    parser = argparse.ArgumentParser(description="Remove and reindex episodes in a LeRobot dataset.")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root directory.")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Episode indices to remove.")
    parser.add_argument("--new-repo-id", type=str, required=True, help="New Hugging Face repo id for upload.")
    parser.add_argument("--backup", action="store_true", help="Backup dataset before modifying.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if args.backup:
        backup_dir(root)
    remove_and_reindex(root, args.episodes)
    upload_to_hub(root, args.new_repo_id)

if __name__ == "__main__":
    main() 