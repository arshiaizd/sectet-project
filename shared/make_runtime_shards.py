#!/usr/bin/env python3
"""Create deterministic strided shards for any validated 250-case manifest."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if len(records) != 250:
        raise ValueError(f"expected 250 records, found {len(records)}")
    image_paths = [record["image_path"] for record in records]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("manifest contains duplicate image paths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned = []
    for rank in range(args.workers):
        shard = records[rank :: args.workers]
        output = args.output_dir / f"rank{rank}-of-{args.workers}.json"
        output.write_text(
            json.dumps(shard, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        assigned.extend(record["image_path"] for record in shard)
        print(f"rank {rank}: {len(shard)} images -> {output}")

    if len(assigned) != len(set(assigned)) or set(assigned) != set(image_paths):
        raise RuntimeError("runtime shards do not form an exact partition")


if __name__ == "__main__":
    main()
