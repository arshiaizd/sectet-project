#!/usr/bin/env python3
"""Create deterministic, disjoint worker shards from the canonical eval list."""

import argparse
import hashlib
import json
from pathlib import Path


CANONICAL_SHA256 = "8efa57d84af25aab013bc98a2deeed5ee6ffd094fec7bb401379f07ecb678804"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    actual_hash = sha256(args.input)
    if actual_hash != CANONICAL_SHA256:
        raise ValueError(
            f"canonical evaluation checksum mismatch: {actual_hash}"
        )

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if len(records) != 250:
        raise ValueError(f"expected 250 records, found {len(records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned_ids = []
    for rank in range(args.workers):
        shard = records[rank::args.workers]
        output = args.output_dir / (
            f"coco_target_eval_250_seed_20260815.rank{rank}-of-{args.workers}.json"
        )
        output.write_text(
            json.dumps(shard, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        assigned_ids.extend(record["image_path"] for record in shard)
        print(f"rank {rank}: {len(shard)} records -> {output}")

    canonical_ids = [record["image_path"] for record in records]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("worker shards overlap")
    if set(assigned_ids) != set(canonical_ids):
        raise ValueError("worker shards do not cover the canonical list")


if __name__ == "__main__":
    main()
