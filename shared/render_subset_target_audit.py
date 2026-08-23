#!/usr/bin/env python3
"""Render COCO target annotations for manual visual auditing."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-page", type=int, default=20)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=20)
    cols, tile_w, tile_h = 4, 640, 500
    rows = math.ceil(args.per_page / cols)

    for start in range(0, len(records), args.per_page):
        page = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        for offset, record in enumerate(records[start : start + args.per_page]):
            image = Image.open(args.image_dir / record["image_path"]).convert("RGB")
            source_w, source_h = image.size
            scale = min(tile_w / source_w, (tile_h - 55) / source_h)
            resized = image.resize(
                (round(source_w * scale), round(source_h * scale)), Image.Resampling.LANCZOS
            )
            overlay = Image.new("RGBA", resized.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for polygon in record["segmentation"]:
                points = [
                    (polygon[i] * scale, polygon[i + 1] * scale)
                    for i in range(0, len(polygon), 2)
                ]
                draw.polygon(points, fill=(255, 0, 0, 125), outline=(255, 255, 0, 255), width=3)
            x, y, w, h = record["bbox_xywh"]
            draw.rectangle(
                (x * scale, y * scale, (x + w) * scale, (y + h) * scale),
                outline=(0, 255, 0, 255),
                width=3,
            )
            resized = Image.alpha_composite(resized.convert("RGBA"), overlay).convert("RGB")
            col, row = offset % cols, offset // cols
            x0 = col * tile_w + (tile_w - resized.width) // 2
            y0 = row * tile_h + 52
            page.paste(resized, (x0, y0))
            label = (
                f"case {start + offset + 1} | {record['image_path']} | "
                f"{record['select_category']} | mask={record['mask_fraction']:.4f}"
            )
            ImageDraw.Draw(page).text((col * tile_w + 8, row * tile_h + 12), label, fill="black", font=font)
        page.save(args.output_dir / f"targets_{start + 1:03d}_{min(start + args.per_page, len(records)):03d}.jpg", quality=92)


if __name__ == "__main__":
    main()
