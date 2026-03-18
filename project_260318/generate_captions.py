#!/usr/bin/env python3
"""Generate image captions using Claude's vision API.

Reads a CSV of prompts, identifies rows with image references,
sends each image to Claude for captioning, and outputs results.
"""

import argparse
import base64
import csv
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import anthropic


def encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (base64_data, media_type)."""
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png"
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime


def generate_caption(client: anthropic.Anthropic, image_path: str, prompt_raw: str, model: str) -> str:
    """Send an image to Claude and get a descriptive caption."""
    b64_data, media_type = encode_image(image_path)

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        },
        {
            "type": "text",
            "text": (
                "다음은 이 이미지와 함께 제출된 사용자의 프롬프트입니다:\n"
                f"「{prompt_raw}」\n\n"
                "위 프롬프트의 맥락을 참고하여, 이 이미지를 상세하게 묘사하는 캡션을 한국어로 작성해 주세요. "
                "캡션에는 이미지의 주요 객체, 구도, 색상, 스타일 등을 포함해 주세요. "
                "캡션만 출력하고 다른 설명은 하지 마세요."
            ),
        },
    ]

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate captions for referenced images using Claude vision API.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--image-dir", required=True, help="Directory containing the referenced images")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <input_stem>_captioned.csv)")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (prefer ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between API calls (rate limiting)")
    args = parser.parse_args()

    # Resolve API key: env var takes precedence
    api_key = os.environ.get("ANTHROPIC_API_KEY") or args.api_key
    if not api_key:
        print("Error: Provide an API key via --api-key or ANTHROPIC_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    image_dir = Path(args.image_dir)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_captioned.csv")

    # Read CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add caption column if not present
    if "image_caption" not in fieldnames:
        fieldnames = list(fieldnames) + ["image_caption"]

    client = anthropic.Anthropic(api_key=api_key)

    total = len(rows)
    img_count = sum(1 for r in rows if r.get("prompt_img", "").strip())
    print(f"Loaded {total} rows, {img_count} with image references.")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}\n")

    captioned = 0
    skipped = 0
    for i, row in enumerate(rows):
        img_filename = row.get("prompt_img", "").strip()
        if not img_filename:
            row.setdefault("image_caption", "")
            continue

        img_path = image_dir / img_filename
        if not img_path.exists():
            print(f"  [{i+1}/{total}] SKIP {row['prompt_id']}: image not found ({img_filename})")
            row["image_caption"] = ""
            skipped += 1
            continue

        prompt_raw = row.get("prompt_raw", "")
        print(f"  [{i+1}/{total}] Captioning {row['prompt_id']} ({img_filename})...", end=" ", flush=True)

        try:
            caption = generate_caption(client, str(img_path), prompt_raw, args.model)
            row["image_caption"] = caption
            captioned += 1
            print("OK")
        except anthropic.APIError as e:
            print(f"API error: {e}")
            row["image_caption"] = ""
            skipped += 1

        if args.delay > 0:
            time.sleep(args.delay)

    # Write output CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Captioned: {captioned}, Skipped: {skipped}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
