#!/usr/bin/env python3
"""Measure a txt2img endpoint and report a clearly labeled 1000-image estimate."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import httpx
from PIL import Image, ImageStat


def _generate(client: httpx.Client, endpoint: str, args, seed: int) -> tuple[float, dict]:
    started = time.perf_counter()
    token = os.environ.get("SD_API_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = client.post(
        endpoint,
        json={
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "seed": seed,
        },
        headers=headers,
        timeout=args.timeout,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    payload = response.json()
    images = payload.get("images") or ([payload["image"]] if payload.get("image") else [])
    if not images:
        raise ValueError("txt2img response did not contain an image")
    encoded = images[0].split(",", 1)[-1]
    image = Image.open(BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
    stddev = round(sum(ImageStat.Stat(image).stddev) / 3, 2)
    if image.width < 1 or image.height < 1 or stddev == 0:
        raise ValueError("txt2img returned a blank or invalid image")
    return elapsed, {
        "shape": [image.height, image.width],
        "pixel_stddev": stddev,
        "device": payload.get("device"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:7861/sdapi/v1/txt2img")
    parser.add_argument("--samples", type=int, default=4, help="total requests, including the first request")
    parser.add_argument("--prompt", default="a small controlled indoor flame, documentary fire safety photo")
    parser.add_argument("--negative-prompt", default="text, watermark, logo")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--proxy", help="optional HTTP proxy for a remote endpoint")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if not 2 <= args.samples <= 1000:
        parser.error("--samples must be between 2 and 1000")

    endpoint = args.endpoint
    benchmark_started = time.perf_counter()
    with httpx.Client(trust_env=False, proxy=args.proxy, timeout=args.timeout) as client:
        first_seconds, first_image = _generate(client, endpoint, args, 20261015)
        warm_seconds = []
        image_details = [first_image]
        for offset in range(args.samples - 1):
            elapsed, details = _generate(client, endpoint, args, 20261016 + offset)
            warm_seconds.append(elapsed)
            image_details.append(details)
            completed = offset + 2
            if completed % 100 == 0 or completed == args.samples:
                print(f"Generated and verified {completed}/{args.samples} images", file=sys.stderr, flush=True)

    mean_seconds = statistics.mean(warm_seconds)
    total_elapsed_seconds = time.perf_counter() - benchmark_started
    pixel_stddevs = [item["pixel_stddev"] for item in image_details]
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "request": {"steps": args.steps, "width": args.width, "height": args.height},
        "request_count": args.samples,
        "first_request_seconds": round(first_seconds, 3),
        "warm_samples": len(warm_seconds),
        "warm_mean_seconds": round(mean_seconds, 3),
        "warm_median_seconds": round(statistics.median(warm_seconds), 3),
        "warm_p95_seconds": round(sorted(warm_seconds)[math.ceil(0.95 * len(warm_seconds)) - 1], 3),
        "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        "mean_total_seconds_per_image": round(total_elapsed_seconds / args.samples, 3),
        "verified_image_count": len(image_details),
        "output_shapes": sorted({tuple(item["shape"]) for item in image_details}),
        "devices": sorted({item["device"] for item in image_details if item["device"]}),
        "pixel_stddev_range": [round(min(pixel_stddevs), 2), round(max(pixel_stddevs), 2)],
        "scope": "txt2img endpoint only; excludes physics, rendering, labeling, quality scoring, and version storage",
        "linear_1000_image_estimate_hours": round(mean_seconds * 1000 / 3600, 3),
        "txt2img_1000_request_batch_measured": args.samples == 1000,
        "full_pipeline_batch_acceptance": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
