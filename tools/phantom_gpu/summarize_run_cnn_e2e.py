#!/usr/bin/env python3
"""Validate and summarize one bounded handwritten Phantom ResNet-20 run."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


BOOTSTRAP_LABELS = {
    "dsl": "ACE DSL Bootstrap Time",
    "matched-native-slim": "Matched Native Slim Bootstrap Time",
}


def _one_per_image(pattern: str, text: str, count: int, description: str) -> list[str]:
    values = re.findall(pattern, text, flags=re.MULTILINE)
    if len(values) != count:
        raise ValueError(f"expected {count} {description} records, found {len(values)}")
    return values


def summarize(
    *,
    mode: str,
    start_image: int,
    end_image: int,
    run_log: Path,
    result_dir: Path,
    require_correct: bool,
) -> dict[str, object]:
    text = run_log.read_text(encoding="utf-8", errors="strict")
    image_count = end_image - start_image + 1
    expected_calls = 18 * image_count
    bootstrap_label = BOOTSTRAP_LABELS[mode]
    bootstrap_times = [
        int(value)
        for value in re.findall(
            rf"^{re.escape(bootstrap_label)}\s*:\s*([0-9]+)\s+ms$",
            text,
            flags=re.MULTILINE,
        )
    ]
    if len(bootstrap_times) != expected_calls:
        raise ValueError(
            f"expected {expected_calls} {bootstrap_label} records, "
            f"found {len(bootstrap_times)}"
        )
    other_label = next(value for key, value in BOOTSTRAP_LABELS.items() if key != mode)
    if re.search(rf"^{re.escape(other_label)}\s*:", text, flags=re.MULTILINE):
        raise ValueError(f"{mode} log unexpectedly contains {other_label}")

    fatal_markers = (
        "ACE_PHANTOM_RUN_CNN_DSL_BOOTSTRAP_ERROR",
        "terminate called",
        "uncaught exception",
        "CUDA error",
        "out of memory",
    )
    for marker in fatal_markers:
        if marker.lower() in text.lower():
            raise ValueError(f"run log contains fatal marker: {marker}")
    if re.search(
        r"(?<![A-Za-z0-9_])[-+]?(?:nan|inf(?:inity)?)(?![A-Za-z0-9_])",
        text,
        flags=re.IGNORECASE,
    ):
        raise ValueError("run log contains a nonfinite numeric value")

    clear_labels = [
        int(value)
        for value in _one_per_image(
            r"^image label:\s*([0-9]+)$", text, image_count, "clear-label"
        )
    ]
    inferred_labels = [
        int(value)
        for value in _one_per_image(
            r"^inferred label:\s*([0-9]+)$", text, image_count, "inferred-label"
        )
    ]
    score_tokens = _one_per_image(
        r"^max score:\s*(\S+)\s*$", text, image_count, "maximum-score"
    )
    scores = [float(value) for value in score_tokens]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("one or more maximum scores are nonfinite")

    inference_milliseconds = [
        int(value)
        for value in _one_per_image(
            r"^total time\s*:\s*([0-9]+)\s+ms$",
            text,
            image_count,
            "inference-time",
        )
    ]
    all_threads_values = re.findall(
        r"^all threads time\s*:\s*([0-9]+)\s+ms$", text, flags=re.MULTILINE
    )
    if len(all_threads_values) != 1:
        raise ValueError(
            f"expected one all-threads timing, found {len(all_threads_values)}"
        )

    label_summary = result_dir / f"resnet20_cifar10_label_{start_image}_{end_image}"
    summary_text = label_summary.read_text(encoding="utf-8", errors="strict")
    summary_rows = re.findall(
        r"^image_id:\s*([0-9]+),\s*image label:\s*([0-9]+),\s*"
        r"inferred label:\s*([0-9]+)$",
        summary_text,
        flags=re.MULTILINE,
    )
    expected_ids = list(range(start_image, end_image + 1))
    if [int(row[0]) for row in summary_rows] != expected_ids:
        raise ValueError("label summary does not contain the exact requested image range")
    if [int(row[1]) for row in summary_rows] != clear_labels:
        raise ValueError("label summary clear labels differ from the run log")
    if [int(row[2]) for row in summary_rows] != inferred_labels:
        raise ValueError("label summary inferred labels differ from the run log")

    for offset, image_id in enumerate(expected_ids):
        image_result = result_dir / f"resnet20_cifar10_image{image_id}.txt"
        image_text = image_result.read_text(encoding="utf-8", errors="strict")
        if f"image label: {clear_labels[offset]}" not in image_text:
            raise ValueError(f"image {image_id} result has the wrong clear label")
        if f"inferred label: {inferred_labels[offset]}" not in image_text:
            raise ValueError(f"image {image_id} result has the wrong inferred label")
        if f"max score: {score_tokens[offset]}" not in image_text:
            raise ValueError(f"image {image_id} result has a different maximum score")

    correct = [clear == inferred for clear, inferred in zip(clear_labels, inferred_labels)]
    if require_correct and not all(correct):
        raise ValueError("the bounded correctness run misclassified an image")

    return {
        "schema_version": "ace.phantom.resnet20-e2e-summary/1.0.0",
        "status": "pass",
        "mode": mode,
        "image_range": [start_image, end_image],
        "image_count": image_count,
        "expected_bootstrap_calls": expected_calls,
        "observed_bootstrap_calls": len(bootstrap_times),
        "bootstrap_time_ms": {
            "values": bootstrap_times,
            "sum": sum(bootstrap_times),
            "mean": statistics.fmean(bootstrap_times),
            "median": statistics.median(bootstrap_times),
            "minimum": min(bootstrap_times),
            "maximum": max(bootstrap_times),
        },
        "inference_time_ms": inference_milliseconds,
        "all_threads_time_ms": int(all_threads_values[0]),
        "timing_note": (
            "The source assigns a duration_cast<milliseconds> result to a "
            "microseconds duration and divides that microsecond count by 1000; "
            "the printed values are whole milliseconds. Bootstrap lines are "
            "synchronized whole milliseconds."
        ),
        "results": [
            {
                "image_id": image_id,
                "clear_label": clear_labels[index],
                "inferred_label": inferred_labels[index],
                "correct": correct[index],
                "maximum_score": scores[index],
            }
            for index, image_id in enumerate(expected_ids)
        ],
        "correct_count": sum(correct),
        "require_correct": require_correct,
        "no_nonfinite_scores": True,
        "no_nonfinite_values_in_run_log": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(BOOTSTRAP_LABELS), required=True)
    parser.add_argument("--start-image", type=int, required=True)
    parser.add_argument("--end-image", type=int, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-correct", action="store_true")
    args = parser.parse_args()
    result = summarize(
        mode=args.mode,
        start_image=args.start_image,
        end_image=args.end_image,
        run_log=args.run_log,
        result_dir=args.result_dir,
        require_correct=args.require_correct,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
