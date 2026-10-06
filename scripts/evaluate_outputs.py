"""QA-01 output evaluation smoke test."""

import argparse
import json
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = Path("artifacts") / args.run_id
    metadata_file = run_dir / "generated" / "metadata.jsonl"
    score_file = run_dir / "generated" / "scores.json"

    if not metadata_file.exists():
        raise SystemExit(f"Metadata not found: {metadata_file}")

    records = []

    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    results = []

    for record in records:
        image_path = Path(record["image_path"])

        if not image_path.exists():
            status = "FAIL"
            width = None
            height = None
        else:
            with Image.open(image_path) as image:
                width, height = image.size

            status = "PASS" if (width, height) == (512, 512) else "FAIL"

        results.append(
            {
                "run_id": record["run_id"],
                "concept_id": record["concept_id"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "image_path": record["image_path"],
                "width": width,
                "height": height,
                "fidelity_score": 1.0 if status == "PASS" else 0.0,
                "alignment_score": 1.0 if status == "PASS" else 0.0,
                "status": status,
            }
        )

    passed = sum(r["status"] == "PASS" for r in results)
    total = len(results)

    output = {
        "run_id": args.run_id,
        "evaluation_mode": "cpu_smoke_placeholder",
        "total_images": total,
        "passed_images": passed,
        "failed_images": total - passed,
        "fidelity_mean": (
            sum(r["fidelity_score"] for r in results) / total
            if total
            else 0.0
        ),
        "alignment_mean": (
            sum(r["alignment_score"] for r in results) / total
            if total
            else 0.0
        ),
        "results": results,
    }

    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Evaluation completed for {args.run_id}")
    print(f"[OK] Images checked: {total}")
    print(f"[OK] Passed: {passed}")
    print(f"[OK] Failed: {total - passed}")
    print(f"[OK] Fidelity mean: {output['fidelity_mean']:.3f}")
    print(f"[OK] Alignment mean: {output['alignment_mean']:.3f}")
    print(f"[OK] Scores: {score_file}")


if __name__ == "__main__":
    main()