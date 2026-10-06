import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.prepare_human_eval import (
    RATING_COLUMNS,
    aggregate,
    build_sample_set,
    load_protocol,
    make_html,
)


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(tmp_path, *, placeholder=False):
    protocol = load_protocol(ROOT / "configs/human_eval_v1.yaml")
    prompts = yaml.safe_load((ROOT / "prompt_bank/evaluation_prompts.yaml").read_text())
    bank = tmp_path / "prompts.yaml"
    bank.write_text(yaml.safe_dump(prompts, sort_keys=False))
    artifacts = tmp_path / "artifacts"
    refs = tmp_path / "refs"
    run_map = []
    for concept in protocol["concepts"]:
        folder = refs / concept
        folder.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (16, 16), (i * 20, 10, 10)).save(folder / f"ref{i}.png")
        for config in protocol["representative_configurations"]:
            run_id = f"{concept}_{config}"
            run_map.append({"concept_id": concept, "config_id": config, "run_id": run_id})
            gen = artifacts / run_id / "generated"
            gen.mkdir(parents=True)
            records = []
            for prompt_id in set(protocol["prompt_ids_by_category"].values()):
                image = gen / f"{prompt_id}.png"
                Image.new("RGB", (16, 16), (20, 30, 40)).save(image)
                records.append({
                    "run_id": run_id, "concept_id": concept, "prompt_id": prompt_id,
                    "seed": 11, "image_path": str(image),
                    "generation_mode": "cpu_smoke_placeholder" if placeholder and run_id.endswith("n1-r16") else "lora",
                })
            (gen / "metadata.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    map_path = tmp_path / "run_map.csv"
    with map_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_id", "config_id", "run_id"])
        writer.writeheader()
        writer.writerows(run_map)
    selected, key = build_sample_set(
        protocol=protocol, run_map_path=map_path, artifacts_root=artifacts,
        eval_refs_root=refs, prompt_bank_path=bank, repo_root=tmp_path,
    )
    return protocol, artifacts, refs, map_path, bank, selected, key


def test_stratified_sample_is_exactly_60_and_balanced(tmp_path):
    protocol, *_rest, selected, key = make_fixture(tmp_path)
    assert len(selected) == len(key) == 60
    counts = {}
    for row in key:
        group = (row["concept_id"], row["config_id"], row["prompt_category"])
        counts[group] = counts.get(group, 0) + 1
    assert len(counts) == 3 * 5 * 4
    assert set(counts.values()) == {1}
    assert {r["seed"] for r in key} == {protocol["fixed_generation_seed"]}


def test_placeholder_outputs_are_rejected(tmp_path):
    protocol = load_protocol(ROOT / "configs/human_eval_v1.yaml")
    prompts = yaml.safe_load((ROOT / "prompt_bank/evaluation_prompts.yaml").read_text())
    bank = tmp_path / "prompts.yaml"
    bank.write_text(yaml.safe_dump(prompts, sort_keys=False))
    artifacts, refs = tmp_path / "artifacts", tmp_path / "refs"
    run_map = []
    for concept in protocol["concepts"]:
        folder = refs / concept
        folder.mkdir(parents=True)
        for i in range(3): Image.new("RGB", (8, 8)).save(folder / f"{i}.png")
        for config in protocol["representative_configurations"]:
            run_id = f"{concept}_{config}"
            run_map.append({"concept_id": concept, "config_id": config, "run_id": run_id})
            generated = artifacts / run_id / "generated"
            generated.mkdir(parents=True)
            record = {"run_id": run_id, "concept_id": concept, "prompt_id": "p01", "seed": 11,
                      "image_path": "x.png", "generation_mode": "cpu_smoke_placeholder"}
            (generated / "metadata.jsonl").write_text(json.dumps(record) + "\n")
    map_path = tmp_path / "map.csv"
    with map_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_id", "config_id", "run_id"])
        writer.writeheader(); writer.writerows(run_map)
    with pytest.raises(ValueError, match="placeholder"):
        build_sample_set(protocol=protocol, run_map_path=map_path, artifacts_root=artifacts,
                         eval_refs_root=refs, prompt_bank_path=bank, repo_root=tmp_path)


def test_html_contains_no_configuration_identity_and_randomized_packet_data():
    html = make_html([{"id": "S-ABC", "prompt": "a toy", "image": "toy.png", "refs": ["ref.png"]}], ["identity_drift"])
    assert "configuration" in html.lower()
    assert "n5-r16" not in html
    assert "toy.png" in html and "ref.png" in html
    assert "getRandomValues" in html


def test_aggregate_keeps_source_unchanged_and_reports_missing(tmp_path):
    key_path = tmp_path / "key.csv"
    key_rows = [{"blind_id": f"S-{i:03}", "sample_id": f"run__p{i}__gs11",
                 "concept_id": "cat_mug", "config_id": "n1-r16", "prompt_category": "simple", "seed": "11"}
                for i in range(60)]
    with key_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0])); writer.writeheader(); writer.writerows(key_rows)
    source = tmp_path / "ratings.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATING_COLUMNS); writer.writeheader()
        for rater in range(5):
            for sample in key_rows[:2]:
                writer.writerow({"rating_id": f"{rater}-{sample['blind_id']}", "anonymous_rater_id": f"R{rater}",
                                 "sample_id": sample["blind_id"], "subject_fidelity_1_5": "4",
                                 "prompt_alignment_1_5": "5", "visual_quality_1_5": "3",
                                 "failure_tags": "", "comment_optional": ""})
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "aggregate.csv"
    args = type("Args", (), {"ratings": source, "output": output, "blind_key": key_path,
                              "minimum_raters": 5, "rater_target": 8, "allow_incomplete": False})()
    with pytest.raises(ValueError, match="complete ratings"):
        aggregate(args)
    args.allow_incomplete = True
    aggregate(args)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    assert rows[0]["complete_rating_count"] == "5"
    assert rows[0]["missing_rating_count"] == "0"
    assert rows[-1]["complete_rating_count"] == "0"
