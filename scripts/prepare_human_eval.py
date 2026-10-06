"""Prepare blinded human-rating packets and aggregate immutable source ratings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

RATING_COLUMNS = [
    "rating_id", "anonymous_rater_id", "sample_id",
    "subject_fidelity_1_5", "prompt_alignment_1_5",
    "visual_quality_1_5", "failure_tags", "comment_optional",
]
METADATA_NAME = "metadata.jsonl"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_protocol(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sample_id(run_id: str, prompt_id: str, seed: int) -> str:
    return f"{run_id}__{prompt_id}__gs{seed}"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            records.append(row)
    return records


def _resolve_image(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def build_sample_set(
    *, protocol: dict, run_map_path: Path, artifacts_root: Path,
    eval_refs_root: Path, prompt_bank_path: Path, repo_root: Path,
) -> tuple[list[dict], list[dict]]:
    prompt_bank = yaml.safe_load(prompt_bank_path.read_text(encoding="utf-8")) or {}
    if prompt_bank.get("version") != protocol["prompt_bank_version"]:
        raise ValueError("Prompt-bank version does not match the frozen human-evaluation plan.")

    run_map = read_csv(run_map_path)
    required_map_cols = {"concept_id", "config_id", "run_id"}
    if not run_map or not required_map_cols.issubset(run_map[0]):
        raise ValueError("Run map must have columns: concept_id, config_id, run_id.")
    expected_pairs = {
        (concept, config)
        for concept in protocol["concepts"]
        for config in protocol["representative_configurations"]
    }
    pairs = [(r["concept_id"], r["config_id"]) for r in run_map]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError(
            f"Run map must contain each of the {len(expected_pairs)} concept/config pairs exactly once."
        )
    run_ids = [r["run_id"] for r in run_map]
    if any(not x for x in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("Run IDs in the run map must be non-empty and unique.")

    selected_prompts: dict[str, dict] = {}
    for concept in protocol["concepts"]:
        prompts = prompt_bank.get("concepts", {}).get(concept, [])
        by_id = {p.get("prompt_id"): p for p in prompts}
        for category, prompt_id in protocol["prompt_ids_by_category"].items():
            prompt = by_id.get(prompt_id)
            if not prompt or prompt.get("category") != category:
                raise ValueError(f"Frozen selection {concept}/{category}/{prompt_id} mismatches prompt bank.")
            selected_prompts[f"{concept}:{category}"] = prompt

    refs: dict[str, list[Path]] = {}
    for concept in protocol["concepts"]:
        folder = eval_refs_root / concept
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if folder.exists() else []
        if len(paths) != 3:
            raise ValueError(f"Expected exactly 3 held-out reference images for {concept}; found {len(paths)} in {folder}.")
        refs[concept] = paths

    selected, key_rows = [], []
    fixed_seed = int(protocol["fixed_generation_seed"])
    for run in run_map:
        concept, config_id, run_id = run["concept_id"], run["config_id"], run["run_id"]
        metadata_path = artifacts_root / run_id / "generated" / METADATA_NAME
        if not metadata_path.is_file():
            raise ValueError(f"Missing generated metadata for {run_id}: {metadata_path}")
        records = _load_jsonl(metadata_path)
        candidates = {}
        for record in records:
            if record.get("run_id") != run_id or record.get("concept_id") != concept:
                continue
            if record.get("generation_mode") == "cpu_smoke_placeholder":
                raise ValueError(f"{run_id} contains CPU smoke placeholders; they cannot enter human evaluation.")
            key = (record.get("prompt_id"), int(record.get("seed", -1)))
            if key in candidates:
                raise ValueError(f"Duplicate generated output for {run_id}, {key}.")
            candidates[key] = record

        for category, prompt_id in protocol["prompt_ids_by_category"].items():
            record = candidates.get((prompt_id, fixed_seed))
            if record is None:
                raise ValueError(f"Missing required sample {run_id}/{prompt_id}/seed={fixed_seed}.")
            image_path = _resolve_image(record.get("image_path", ""), repo_root)
            if not image_path.is_file():
                raise ValueError(f"Missing generated image for {run_id}/{prompt_id}: {image_path}")
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable image for {run_id}/{prompt_id}: {image_path}") from exc

            sid = sample_id(run_id, prompt_id, fixed_seed)
            blind_id = "S-" + secrets.token_hex(6).upper()
            prompt = selected_prompts[f"{concept}:{category}"]
            if record.get("prompt") and record["prompt"] != prompt["prompt"]:
                raise ValueError(f"Generated prompt differs from frozen DATA-04 input for {run_id}/{prompt_id}.")
            selected.append({
                "blind_id": blind_id,
                "prompt": prompt["prompt"],
                "generated_source": image_path,
                "reference_sources": refs[concept],
                "concept_id": concept,
                "config_id": config_id,
            })
            key_rows.append({
                "blind_id": blind_id,
                "sample_id": sid,
                "concept_id": concept,
                "config_id": config_id,
                "run_id": run_id,
                "prompt_id": prompt_id,
                "prompt_category": category,
                "seed": fixed_seed,
                "source_image": str(image_path),
            })

    if len(selected) != 60 or len({r["blind_id"] for r in selected}) != 60:
        raise ValueError(f"Frozen design requires 60 unique samples; selected {len(selected)}.")
    return selected, key_rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_html(task_rows: list[dict], tags: list[str]) -> str:
    tasks_json = json.dumps(task_rows, ensure_ascii=False).replace("</", "<\\/")
    tags_json = json.dumps(tags, ensure_ascii=False)
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind image rating</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#172033}}
.refs{{display:flex;gap:.7rem;flex-wrap:wrap}} img{{max-width:100%;object-fit:contain;background:#eee}}
.ref{{width:130px}} .generated{{max-width:640px;max-height:560px;display:block;margin:1rem auto}}
.card{{border:1px solid #ccd3df;border-radius:12px;padding:1.2rem;margin:1rem 0}}
fieldset{{margin:.8rem 0;border:0;padding:0}} label{{margin-right:1rem;white-space:nowrap}}
button{{padding:.65rem 1rem;margin:.4rem}} .muted{{color:#536179}} .hidden{{display:none}}
</style>
<main><h1>Blind image rating</h1>
<p>Rate the generated image against the reference photos and the prompt. Configuration identity is hidden. Use the same anonymous rater code supplied by the study organizer if you resume later. Do not enter your name or email.</p>
<label>Anonymous rater code <input id="rater" autocomplete="off" placeholder="e.g. R01"></label>
<p id="progress" class="muted"></p><section id="task" class="card hidden"></section>
<button id="prev" type="button">Previous</button><button id="save" type="button">Save and next</button>
<button id="export" type="button">Download ratings CSV</button>
<p id="notice" role="status"></p></main>
<script>
const tasks={tasks_json}; const allowedTags={tags_json}; const keyPrefix='human-eval-v1:';
let state={{order:null,ratings:{{}},index:0,rater:''}};
const root=document.getElementById('task'), rater=document.getElementById('rater');
function persist(){{if(state.rater)localStorage.setItem(keyPrefix+state.rater,JSON.stringify(state));}}
function startForRater(code){{const saved=localStorage.getItem(keyPrefix+code);state=saved?JSON.parse(saved):{{order:[...tasks.keys()],ratings:{{}},index:0,rater:code}};if(!state.order)state.order=[...tasks.keys()];if(!saved)for(let i=state.order.length-1;i>0;i--){{const j=crypto.getRandomValues(new Uint32Array(1))[0]%(i+1);[state.order[i],state.order[j]]=[state.order[j],state.order[i]];}}state.rater=code;persist();draw();}}
rater.value=''; rater.addEventListener('change',()=>{{const code=rater.value.trim();if(code)startForRater(code);}});
function draw(){{const i=Math.max(0,Math.min(state.index,state.order.length-1));state.index=i;const t=tasks[state.order[i]], old=state.ratings[t.id]||{{}};root.classList.remove('hidden');
document.getElementById('progress').textContent=`Sample ${{i+1}} of ${{tasks.length}}`;
root.innerHTML=`<h2>Prompt</h2><p>${{t.prompt}}</p><h3>Held-out reference photos</h3><div class="refs">${{t.refs.map((x,n)=>`<figure class="ref"><img src="${{x}}"><figcaption>Reference ${{n+1}}</figcaption></figure>`).join('')}}</div><h3>Generated image</h3><img class="generated" src="${{t.image}}"><p><strong>Blinded sample:</strong> ${{t.id}}</p>${{rating('Subject fidelity','subject_fidelity_1_5',old.subject_fidelity_1_5)}}${{rating('Prompt adherence','prompt_alignment_1_5',old.prompt_alignment_1_5)}}${{rating('Visual quality / artifacts','visual_quality_1_5',old.visual_quality_1_5)}}<fieldset><legend>Failure tags (select all that apply)</legend>${{allowedTags.map(tag=>`<label><input type="checkbox" name="tag" value="${{tag}}" ${{(old.tags||[]).includes(tag)?'checked':''}}> ${{tag}}</label>`).join('')}}</fieldset><label>Optional comment <textarea id="comment">${{old.comment||''}}</textarea></label>`;
document.getElementById('prev').disabled=i===0;persist();}}
function rating(label,key,value){{return `<fieldset><legend>${{label}} (1=poor, 5=excellent)</legend>${{[1,2,3,4,5].map(n=>`<label><input type="radio" name="${{key}}" value="${{n}}" ${{String(value)===String(n)?'checked':''}}> ${{n}}</label>`).join('')}}</fieldset>`}}
function capture(){{const t=tasks[state.order[state.index]], read=k=>root.querySelector(`input[name="${{k}}"]:checked`)?.value||'';state.ratings[t.id]={{subject_fidelity_1_5:read('subject_fidelity_1_5'),prompt_alignment_1_5:read('prompt_alignment_1_5'),visual_quality_1_5:read('visual_quality_1_5'),tags:[...root.querySelectorAll('input[name="tag"]:checked')].map(x=>x.value).join(';'),comment:root.querySelector('#comment')?.value||''}};persist();}}
document.getElementById('save').onclick=()=>{{if(!state.rater){{document.getElementById('notice').textContent='Enter the anonymous rater code and press Tab first.';return;}}capture();if(state.index<state.order.length-1)state.index++;draw();}};
document.getElementById('prev').onclick=()=>{{capture();state.index--;draw();}};
document.getElementById('export').onclick=()=>{{if(!state.rater){{document.getElementById('notice').textContent='Enter the anonymous rater code first.';return;}}capture();const header={json.dumps(RATING_COLUMNS)};const lines=[header.join(',')];for(const i of state.order){{const t=tasks[i],a=state.ratings[t.id]||{{}}, vals=[crypto.randomUUID(),state.rater,t.id,a.subject_fidelity_1_5||'',a.prompt_alignment_1_5||'',a.visual_quality_1_5||'',a.tags||'',a.comment||''];lines.push(vals.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(','));}}const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv;charset=utf-8'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='human_ratings_'+state.rater+'.csv';link.click();URL.revokeObjectURL(link.href);document.getElementById('notice').textContent='CSV downloaded. Incomplete ratings remain blank; do not fill guesses.';}};
root.innerHTML='<p>Enter your anonymous rater code above to begin.</p>';
</script></html>'''


def prepare(args: argparse.Namespace) -> None:
    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite blind key or packet: {out}")
    protocol = load_protocol(args.protocol)
    samples, key_rows = build_sample_set(
        protocol=protocol, run_map_path=args.run_map, artifacts_root=args.artifacts_root,
        eval_refs_root=args.eval_refs_root, prompt_bank_path=args.prompt_bank,
        repo_root=args.repo_root,
    )
    assets = out / "packet" / "assets"
    private = out / "private"
    assets.mkdir(parents=True)
    private.mkdir(parents=True)
    task_rows = []
    for sample in samples:
        blind_id = sample["blind_id"]
        generated_name = f"{blind_id}.generated.png"
        with Image.open(sample["generated_source"]) as image:
            image.convert("RGB").save(assets / generated_name, format="PNG", optimize=True)
        ref_names = []
        for index, source in enumerate(sample["reference_sources"], 1):
            target_name = f"{blind_id}.ref{index:02d}.jpg"
            with Image.open(source) as image:
                image.convert("RGB").save(assets / target_name, format="JPEG", quality=92, optimize=True)
            ref_names.append(f"assets/{target_name}")
        task_rows.append({
            "id": blind_id, "prompt": sample["prompt"],
            "image": f"assets/{generated_name}", "refs": ref_names,
        })
    tags = protocol.get("failure_tags", [
        "identity_drift", "background_leakage", "pose_copy", "prompt_refusal",
        "memorization", "structure_artifact", "rendering_artifact", "other",
    ])
    (out / "packet" / "index.html").write_text(
        make_html(task_rows, tags), encoding="utf-8"
    )
    _write_csv(private / "blind_key.csv", list(key_rows[0]), key_rows)
    (private / "packet_manifest.json").write_text(json.dumps({
        "protocol_version": protocol["protocol_version"],
        "sample_count": len(samples),
        "concept_count": len(protocol["concepts"]),
        "configuration_count": len(protocol["representative_configurations"]),
        "prompt_categories": protocol["prompt_ids_by_category"],
        "fixed_generation_seed": protocol["fixed_generation_seed"],
        "target_raters": args.rater_target,
        "minimum_raters": protocol["minimum_raters"],
        "source_run_map_sha256": hashlib.sha256(args.run_map.read_bytes()).hexdigest(),
        "source_prompt_bank_sha256": hashlib.sha256(args.prompt_bank.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    package_path = out / "human_eval_v1_blind_packet.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out / "packet").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out / "packet"))
    print(f"Prepared {len(samples)} blinded samples: {package_path}")
    print(f"Private identity key (do not distribute): {private / 'blind_key.csv'}")
    print(f"Target raters: {args.rater_target}; minimum: {protocol['minimum_raters']}")


def aggregate(args: argparse.Namespace) -> None:
    source = args.ratings.resolve()
    output = args.output.resolve()
    key_path = args.blind_key.resolve()
    if source == output or source == key_path or output == key_path:
        raise ValueError("Ratings source, derived output and blind key must be separate files.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite derived aggregate: {output}")
    ratings = read_csv(source)
    if not ratings or not set(RATING_COLUMNS).issubset(ratings[0]):
        raise ValueError(f"Ratings file must contain columns: {', '.join(RATING_COLUMNS)}")
    key = read_csv(key_path)
    key_by_blind = {r["blind_id"]: r for r in key}
    if len(key_by_blind) != len(key) or len(key_by_blind) != 60:
        raise ValueError("Blind key must contain 60 unique sample IDs.")
    raters = {r["anonymous_rater_id"].strip() for r in ratings if r.get("anonymous_rater_id", "").strip()}
    if len(raters) < args.minimum_raters:
        raise ValueError(f"Need at least {args.minimum_raters} distinct raters; found {len(raters)}.")
    seen = set()
    values: dict[str, list[dict]] = defaultdict(list)
    for row in ratings:
        rid, rater, sid = (row.get(c, "").strip() for c in ("rating_id", "anonymous_rater_id", "sample_id"))
        if not rid or not rater or sid not in key_by_blind:
            raise ValueError("Every source row needs a rating ID, anonymous rater ID and known blinded sample ID.")
        if (rater, sid) in seen:
            raise ValueError(f"Duplicate rating from {rater} for {sid}.")
        seen.add((rater, sid))
        values[sid].append(row)

    dimensions = ["subject_fidelity_1_5", "prompt_alignment_1_5", "visual_quality_1_5"]
    insufficient = []
    for sid in key_by_blind:
        complete = 0
        for row in values.get(sid, []):
            if all(row.get(column, "").strip() for column in dimensions):
                complete += 1
        if complete < args.minimum_raters:
            insufficient.append(f"{sid}={complete}")
    if insufficient and not args.allow_incomplete:
        preview = ", ".join(insufficient[:8])
        suffix = " ..." if len(insufficient) > 8 else ""
        raise ValueError(
            f"Each sample needs {args.minimum_raters} complete ratings; "
            f"{len(insufficient)} samples are below coverage: {preview}{suffix}. "
            "Use --allow-incomplete only for a progress report."
        )
    result = []
    for sid, key_row in key_by_blind.items():
        rows = values.get(sid, [])
        dimension_scores = {name: [] for name in dimensions}
        tag_counts = Counter()
        complete_ratings = 0
        for row in rows:
            score_values = []
            for dimension in dimensions:
                raw = row.get(dimension, "").strip()
                if raw:
                    try:
                        score = int(raw)
                    except ValueError as exc:
                        raise ValueError(f"Invalid score {raw!r} for {sid}.") from exc
                    if not 1 <= score <= 5:
                        raise ValueError(f"Score must be 1..5 for {sid}.")
                    dimension_scores[dimension].append(score)
                    score_values.append(score)
                else:
                    score_values.append(None)
            if all(x is not None for x in score_values):
                complete_ratings += 1
            tag_counts.update(t for t in row.get("failure_tags", "").split(";") if t)
        result.append({
            "sample_id": key_row["sample_id"], "concept_id": key_row["concept_id"],
            "config_id": key_row["config_id"], "prompt_category": key_row["prompt_category"],
            "seed": key_row["seed"], "rater_count": len(rows),
            "complete_rating_count": complete_ratings,
            "missing_rating_count": max(0, len(raters) - complete_ratings),
            **{f"{name}_mean": round(statistics.mean(scores), 4) if scores else "" for name, scores in dimension_scores.items()},
            **{f"{name}_sd": round(statistics.stdev(scores), 4) if len(scores) > 1 else (0 if scores else "") for name, scores in dimension_scores.items()},
            "failure_tag_counts": ";".join(f"{tag}:{count}" for tag, count in sorted(tag_counts.items())),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(result[0])
    _write_csv(output, columns, result)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"Distinct raters: {len(raters)}; blinded samples in key: {len(key_by_blind)}")
    print(f"Wrote derived aggregates to {output}; source ratings left unchanged (sha256 {before}).")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build a 60-sample blinded packet from completed real runs.")
    prep.add_argument("--protocol", type=Path, default=Path("configs/human_eval_v1.yaml"))
    prep.add_argument("--run-map", type=Path, required=True, help="Private CSV: concept_id,config_id,run_id (15 rows).")
    prep.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    prep.add_argument("--eval-refs-root", type=Path, default=Path("data/eval_refs"))
    prep.add_argument("--prompt-bank", type=Path, default=Path("prompt_bank/evaluation_prompts.yaml"))
    prep.add_argument("--repo-root", type=Path, default=Path("."))
    prep.add_argument("--output-root", type=Path, default=Path("artifacts/human_eval_v1"))
    prep.add_argument("--rater-target", type=int, default=8)
    prep.set_defaults(func=prepare)
    agg = sub.add_parser("aggregate", help="Create a separate aggregate without modifying source ratings.")
    agg.add_argument("--ratings", type=Path, default=Path("results/human_ratings.csv"))
    agg.add_argument("--blind-key", type=Path, required=True, help="Private key generated by prepare.")
    agg.add_argument("--output", type=Path, default=Path("results/human_ratings_aggregate.csv"))
    agg.add_argument("--minimum-raters", type=int, default=5)
    agg.add_argument("--rater-target", type=int, default=8)
    agg.add_argument("--allow-incomplete", action="store_true", help="Write a progress aggregate below five complete ratings per sample.")
    agg.set_defaults(func=aggregate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
