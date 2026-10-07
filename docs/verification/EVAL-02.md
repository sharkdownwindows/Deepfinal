# EVAL-02 Verification — CLIP Prompt Alignment

## Implemented

- Loads the pinned CLIP ViT-B/32 processor and model revision.
- Replaces the concept's unique token with its class noun before text encoding.
- Exports per-sample CLIP cosine scores alongside run, concept, prompt, seed,
  generation mode, rank, and data-size metadata.
- Retains invalid samples with an explicit reason and updates only CLIP-owned
  metric fields, preserving DINO results already stored for that sample.

## Verification evidence

`tests/test_alignment.py` covers token normalization for all three concepts,
finite smoke scores using an injected scorer, per-sample export, invalid-image
reporting, run-ID checks, and metric upserts. No pretrained CLIP weights were
downloaded for these tests.

Run real scoring after a generated run exists:

```bash
python scripts/evaluate_outputs.py --run-id <run_id> --metric clip
```
