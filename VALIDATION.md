# Validation status

The initial implementation was validated locally before cloud execution. The
user subsequently ran it on Modal. Their November 5 log shows Task 2 evaluation
completed, followed by a CPU statistics failure on a stale Volume snapshot.
No cloud jobs were launched as part of this repair.

## November 5 launcher repair

- Refresh the Volume before every child job and on orchestrator entry.
- Propagate cloud failures to the CLI instead of falsely reporting continuation.
- Regression coverage simulates warm CPU/GPU workers receiving new committed
  outputs, failed storage refreshes, remote exception propagation, and CPU-only
  statistics from saved predictions without modifying or regenerating them.
- Experiment source files and their cache signatures are unchanged by this fix.
- Actual cloud recovery remains to be checked with the CPU-only command in README.
- All 37 selected CPU tests passed (28.8 seconds of test execution), including
  the four new regression tests. Compilation/global-reference checks passed.
  No GPU tests or paid cloud jobs were run for this repair.

## Passed locally

- `python scripts/check_code.py`: 19 experiment modules compile, with no unresolved
  module-global references; the Modal launcher also compiles and imports.
- `python -m unittest discover -v`: **34 tests passed** in the final local run.
- A tiny CUDA model completed one FP16-autocast update with FP32 LoRA parameters
  and FP32 optimizer moments. No pretrained weights were needed for this test.
- Exact interrupted/uninterrupted CPU training agreement, optimizer/RNG recovery,
  fallback from a corrupt newest checkpoint, and permanent-checkpoint auditing.
- Assistant-token masks, token/example accumulation, L2-SP tied-parameter handling,
  double-sided projection constraints, factor initialization, and adapter merging.
- A tiny real Qwen3 model class verified connector gradients through the frozen,
  checkpointed backbone and cached generation from visual embeddings.
- The actual Qwen tokenizer and real harness API verified chat formatting, both
  GSM8K answer-filter records, truncation metadata, and reuse of cached requests.
- Synthetic complete offline/adaptive pipelines checked prefix-local decisions,
  development-only threshold selection, paired resampling, and result reuse.
- Actual Math-Verify evaluated known small mathematical-equivalence examples.
  Windows tests bypass its multiprocessing timeout only for those fixed trusted
  expressions; production scoring requires bounded calls in a Linux main process.
- Parent/child persistence acknowledgements survived progress-bar output, rejected
  incorrect tokens, and terminated the child on a simulated storage failure.
- Unicode CSV/Parquet round trips, spreadsheet-formula escaping, HTML escaping,
  ZIP integrity, explicit incomplete-result counts, and preservation of manual
  annotation fields.
- Public COCO HTTPS endpoints, complementary-pair metadata format, the assignment
  Drive listing, official VQA contraction data, and remaining dataset identifiers.

The local runtime was Windows, Python 3.14.6, PyTorch 2.11.0+cu128, Transformers
4.57.1, and an NVIDIA GeForce RTX 5060 Laptop GPU. The Modal target is Python 3.11
and the versions pinned in `requirements.txt`. The actual cloud environment still
needs its short preflight; local tests cannot substitute for that check.

## Required next check, after adding credits

Follow the README's launcher installation and authentication instructions, then:

```powershell
py -m modal run --detach modal_app.py --command quick-check
```

This downloads models on CPUs before allocating a GPU and caches each successful
GPU subcheck. It does not run the full assignment benchmarks. Production commands
refuse to proceed without a matching successful preflight receipt.

After it passes, use the README's complete-run command. Checkpoints and individual
completed result units are reusable. A hard failure can still repeat work since
the last committed checkpoint/unit; zero lost computation cannot be promised.
