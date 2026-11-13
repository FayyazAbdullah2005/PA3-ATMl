# ATML Programming Assignment 3

Reproducible code for all three tasks, including the optional 49-token VLM run.
The code generates numerical evidence, figures, raw predictions and review sheets.
It does **not** write report text, interpretations, captions for the report, or LaTeX.
Manual qualitative annotations remain your work.

## Start here on Windows

Add your Modal credits before running any `modal run` command below. Local tests
and imports do not launch remote jobs automatically.

Use a terminal in this assignment folder. Only the small Modal launcher is installed
locally; the experiment environment is built and run on Modal with Python 3.11.

```powershell
py -m pip install -r requirements-launcher.txt
py -m modal setup
py -m modal run --detach modal_app.py --command quick-check
```

The last command first downloads model files on a **CPU** and runs the CPU test
suite. It then performs **one short GPU preflight session**: one update for each
fine-tuning path, one real OPLoRA matrix projection, one frozen SigLIP forward,
three tiny VLM stage updates for each token setting, and two-token generation
probes. One Full-FT + L2-SP microbatch uses the required maximum sequence length
to test the largest memory path. There are no benchmark sweeps or full-dataset
training runs in this check. Passing subchecks are persisted and reused.

Once it passes, launch the complete assignment pipeline:

```powershell
py -m modal run --detach modal_app.py --command all --run-id main
```

`--detach` lets the remote work continue after you close the terminal or laptop.
Follow progress and failures in the Modal app's logs. The pipeline uses one GPU
at a time. It includes the optional 49-token rerun and full required evaluations.
Those production budgets are not shortened: doing so would change the assignment.
All GPU commands are gated by the saved preflight receipt.

For a CPU-only preparation run before committing GPU time:

```powershell
py -m modal run --detach modal_app.py --command prepare
```

This prepares all data and model caches, and creates the Task 2 label-review
gallery. Images are downloaded individually in parallel, only for selected IDs;
the large COCO image archives are not needed. Task 3 is mostly CPU analysis of
the supplied rollouts; only its 50-question live comparison needs a GPU.

## Recovery and failure behavior

- Rerun the **same command and run ID** after an interruption. Valid completed
  units are reused, and training resumes from the newest verified checkpoint.
- Every 25 completed training updates, save a complete recovery state. Keep two
  alternating recovery files and permanent assignment milestones. State includes
  model/adapter parameters, optimizer, schedule position, dynamic scaler, RNG,
  data position, curriculum stage and mixture counts.
- Files are written to temporary siblings, flushed, atomically replaced, and
  checksummed. `DONE.json` is written last. A parent/child acknowledgement protocol
  pauses the worker until Modal confirms the Volume commit.
- Do not accept a corrupt newest checkpoint: automatically try an older verified
  one. Never treat a partial download or partial result as completed.
- Cache SVDs per matrix, SigLIP features per shard, Task 3 results per question,
  live responses individually, and final VLM evaluations per eight-answer batch.
  Benchmark requests are cached in small batches, including raw generation IDs.
- Retry AMP overflow on the same effective batch without consuming more examples
  or advancing the schedule. Persist skip events. Deterministic exceptions fail
  immediately rather than automatically wasting several GPU retries.
- A sudden failure can repeat work since the **last successfully committed
  checkpoint/unit**. Zero lost work under every hardware or storage failure cannot
  be guaranteed. Saving a full optimizer state after every token/update would add
  substantial I/O and storage cost.
- Configuration and relevant implementation fingerprints prevent silently mixing
  incompatible results. Use a new run ID when intentionally changing the recipe.
  Plotting/export changes do not invalidate trained model states.
- Do not run competing full pipelines under different GPU overrides against the
  same run ID. Normal commands are serialized by the orchestrator and workers.

If the default A10G cannot fit the prescribed microbatch, preserve the recipe and
rerun on a larger GPU:

```powershell
py -m modal run --detach modal_app.py --command all --run-id main --gpu A100-40GB
```

The GPU preflight's environment/code receipt remains mandatory. Training learning
rates, effective batch sizes, precision and quantization policy are never changed
automatically to hide an out-of-memory error.

## Individual commands

Pass any command below as `--command NAME`. All commands also accept `--run-id`.

| Command | Resources / purpose |
|---|---|
| `doctor` | CPU environment information |
| `tests` | CPU numerical/recovery/interface tests |
| `prepare-models` | CPU model downloads and pinned identities |
| `prepare-benchmarks` | CPU benchmark datasets, processing, and metric initialization |
| `task1-prepare` | CPU MetaMath filtering, prompt masks, probe IDs |
| `task1-train --method full` | GPU; methods: `full`, `l2sp`, `lora4`, `lora16`, `oplora` |
| `task1-evaluate-condition --condition baseline` | GPU; one model load for all four benchmarks and qualitative responses |
| `task1-eval --condition full-mid --task gsm8k_cot` | GPU; one specific benchmark |
| `task1-diagnostics --condition full` | GPU; weight and activation diagnostics |
| `task2-prepare` | CPU COCO/VQAv2 selection and downloads |
| `label-gallery` | CPU label-review HTML and blank review CSV |
| `task2-features` | GPU; one shared 196-token SigLIP feature cache |
| `task2-train --tokens 196` | GPU; complete curriculum, or `--tokens 49` |
| `task2-eval --tokens 196` | GPU final controls, followed automatically by CPU pair statistics |
| `task2-statistics --tokens 196` | CPU; 4,000 complete-pair bootstraps |
| `task3-prepare` | CPU supplied-data validation, joins, scoring, splits |
| `task3-offline` | CPU fixed-budget selection and signal diagnostics |
| `task3-adaptive` | CPU development selection and held-out stopping simulation |
| `task3-live` | GPU; 50 questions, 250 unique calls with shared-initial reuse |
| `summarize` | CPU CSV/JSON/Parquet exports, figures and galleries |
| `status` | CPU list of completed units |
| `audit` | CPU checksum verification of completed artifacts |
| `verify-complete` | CPU evidence counts and required experiment completeness |

Task 1 conditions are `baseline`, `full-mid`, `full`, `l2sp`, `lora4`, `lora16`,
and `oplora`. Benchmarks are `gsm8k_cot`, `hellaswag`, `boolq`, and `arc_easy`.

For example:

```powershell
py -m modal run --detach modal_app.py --command task1-train --method oplora
py -m modal run --detach modal_app.py --command task2-train --tokens 49
py -m modal run --detach modal_app.py --command summarize
py -m modal run modal_app.py --command status
```

`--smoke` is available for reduced experimental/debugging runs, kept in separate
directories. Prefer `quick-check` for the shortest practical compatibility check;
smoke outputs never count as assignment evidence.

## Results you can read

Results live on the persistent Volume `atml-pa3-artifacts`, independently of any
container or laptop session. After `summarize`, download the readable bundle:

```powershell
py -m modal volume get atml-pa3-artifacts exports/main.zip results-main.zip
```

Extract it and open `index.html`. Tables are UTF-8 CSV and JSON, detailed numeric
records are also Parquet, and figures are PNG/PDF. Long responses are displayed
in HTML with aligned model/condition labels. Blank annotation fields are not
filled by an AI judge. Original model strings remain in JSON/Parquet; CSV exports
escape spreadsheet formula prefixes for safe opening in Excel.

The inventory explicitly lists expected counts and available artifacts. Missing
results remain missing; no placeholder or simulated score is presented as a real
experiment. Synthetic validation files live under `checks/` or local `artifacts/`,
not under the main assignment results.

## Experimental defaults

All prescribed constants are in `pa3/config.py`; each run saves its complete
configuration and installed package versions. Model/dataset revisions are resolved
once and saved in `data/pins.json`. The Task 3 live model and MATH dataset identities
are fixed to the revisions in the supplied README.

- Task 1: 40,000 source-ordered eligible examples, 1,250 updates, microbatch 2 ×
  accumulation 16, assistant/EOT **token-weighted** loss. No old-task replay.
  AdamW decay excludes biases and parameters with fewer than two dimensions.
  L2-SP includes all unique parameters, is not averaged over parameter count,
  and is distinct from AdamW weight decay. OPLoRA exports projected factors.
- Task 2: three stages of 2,250/1,500/750 updates, microbatch 8 × accumulation 2,
  **example-weighted** supervised loss. The connector always learns; original
  Qwen/SigLIP weights stay frozen. AdamW moments continue across stages, while
  warmup/cosine schedules restart. Warmup updates are rounded upward. Caption
  alignment uses two seeded permutations; paired draws are preserved later.
  The optional 49-token run uses spatial 2×2 pooling before a fresh connector.
- Task 2 data: filter over-cap content before selection rather than truncating
  labels; use equal presence/counting evaluation-pair counts; exclude crowd
  annotations from counting labels; keep all evaluation suites image-disjoint.
  These choices are recorded in the split manifest. Inspect the label gallery
  independently of model predictions.
- Task 3: save the exact seed-6304 splits and ten nested candidate orders; use
  prefix-local calibration/grouping; keep missing extractions in accounting.
  Gold correctness is excluded from selector inputs. Tune stopping thresholds on
  development questions only. Bootstrap questions after averaging orders.
- CUDA computation uses FP16 autocast, dynamic scaling, FP32 trainable parameters
  and optimizer states, SDPA attention, and no quantization. Frozen backbones are
  FP16. Gradient clipping follows unscaling. No distributed training is used.

## Local tests and validation limits

```powershell
py -m unittest discover -v
```

Local numerical tests use tiny synthetic models, including the real Qwen3 model
class. They check gradient accumulation, response masks, projection constraints,
merge equivalence, tied parameters, deterministic checkpoint recovery, image
embedding gradients/cached generation, spatial pooling, answer extraction, ties,
prefix calibration, covariance fusion, and stopping causality.

The production Math-Verify scorer requires a Linux main process for bounded signal
timeouts; Modal workers deliberately execute experiments in child main processes.
Do not run Task 3 scoring directly in Windows or a notebook worker thread.

A passing local suite does not prove that a paid production run has passed. The
Modal preflight and real data validations are separate requirements. The code
cannot guarantee that every future service outage or library change is harmless.
Keep the pinned environment and use the saved verification evidence.

## Repository submission and attribution

Create your public GitHub repository with this code, requirements, README, selected
IDs/configuration manifests, and compact real results. Do not commit `artifacts/`,
raw data, `.testdeps/`, authentication files, or large checkpoints. Keep large raw
outputs on the Volume or a separately accessible artifact location. Include the
repository URL in your own report as required by the assignment.

This implementation uses public libraries and follows these references:

- [Qwen3 model and chat template](https://huggingface.co/Qwen/Qwen3-0.6B)
- [SigLIP implementation](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/siglip)
- [lm-evaluation-harness v0.4.9.2](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.2), including stock benchmark tasks and filters; the local wrapper adds durable request caches.
- [OPLoRA, Xiong and Xie](https://arxiv.org/abs/2510.13003), including projected factors and subspace diagnostics.
- [Official VQA evaluation](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py): the official contraction mapping is downloaded, parsed as literal data, and attributed; normalization/scoring follow the official rules.
- [COCO](https://cocodataset.org/#download), [VQAv2](https://visualqa.org/download.html), [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), and the supplied ATML-PA3 rollout export.
- [Math-Verify](https://github.com/huggingface/Math-Verify) and [MathVista metadata](https://github.com/lupantech/MathVista/blob/main/data/testmini.json).
- [Fixed qualitative prompts](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml); no external AI-judge settings are used.
- [Modal checkpoint/Volume guidance](https://modal.com/docs/examples/long-training).

Understand every submitted line and describe any deviations in your own report.

## Recovery for the November 5 statistics failure

To view committed progress during a quiet run, open a second terminal here:

```powershell
py scripts/watch_progress.py --watch
```

This reads the Modal Volume directly and refreshes every 60 seconds. It launches
no containers, counts saved Task 3 responses out of 250, and shows Task 2 training
updates. Ctrl+C stops only the viewer. Omit `--watch` for a single snapshot.
Saved counts can lag ongoing generation and are not a remaining-time estimate.

If the log says `task2-eval` completed followed by `Run task2-eval first`, the
warm CPU worker was reading an old Volume snapshot. The launcher now reloads the
Volume before each child job and at orchestrator entry. It also propagates remote
errors rather than incorrectly claiming the cloud job continues.

From this updated checkout, first verify recovery using CPU only:

```powershell
py -m modal run --detach modal_app.py --command task2-statistics --run-id main --tokens 196
```

After it succeeds, and with no other pipeline running for `main`, resume:

```powershell
py -m modal run --detach modal_app.py --command all --run-id main
```

This repair changes no experiment modules or cache signatures. Existing verified
GPU outputs remain eligible for reuse. Do not delete checkpoints or change the
run ID. The 49-token variant still requires its full 4,500-update curriculum;
the quoted 45–50 minute / $1.10 remainder is not a verified runtime or cost bound.
