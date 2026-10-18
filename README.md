# Robot Data Quality & Success Prediction

A reproducible portfolio project for AI data and algorithm engineering: audit real robot demonstrations, predict eventual demonstration success from early motion, and test cleaning in a separate simulated policy experiment.

**Status:** real-data audit and prediction complete: **15,000 demonstrations**, **30 sampled videos**, and 13 research-control tests passed in an independent D-drive environment. The five policy datasets have been prepared and checked; the formal policy comparison is running, and no policy improvement is claimed.

| Model | Held-out ROC-AUC | Failure PR-AUC | Failure recall |
|---|---:|---:|---:|
| constant_prior | 0.500 | 0.245 | 1.000 |
| logistic_regression | 0.648 | 0.376 | 0.533 |
| lightgbm | 0.592 | 0.333 | 0.548 |

The validation-selected model is **lightgbm**. Its validation ranking did not carry over to held-out collectors; the simpler logistic regression achieved the higher test AUC. The original resume AUC of 0.85 is not supported by this experiment.

![Local evidence dashboard](reports/figures/demo_overview.jpg)

An independent environment regenerated 156 prefix features and all three model predictions for one real held-out success and one failure episode. See [`reports/real_sample_verification.json`](reports/real_sample_verification.json). Full methods and uncertainty are in the research report.

## Why this project

Robot learning depends on demonstration quality. This project makes that dependency inspectable with bounded downloads, source hashes, structural/time checks, anomaly review, collector-held-out evaluation, and a controlled policy comparison. The contribution is the research and engineering pipeline, not a new foundation model.

```mermaid
flowchart LR
  A[Pinned DROID / LeRobot data] --> B[Structure + time audit]
  B --> C[Motion anomaly review]
  A --> D[First 30 motion frames]
  D --> E[Collector-grouped models]
  C --> F[Evidence report + demo]
  E --> F
  G[robomimic Lift demonstrations] --> H[Controlled action corruption]
  H --> I[Five policy conditions]
  I --> F
```

## Data and attribution

- Real data: [DROID](https://droid-dataset.github.io/), using the community [LeRobot success conversion](https://huggingface.co/datasets/jnogga/droid_success) and [failure conversion](https://huggingface.co/datasets/jnogga/droid_failure), CC BY 4.0. Credit to the DROID authors and conversion maintainer. These converted datasets already exclude some inconsistent raw episodes.
- Simulation data: [robomimic datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets), MIT; environment [robosuite](https://github.com/ARISE-Initiative/robosuite), powered by MuJoCo. This project implements its own small PyTorch BC network using the released environment metadata.
- Data commits are pinned in `configs/project.yaml`; downloaded file SHA256 values are recorded. Raw data is not redistributed in this repository. Source licenses govern source data; the MIT license here covers project code.

## Run on Windows, with project writes on D:

Use an existing Python 3.12 interpreter. The bootstrap script creates a dedicated environment on D:. It does not install Python, CUDA, Docker, WSL, or a browser on C:.

```powershell
# Run from the project directory on D:
.\scripts\bootstrap.ps1 -BasePython 'PATH_TO_EXISTING_PYTHON.exe'
.\scripts\run.ps1 preflight
.\scripts\run.ps1 test
.\scripts\run.ps1 fetch
.\scripts\run.ps1 audit
.\scripts\run.ps1 detect
.\scripts\run.ps1 train
.\scripts\run.ps1 video --episodes 30
.\scripts\run.ps1 report
.\scripts\run.ps1 demo
```

`fetch --small` selects a clearly smaller cohort for a network smoke test; do not mix its results with the full experiment. Public downloads do not require a Hugging Face token. Downloading selected episodes can require entire shared Parquet shards (~7.2 GB controls across both sources); videos have a separate 8 GiB cap. Total project storage is capped at 30 GiB. Partial downloads and episode audit checkpoints can resume. Changing audit/feature code invalidates audit checkpoints.

All project-controlled caches, temporary files, environments and outputs are rooted under the project. The startup guard resolves paths and rejects escapes or Windows drives other than D:. Existing Windows/browser application logs are outside this project's write control.

For a clean Linux/CI/cloud workspace, install `requirements.txt`, set `PYTHONPATH=src`, and use `python -m robot_audit --root /content/robot-audit COMMAND` (or a `/home/...` workspace). See the main notebook.

## Research protocol

**Real data.** Target 12,000 successful and 3,000 failed demonstrations after UUID checks, excluding episodes shorter than 30 frames. Full/prefix motion duplicates are excluded conservatively. Split by collector (approximately 5/7 train, 1/7 validation, 1/7 test). All models use fixed-prefix motion features only. No source, text, IDs, total duration, reward or terminal state is a feature. Imputation/scaling fit on training data. Thresholds and presentation model are chosen on validation; test results remain held out. ROC-AUC, failure PR-AUC, failure recall, confusion matrices and collector-bootstrap intervals are reported.

**Quality.** Rules detect structure and converted time inconsistencies. Motion outliers and Isolation Forest scores identify review candidates. Task failure is not a corruption label. MP4 checks apply only to the reported sample and converted timestamps, not original sensor-clock drift.

**Policy.** The dedicated [Colab notebook](notebooks/01_colab_policy.ipynb) performs the simulated Lift experiment. It requires a free runtime and user-authorized Google Drive mounting for checkpoints. Use 180/20 released train/validation demonstration masks; inject finite action spikes/misalignment into 10% of training episodes only. Compare clean, contaminated, random-drop, audit-drop and oracle-drop at equal optimization budgets. A simulator smoke check verifies observations, actions and reproducible resets before the clean pilot. The pilot requires at least 70% on 20 independent starts before freezing one of 30k/60k/90k steps: the measured 30k pilot achieved 40%, and 60k achieved 70%, freezing **60,000 steps**. Three prespecified seeds, 100 matched evaluation starts per policy, 1,500 formal rollouts total. Initial-state hashes and training budgets must match. Report percentage-point differences and paired two-way cluster bootstrap intervals over training seeds and shared initial states. Failed/negative results are retained.

## Inspect the evidence

- `reports/research_report.md`: question, methods, measured results and limitations.
- `reports/prediction_metrics.json`: model selection, validation/test metrics and uncertainty.
- `reports/audit_summary.json`, `reports/video_summary.json`: actual coverage.
- `reports/policy_results.json`: created only after all formal rollouts pass acceptance.
- `reports/resume_evidence.md`: fact-bounded English bullets and Chinese interview explanation.

The local Streamlit app has four views: overview, quality review, trajectories, and success prediction. Missing outputs display a pending state rather than invented demo scores. No hosting subscription is required.

## Important boundaries

Labels describe human demonstrations, not autonomous robot reliability. Collector-held-out testing does not establish unseen-task or unseen-scene generalization. The selected 80/20 outcome prevalence affects precision/calibration. Simulated injected-corruption results cannot establish real DROID policy improvements. CI checks research controls with synthetic fixtures and does not rerun the multi-GB real-data experiment.

## References

Khazatsky et al. (2024), *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset*.

Mandlekar et al. (2021), *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*.
