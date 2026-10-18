# Local verification record — 2026-10-18

- Existing project environment: 13 research-control tests passed, including safe pilot resumption and rejecting formal policies trained beyond the frozen budget.
- Independent environment under `.cache/verification-env` on D: 13 tests passed; Python and notebook artifact checks passed.
- Independent real-data sample check: regenerated 156 features for two held-out demonstrations (one per outcome) and reproduced predictions from all three saved models, with feature tolerance 1e-12 and prediction tolerance 1e-10.
- Streamlit AppTest: zero exceptions; rendered 15,000 audited episodes, 1,495 review candidates, selected-model AUC 0.592 and 30 checked video episodes.
- Video sample: 30 decoded, zero frame-count mismatches, zero decode errors; left external camera only.
- Policy preparation on D: the pinned robomimic file passed its upstream SHA256 check; the released masks contain 180 training and 20 validation demonstrations. Exactly 18 training demonstrations are corrupted; each drop condition retains 162. This verifies dataset preparation, not policy performance. See `policy_preparation.json`.
- Notebook rebuilds produce identical SHA256 values with stable cell IDs, fixed ZIP timestamps and normalized source line endings. Source and notebook artifact checks passed after this change.
- Colab runtime 2026.07: Python 3.12.13, PyTorch 2.11.0+cpu, robosuite 1.5.1 and MuJoCo 3.3.7 imported successfully after a NumPy ABI restart. A free GPU backend could not be allocated. The user completed Drive authorization; the notebook printed `Mounted at /content/drive` and `CLOUD STORAGE PREFLIGHT PASS` before starting the policy pilot. Policy evaluation is still pending.
- Cloud simulator checks passed for 19-dimensional observations, actions and repeated resets with differing action histories. The clean pilot achieved 8/20 at 30k steps and 14/20 at 60k steps; the formal protocol is frozen at 60k. Raw pilot evaluations and checkpoints remain in the Drive project directory.
- The formal cloud policy comparison is running; GitHub Actions is pending. These checks do not establish that the full policy experiment or remote CI has passed.

Commands: `scripts/run.ps1 test`, fresh-environment `python -m pytest --basetemp .cache/verification-tests`, `python scripts/validate_artifacts.py`, and `python scripts/verify_real_sample.py`.
