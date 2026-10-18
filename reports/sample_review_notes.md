# Video sample review notes

Scope: 30 selected episodes, one left external camera per episode. The full converted MP4 interval was decoded automatically for each selected episode; a first-frame image was retained locally for visual inspection. The sample is stratified by outcome and review flag and is not a representative defect-rate estimate.

The following four first-frame images were inspected on 2026-10-18. Outcomes below come from the source label, not from an inference made using a still image.

| Hashed episode | Source outcome | Motion review flag | Observed first-frame content |
|---|---|---|---|
| 38d74a6d4cb6390ea204 | Failure | No | Kitchen counter/cabinet scene with visible robot arm and gripper; image is readable. |
| 0facd43e383a6fd36627 | Failure | Yes | Table scene with visible robot arm, packaging and small objects; image is readable. |
| 74ffe56274829caa737a | Success | No | Counter/table scene with towel and kitchen objects; the robot is outside this first-frame view. |
| 269a6ee9bea4285d2b41 | Success | Yes | Sink/counter scene with household objects; the robot is outside this first-frame view. |

All four converted episode intervals matched the expected frame count during the automatic decode check. Across all 30 sampled episodes, there were zero decode errors and zero frame-count mismatches. These findings do not establish that task labels are correct or that the underlying raw sensor streams are synchronized. A robot being outside a first-frame view is not, by itself, a data defect. The motion anomaly flags remain review suggestions; still images cannot confirm those flags as corruption.

Real source images remain under the ignored local `artifacts/video_frames` directory. No AI-generated scene is used as dataset evidence. Attribution: DROID authors and the jnogga LeRobot conversion, CC BY 4.0, linked in the README.
