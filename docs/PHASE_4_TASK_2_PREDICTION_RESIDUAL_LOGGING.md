# Phase 4 --- Task 2: Prediction Residual Logging

## Status

**Residual logging validation: PASSED**\
**Diagnostic-field correction: APPLIED AND VERIFIED**

This task adds prediction-residual logging to the selected ByteTrack
Test E run. It records the discrepancy between the fused detection
position and the track's predicted position for each logged
prediction/match event. The output provides groundwork for later anomaly
analysis.

## Configuration

The selected Test E tracking parameters remain unchanged:

  Parameter           Value
  ---------------- --------
  `track_thresh`     `0.45`
  `track_buffer`       `50`
  `match_thresh`     `0.90`
  `frame_rate`         `30`

## Residual definition

For each recorded prediction/match event:

\[ r_t = `\sqrt{(x_t - \hat{x}_t)^2 + (y_t - \hat{y}_t)^2}`{=tex} \]

where (x_t, y_t) are the matched fused detection coordinates and
(`\hat{x}`{=tex}\_t, `\hat{y}`{=tex}\_t) are the track's predicted
coordinates captured before the update. Residuals are measured in
pixels.

A larger residual indicates a greater discrepancy between prediction and
observation; it is not automatically an anomaly. Anomaly thresholds and
interpretation belong to a later analysis step.

## Validation results

  -----------------------------------------------------------------------
  Check                                                            Result
  ------------------------------ ----------------------------------------
  Tracked output rows                                               3,623

  Unique track IDs                                                     64

  Prediction/match events logged                                    3,559

  Finite trajectory residuals                                       3,559

  Nonzero trajectory residuals                                      3,559

  NaN trajectory residuals                                             64

  Residual formula max absolute                        Approximately (4.0
  difference                             `\times 10`{=tex}\^{-13}) pixels

  Input/fusion row IDs for                     Match for all 3,559 events
  logged events                  
  -----------------------------------------------------------------------

The 64 NaN trajectory values correspond to rows without a recorded
prediction/match event. They must not be interpreted as zero residuals.

The residual values were checked against the Euclidean-distance formula
and matched between the event log and trajectory output by frame and
track ID.

## Diagnostic-field correction

The original event logger incorrectly derived `detection_center_x` and
`detection_center_y` from the installed ByteTrack object's `.tlwh`
property. For fresh detection objects in this installed version, that
property is not a reliable way to recover the input detection center.

The diagnostic fields were corrected using the exact input detection row
identified by `input_detection_row_id`, with the corresponding
calibrated fusion row identified by `fused_row_id`:

-   `detection_center_x` = (`x1` + `x2`) / 2
-   `detection_center_y` = (`y1` + `y2`) / 2
-   `detection_to_fused_distance` = Euclidean distance between that
    input-box center and the corresponding calibrated fused center

Verification against the uploaded source CSVs confirmed: - All 3,559
logged input row IDs match their fusion row IDs. - The input and fusion
row frames match for every event. - Corrected detection-to-fusion
distance ranges from 0 to approximately 14.5901 pixels; the median is 0
pixels. - The residual values still match the residual formula to
floating-point precision.

The corrected diagnostic fields are now suitable for describing
input-box center versus fused-center alignment. The residual remains
specifically the distance from the **predicted track position** to the
**fused detection position**, not the diagnostic detection-to-fusion
distance.

## Output artifacts

The relevant outputs are stored under `outputs/`:

-   `tracked_detections_test_e_residuals.csv` --- tracked detections
    from the residual-logging run.
-   `prediction_residuals_test_e.csv` --- prediction/match event log.
    Use the corrected version of this file.
-   `trajectory_test_e_with_residuals.csv` --- trajectory output with
    `prediction_residual_pixels`.

A corrected event-log CSV was generated as
`prediction_residuals_test_e_corrected.csv` from the uploaded event log
and source inputs. Copy it into the project's `outputs/` folder and
rename it to `prediction_residuals_test_e.csv`, replacing the older
event log after keeping a backup if desired.

A corrected script is also available as
`run_bytetrack_with_residuals_corrected_v3.py`. It calculates
detection-center diagnostics from the exact matched input row rather
than `new_track.tlwh`. Copy it into the project's `src/tracking/` folder
if you want to retain this corrected implementation. Running that script
will regenerate the three Test E residual outputs; it uses the same Test
E parameters.

Keep these related inputs and comparison artifacts:

-   `fused_detections_calibrated.csv`
-   `bytetrack_detections.csv`
-   `tracked_detections_test_e.csv`
-   `bytetrack_comparison_test_e.mp4`
-   `bytetrack_tuning_summary.csv`

Preserve the original baseline `tracked_detections.csv`,
`trajectory.csv`, and `src/tracking/generate_trajectory.py` unchanged.

## Scope and limitations

This task logs prediction residuals; it does not implement anomaly
detection, establish thresholds, or prove that a large residual is an
anomaly. No change to the selected Test E tracking parameters was made.

## Completion checklist

-   [x] Run the residual-logging pipeline with the selected Test E
    parameters.
-   [x] Confirm finite, nonzero residual values are present.
-   [x] Verify residuals against the formula and match them to
    trajectory rows by frame and track ID.
-   [x] Correct and verify detection-center and
    detection-to-fusion-distance diagnostics.
-   [x] Preserve baseline outputs and selected Test E settings.
-   [ ] Continue to the next roadmap task according to the project
    roadmap.
