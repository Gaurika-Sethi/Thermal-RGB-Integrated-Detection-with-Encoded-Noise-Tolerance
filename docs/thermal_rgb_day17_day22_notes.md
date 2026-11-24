# Thermal--RGB Integrated Detection --- Day 17 to Day 21 Notes

## Scope and pipeline context

These notes consolidate the work completed across Days 17--21: ByteTrack
baseline integration, parameter evaluation, residual logging, track
lifecycle and occlusion analysis, and the detection/fusion/tracking
audit.

The tracking pipeline used during this phase was:

1.  Run person detection independently on RGB and thermal frames.
2.  Transform thermal detections into RGB coordinate space using the
    thermal-to-RGB homography.
3.  Perform reliability-aware fusion in RGB coordinates, retaining
    matched detections and unmatched `rgb_only` / `thermal_only`
    detections.
4.  Feed the fused detections, in RGB coordinate space, to one ByteTrack
    instance.
5.  Export persistent track IDs, tracked detections, and trajectory
    data.

**Coordinate-space note:** The IR video was warped/resized to 1920×1080
for visualization. This visualization transform does not mean that the
underlying detection/tracking pipeline was run in thermal coordinates;
the tracking inputs were represented in RGB coordinate space.

------------------------------------------------------------------------

# Day 17 --- ByteTrack Baseline

## Objective

Establish a baseline multi-person tracking result for both LLVIP Light
and Dark sequences using the fused detections. The same baseline
ByteTrack configuration was used for both sequences so their behaviour
could be compared.

## Baseline configuration (B0)

  Parameter           Value
  ---------------- --------
  `track_thresh`       0.45
  `track_buffer`         25
  `match_thresh`       0.80
  `frame_rate`       25 FPS

## Results

  Metric                                 LLVIP Light   LLVIP Dark
  ------------------------------------ ------------- ------------
  Input fused detections                      19,985          882
  Tracked rows                                14,647          844
  Unique track IDs                               288            8
  Short tracks (\<5 observed frames)              58            2
  Tracks containing gaps                         186            7
  Total internal missing frames                2,458          119
  Mean observed track length                   50.86        105.5
  Tracked rows / input detections             0.7329       0.9569
  Frame range in tracking output           0--10,443   642--2,631

The tracked-rows/input-detections ratio is an output-retention
statistic. It is **not** a tracking-accuracy score.

## Initial interpretation

-   The Light sequence produced substantially more detections and track
    IDs. It also contained more fragmented tracks and more internal
    missing frames.
-   The Dark sequence produced fewer detections and only eight IDs. Its
    high tracked-row/input ratio does not, by itself, establish better
    tracking accuracy.
-   The baseline was retained as the reference configuration for
    subsequent tuning and validation.

## Outputs

-   `outputs/tracked_detections_llvip_light_B0.csv`
-   `outputs/metrics_llvip_light_B0.csv`
-   `outputs/tracked_detections_llvip_dark_B0.csv`
-   `outputs/metrics_llvip_dark_B0.csv`

------------------------------------------------------------------------

# Day 18 --- ByteTrack Parameter Evaluation

## Objective

Evaluate at least five ByteTrack configurations against B0 and determine
whether parameter changes provide a convincing improvement. The
evaluation focused on both numerical metrics and visual tracking
behaviour.

## Configurations tested

  Run     `track_thresh`   `track_buffer`   `match_thresh`
  ----- ---------------- ---------------- ----------------
  B0                0.45               25             0.80
  T1                0.35               25             0.80
  T2                0.55               25             0.80
  T3                0.45               50             0.80
  T4                0.45               25             0.70
  T5                0.45               25             0.90

The complete numerical comparison below is for the **Light** sequence.

## Light sequence results

  ----------------------------------------------------------------------------
  Metric             B0         T1         T2         T3         T4         T5
  ---------- ---------- ---------- ---------- ---------- ---------- ----------
  Tracked        14,647     16,992     10,536     14,739     13,930     15,005
  rows                                                              

  Unique IDs        288        426        187        279        386        258

  Short              58        131         16         55        129         34
  tracks                                                            
  (\<5                                                              
  frames)                                                           

  Tracks            186        278        139        183        210        188
  with gaps                                                         

  Total           2,458      3,932      2,059      2,932      2,249      2,964
  internal                                                          
  missing                                                           
  frames                                                            

  Mean            50.86      39.89      56.34      52.83      36.09      58.16
  observed                                                          
  track                                                             
  length                                                            

  Median           33.5         17         45         34         13       44.5
  observed                                                          
  track                                                             
  length                                                            
  ----------------------------------------------------------------------------

## Parameter interpretation

-   **T1 --- lower `track_thresh` (0.35):** Retained more detections,
    but also produced many more IDs, short tracks, and missing frames.
    This indicates increased fragmentation; the larger output is not
    automatically an improvement.
-   **T2 --- higher `track_thresh` (0.55):** Produced fewer IDs and
    fewer short tracks, with the lowest total missing-frame count among
    the tested runs. However, the visual improvement was not
    sufficiently convincing to replace B0.
-   **T3 --- larger `track_buffer` (50):** Did not provide a clear
    overall improvement. The total number of missing frames increased
    relative to B0.
-   **T4 --- `match_thresh` 0.70:** Produced substantially more IDs and
    short tracks, with a shorter median track length. It was not
    selected.
-   **T5 --- `match_thresh` 0.90:** Produced fewer IDs than B0 and a
    longer mean track length, but visual inspection showed missed
    detections. It was rejected.

### Important `match_thresh` interpretation

For this ByteTrack implementation, matching is based on an IoU-derived
cost (`cost = 1 - IoU`). Therefore, increasing `match_thresh` makes the
accepted cost range more permissive, corresponding to a lower minimum
IoU requirement. In this tested setup, 0.90 is more permissive than
0.80; it should not be described as a stricter matching threshold.

## Decision

**Retain B0 for the Light sequence.** Although some configurations
improved selected numerical metrics, none demonstrated a sufficiently
convincing overall visual improvement. B0 remains the selected baseline
because it provided the most acceptable balance in the visual checks.

The Dark sequence's B0 result remains the baseline. The Dark T1--T5
sweep was not comprehensively evaluated, so B0 should not be described
as proven optimal among all Dark configurations.

------------------------------------------------------------------------

# Day 19 --- Prediction Residual Logging

## Objective

Add prediction-residual logging to the ByteTrack output for the Light
and Dark B0 runs. The residual was intended as a groundwork signal for
later trajectory analysis and anomaly detection.

The intended comparison was between the predicted track centre
immediately after the Kalman prediction step and the centre of the
detection supplied to the tracker for the matched observation.

## Intended quantities

For a matched detection:

-   Predicted centre: Kalman state `mean[:2]` (XY centre).
-   Detection centre:
    -   `det_x = (x1 + x2) / 2`
    -   `det_y = (y1 + y2) / 2`
-   Pixel residual:

\[ r_t = `\sqrt{(det_x-\hat{x}_t)^2+(det_y-\hat{y}_t)^2}`{=tex} \]

-   Normalized residual:

\[ r\_{norm,t} = `\frac{r_t}{\hat{h}_t}`{=tex} \]

where (`\hat{h}`{=tex}\_t) is the predicted track height.

The tracker uses an XYAH Kalman state: centre-x, centre-y, aspect ratio,
height, and their corresponding velocities. For an active track,
`mean[:2]` therefore represents the predicted centre coordinates.

## Investigation and limitation

The residual output showed an anomaly during validation. The logged
residuals were unusually close to the half-diagonal of the corresponding
bounding boxes, rather than appearing consistent with the independently
measured frame-to-frame centre movement.

For the `k=0` event comparison:

  -----------------------------------------------------------------------
  Check                                   Light                      Dark
  ------------------- ------------------------- -------------------------
  Logged residual                     111.43 px                  57.43 px
  median                                        

  Bounding-box                        110.93 px                  57.50 px
  half-diagonal                                 
  median                                        

  Correlation between                     0.882                     0.991
  residual and                                  
  half-diagonal                                 

  Consecutive                           6.05 px                   0.67 px
  observed centre                               
  displacements:                                
  median                                        

  Consecutive                           7.29 px                   1.03 px
  observed centre                               
  displacements: mean                           

  Consecutive                          15.05 px                   2.37 px
  observed centre                               
  displacements: P90                            
  -----------------------------------------------------------------------

These comparisons raise a serious concern about the residual logger.
Consecutive observed centre displacement and Kalman prediction error are
not the same measurement, so the comparison alone does not prove the
precise source of the problem. However, the close correspondence between
logged residuals and box half-diagonals means the residual values cannot
currently be trusted as valid prediction errors.

The instrumentation and association lifecycle were investigated. The
installed package was `bytetracker==0.3.2`; its active-track Kalman
state uses XYAH, and `mean[:2]` is the centre. A separate diagnostic
also showed that the incoming, not-yet-activated `new_track._tlwh`
representation in the observed code path contained centre-x, centre-y,
width, and height. This means that replacing the centre-based input
matching with a top-left TLWH comparison was not a valid fix.

Changing the residual comparison from the fused centroid to the
tracker-input detection centre had negligible effect on the summary
statistics. The underlying logging anomaly was not conclusively
resolved.

## Residual values recorded by the logger --- provisional/unverified

  Metric                          Light B0    Dark B0
  ---------------------------- ----------- ----------
  Prediction/match events           14,359        836
  Pixel residual mean            109.11 px   61.96 px
  Pixel residual median          111.10 px   57.42 px
  Pixel residual P90             133.15 px   67.42 px
  Pixel residual P95             140.15 px   77.48 px
  Normalized residual mean          0.5839     0.5281
  Normalized residual median        0.5730     0.5250
  Normalized residual P90           0.6724     0.5479
  Normalized residual P95           0.7018     0.5543

**These are retained as recorded diagnostic outputs only. They are not
validated prediction-error measurements and must not be used to claim
that one modality/sequence has better prediction performance than the
other.** In particular, the lower Dark normalized residual must not be
interpreted as evidence of better Dark tracking.

## Day 19 decision

Close the current investigation by documenting residual logging as a
**known limitation**. Preserve the recorded values with the
qualification above. Do not use them as validated evidence for tracking
accuracy or anomaly-detection thresholds until the logging issue is
independently resolved.

## Outputs

The residual runs were saved under `outputs/day19_residuals/`,
including:

-   `tracked_detections_light_B0_residuals.csv`
-   `prediction_residuals_light_B0.csv`
-   `trajectory_light_B0_residuals.csv`
-   `residual_summary_light_B0.csv`
-   `tracked_detections_dark_B0_residuals.csv`
-   `prediction_residuals_dark_B0.csv`
-   `trajectory_dark_B0_residuals.csv`
-   `residual_summary_dark_B0.csv`

------------------------------------------------------------------------

# Day 20 --- Track Lifecycle and Occlusion Analysis

## Objective

Inspect track lifecycles and identify gaps that could represent
temporary occlusion or track fragmentation. Candidate gaps were
generated for later visual verification; a gap alone is not proof of
occlusion.

## Lifecycle results

  Metric                                                Light         Dark
  ----------------------------------------------- ----------- ------------
  Trajectory/tracked rows                              14,647          844
  Unique IDs                                              288            8
  Mean observed track length                            50.86        105.5
  Median observed track length                           33.5          ---
  Short tracks (\<5 observed frames)                       58            2
  Tracks with internal gaps                               186            7
  Total internal missing frames                         2,458          119
  Tracks beginning after sequence minimum frame             2            8
  Tracks ending before sequence maximum frame               0            8
  Boundary-start count                                      2            0
  Boundary-end count                                        0            0
  Observed frame range                              0--10,443   642--2,097

The Dark trajectory file contained 844 rows. The lifecycle report used
the observed trajectory range 642--2,097; the tracking output frame
range recorded in the baseline report was 642--2,631. These are
different reported ranges and should not be conflated.

## Occlusion candidate analysis

A maximum gap threshold of 15 frames was used to identify candidate
interruptions.

  -----------------------------------------------------------------------
  Item                                    Light                      Dark
  ------------------- ------------------------- -------------------------
  Candidate gaps                            789                        29

  IDs represented in                        184                       ---
  candidates                                    

  Maximum candidate                   15 frames                 15 frames
  gap                                           

  Verification status     Requires visual check     Requires visual check
  at generation                                 
  -----------------------------------------------------------------------

Candidate reports:

-   `outputs/day20/light/occlusion_candidates.csv`
-   `outputs/day20/dark/occlusion_candidates.csv`

Lifecycle reports:

-   `outputs/day20/light/track_lifecycle_report.csv`
-   `outputs/day20/dark/track_lifecycle_report.csv`

## Visual validation

### Light sequence

Visual inspection indicated that tracking was generally consistent. Most
people were tracked reasonably well, including through ordinary
movement. The more difficult cases were:

-   Extremely crowded regions.
-   Two people riding on a bike, where the rear person was occluded.

Outside these difficult cases, detections and tracks appeared
consistent.

### Dark sequence

Visual inspection confirmed that seven of the eight IDs corresponded to
one physical person who repeatedly disappeared behind trees. This
behaviour is best described as **track fragmentation caused by repeated
disappearance/occlusion**, rather than an identity-switch problem. One
other person retained one ID.

The observation is based on visual inspection; do not infer a more
precise ID-to-person mapping unless it has been explicitly verified.

## Interpretation

-   A track gap is only an occlusion candidate until the video is
    checked.
-   The Light sequence has many more candidate gaps and fragmented IDs,
    but the visual review found overall tracking to be reasonably
    consistent apart from difficult crowding/occlusion cases.
-   The Dark sequence's main observed issue was one person's track being
    split across multiple IDs after repeated disappearance behind trees.
-   These findings describe lifecycle continuity and fragmentation; they
    do not establish identity-switch rates.

------------------------------------------------------------------------

# Day 21 --- Detection, Fusion, Tracking and Output Audit

## Objective

Audit the inputs and outputs of the current pipeline for data integrity,
modality composition, bounding-box validity, track-output integrity, and
the status of residual logging.

## Detection confidence

  Statistic               Light     Dark
  -------------------- -------- --------
  Fused detections       19,985      882
  Confidence median      0.5741   0.7240
  Confidence mean        0.5819   0.7061
  Confidence minimum     0.2500   0.3284
  Confidence maximum     0.8989   0.8930

These are descriptive confidence statistics. They are not ground-truth
detection-accuracy measurements.

## Fusion composition

  Source category           Light      Dark
  ------------------ ------------ ---------
  RGB-only                 16,333         2
  Matched (`both`)          2,845       109
  Thermal-only                807       771
  **Total**            **19,985**   **882**

The Light sequence is dominated by RGB-only detections. The Dark
sequence is dominated by thermal-only detections, with very few RGB-only
detections. This is consistent with the different modality availability
in the two lighting conditions.

## Bounding-box audit

Median box dimensions:

  -----------------------------------------------------------------------
  Source category          Light median width ×       Dark median width ×
                                         height                    height
  ------------------- ------------------------- -------------------------
  RGB-only                   103.32 × 188.37 px         37.71 × 101.18 px

  Matched (`both`)            99.33 × 191.31 px         42.40 × 110.65 px

  Thermal-only                95.82 × 212.57 px         35.00 × 109.09 px
  -----------------------------------------------------------------------

-   No invalid box dimensions were found in the audited fused
    detections.
-   No missing fused coordinates were found.
-   The audit did not provide sufficient evidence to replace the
    existing bounding-box construction with a four-corner homography
    transformation. The current construction was therefore retained.

## Tracked-output integrity

  Check                                    Light   Dark
  ------------------------------------- -------- ------
  Tracked rows                            14,647    844
  Unique IDs                                 288      8
  Duplicate `(frame, track_id)` pairs          0      0
  Invalid box dimensions                       0      0

No duplicate frame/track pairs or invalid tracked-box dimensions were
found in the audited outputs.

## Residual audit status

The residual statistics listed in Day 19 were reviewed again on Day 21.
The close relationship between the logged residual and the bounding-box
half-diagonal, together with the much smaller independent
consecutive-centre displacement, made the residual values unreliable as
prediction-error measurements.

**Status: provisional/unverified.** The residual values are retained for
traceability but excluded from performance conclusions. The residual
issue does not invalidate the independently audited detection, fusion,
or tracked-output integrity checks.

## Day 21 decisions

1.  Retain the current fusion and bounding-box construction; the audit
    did not justify changing it.
2.  Retain the Light B0 ByteTrack configuration selected after Day 18
    visual evaluation.
3.  Keep Dark B0 as the current Dark baseline, while noting that the
    Dark parameter sweep was not comprehensively evaluated.
4.  Treat residual logging as a known limitation and do not use the
    current residual values as validated performance evidence.
5.  Preserve the audited detection, fusion, and tracking outputs; no
    rerun was indicated for those checks.

------------------------------------------------------------------------

# Consolidated Decisions and Current State (Days 17--21)

  -----------------------------------------------------------------------
  Area                                Decision / status
  ----------------------------------- -----------------------------------
  Tracking architecture               One ByteTrack instance consumes
                                      fused detections in RGB coordinate
                                      space

  Light ByteTrack configuration       B0 retained: 0.45 / 25 / 0.80 / 25
                                      FPS

  Dark ByteTrack configuration        B0 retained as baseline; not proven
                                      optimal across a complete Dark
                                      sweep

  Parameter tuning                    Light T1--T5 compared; no tested
                                      configuration justified replacing
                                      B0

  Light visual tracking               Generally consistent; difficult
                                      cases include extreme crowds and
                                      rear passenger occlusion on a bike

  Dark visual tracking                Repeated disappearance behind trees
                                      fragmented one person's track
                                      across seven IDs; not identified as
                                      an identity-switch issue

  Lifecycle/occlusion analysis        Candidate gaps generated; only
                                      visually checked cases should be
                                      called confirmed occlusions

  Detection/fusion integrity          Audited; no missing fused
                                      coordinates or invalid fused box
                                      dimensions found

  Tracked-output integrity            Audited; no duplicate frame/track
                                      pairs or invalid tracked-box
                                      dimensions found

  Bounding-box geometry               Existing construction retained; no
                                      evidence sufficient to justify
                                      four-corner homography

  Prediction residuals                Known limitation; values recorded
                                      but unverified and excluded from
                                      performance claims
  -----------------------------------------------------------------------

## Important interpretation rules

-   More tracked rows do not necessarily mean better tracking.
-   Fewer IDs or fewer short tracks do not independently prove better
    identity continuity.
-   `tracked rows / input detections` is an output-retention ratio, not
    accuracy.
-   Occlusion-candidate CSV entries are not confirmed occlusions until
    visually checked.
-   Track fragmentation and identity switches are different failure
    modes.
-   Confidence values are not ground-truth accuracy.
-   Current residual statistics are not validated prediction errors and
    must not be used to compare Light and Dark tracking performance.

## Files and outputs referenced

### Day 17 / Day 18

-   `outputs/tracked_detections_llvip_light_B0.csv`
-   `outputs/metrics_llvip_light_B0.csv`
-   `outputs/tracked_detections_llvip_dark_B0.csv`
-   `outputs/metrics_llvip_dark_B0.csv`

### Day 19

-   `outputs/day19_residuals/`

### Day 20

-   `outputs/day20/light/track_lifecycle_report.csv`
-   `outputs/day20/light/occlusion_candidates.csv`
-   `outputs/day20/dark/track_lifecycle_report.csv`
-   `outputs/day20/dark/occlusion_candidates.csv`
-   `outputs/day20/dark/annotated_dark.avi`

### Day 21

-   Detection, fusion, bounding-box, tracked-output, and residual audit
    findings documented above.

### Day 22

-   Weighted bounding-box dimensions were adopted for matched RGB–thermal   detections after winning the visual comparison in all 20 selected low-RGB-confidence frames. The weighted-box ByteTrack rerun produced identical track IDs, frame assignments, and lifecycle metrics to the original Dark B0 baseline, while updating bounding-box geometry. Thus, the modification improved box representation without measurable change in tracking continuity.
  
------------------------------------------------------------------------

**Overall conclusion:** Days 17--21 established and evaluated the
ByteTrack baseline, selected the Light B0 configuration after numerical
and visual comparison, inspected lifecycle continuity and
occlusion-related fragmentation, and audited
detection/fusion/tracked-output integrity. The principal unresolved
limitation is the prediction-residual logger: its recorded values are
preserved for traceability but are not treated as validated prediction
errors.
