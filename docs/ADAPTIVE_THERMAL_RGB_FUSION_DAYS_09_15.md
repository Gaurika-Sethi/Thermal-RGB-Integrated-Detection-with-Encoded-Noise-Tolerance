# Adaptive Thermal--RGB Fusion Implementation

## Phase 4 --- Days 9--15

**Project:** Thermal-RGB Integrated Detection with Encoded Noise
Tolerance\
**Implementation period:** Phase 4, Days 9--15\
**Status:** Implementation and final audit checks completed, with
documented experimental deviations\
**Current adaptive fusion implementation:** `src/fusion/fusion.py`\
**Selected parameters:** (`\alpha=1.0`{=tex}), (`\beta=0.75`{=tex}),
(W=30)

------------------------------------------------------------------------

## 1. Objective

This phase adds reliability-aware adaptive fusion to the existing
RGB--thermal detection pipeline.

The central idea is to estimate how reliably each modality is detecting
people over a rolling time window, then use those reliability estimates
together with detector confidence when combining matched RGB and
infrared detections.

The intended behaviour is:

-   In well-lit scenes, RGB should generally contribute more strongly.
-   In dark scenes, infrared should generally contribute more strongly.
-   Detections available from only one modality should be filtered using
    a reliability- and confidence-based threshold.
-   The resulting fused CSV must remain compatible with the existing
    ByteTrack preparation workflow.

The original baseline fusion implementation was preserved. Adaptive
fusion was implemented separately and the current adaptive
implementation was subsequently renamed to:

``` text
src/fusion/fusion.py
```

The ByteTrack preparation script was not modified for this phase.

------------------------------------------------------------------------

## 2. Implementation overview

The Days 9--15 workflow consists of the following stages:

1.  Estimate rolling RGB and IR reliability for each frame.
2.  Search a predefined grid of (`\alpha`{=tex}) and (`\beta`{=tex})
    values.
3.  Validate and select the parameters.
4.  Transform thermal bounding boxes using the saved scene calibration
    homography.
5.  Perform reliability-aware adaptive fusion.
6.  Apply modality- and scene-specific gates to unmatched detections.
7.  Validate the fused output and its compatibility with ByteTrack.
8.  Evaluate the behaviour on Light and Dark recordings.
9.  Run the final audit.

### Main files

  -------------------------------------------------------------------------------
  File                                        Purpose
  ------------------------------------------- -----------------------------------
  `src/fusion/reliability.py`                 Rolling modality reliability
                                              estimator

  `src/fusion/alpha_beta_search.py`           Grid search for (`\alpha`{=tex})
                                              and (`\beta`{=tex})

  `src/fusion/fusion.py`                      Current adaptive fusion
                                              implementation

  `src/fusion/warp_thermal_boxes.py`          Transform thermal bounding boxes
                                              using homography

  `src/fusion/verify_fusion.py`               Fusion output and bounding-box
                                              audit

  `src/tracking/prepare_bytetrack_input.py`   Existing conversion to ByteTrack
                                              input; unchanged

  `scripts/audit_day15.py`                    Final CSV and parameter audit
  -------------------------------------------------------------------------------

### Important output files

  ---------------------------------------------------------------------------------------------
  Output                                                    Description
  --------------------------------------------------------- -----------------------------------
  `outputs/calibration/selected_alpha_beta.json`            Selected reliability parameters and
                                                            selection metrics

  `outputs/reliability_scores_light_selected.csv`           Selected-parameter Light
                                                            reliability time series

  `outputs/reliability_scores_dark_selected.csv`            Selected-parameter Dark reliability
                                                            time series

  `outputs/IR_light_detection_warped.csv`                   Light thermal detections after box
                                                            transformation

  `outputs/IR_dark_detection_warped.csv`                    Dark thermal detections after box
                                                            transformation

  `outputs/fused_detections_adaptive_light.csv`             Adaptive Light fused detections

  `outputs/fused_detections_adaptive_dark.csv`              Adaptive Dark fused detections

  `outputs/bytetrack_detections_adaptive_light_audit.csv`   Light ByteTrack compatibility audit
                                                            output

  `outputs/bytetrack_detections_adaptive_dark_audit.csv`    Dark ByteTrack compatibility audit
                                                            output
  ---------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Day 9 --- Rolling reliability estimator

**Implementation:** `src/fusion/reliability.py`

For each modality and frame (t), reliability is calculated using a
rolling window of (W=30) frames.

### 3.1 Detection rate

For a modality:

\[
d(t)=`\frac{\text{number of frames with at least one valid detection in the window}}{W}`{=tex}
\]

A frame contributes at most one count to the numerator, even if it
contains multiple detections.

The window is:

\[ \[`\max`{=tex}(0,t-W+1),t\] \]

At the beginning of a sequence, fewer than (W) frames may be available.
The denominator remains (W), so the initial reliability estimates are
lower while the window is filling.

### 3.2 Mean confidence

Mean confidence is calculated over valid detections in the rolling
window:

\[ c(t)=`\text{mean confidence of valid detections in the window}`{=tex}
\]

If there are no valid detections, mean confidence is set to zero.

### 3.3 Reliability

The estimator uses:

\[ R(t)=d(t)^`\alpha `{=tex}`\times `{=tex}c(t)^`\beta`{=tex} \]

The initial values were:

\[ `\alpha=1.5`{=tex},`\quad `{=tex}`\beta=1.0`{=tex},`\quad `{=tex}W=30
\]

The estimator validates the input detection records, including frame
indices and finite coordinate/confidence values. Its output contains:

``` text
frame,
rgb_detection_rate,
ir_detection_rate,
rgb_mean_conf,
ir_mean_conf,
rgb_reliability,
ir_reliability
```

The output includes frames from zero through the maximum frame index
present in the input data. It cannot infer trailing frames beyond the
maximum frame represented in the detection CSVs.

### 3.4 Initial Day 9 results

  Metric                      Light       Dark
  ---------------------- ---------- ----------
  RGB input rows             31,621      4,167
  RGB valid detections       29,270        135
  IR input rows              11,575      4,611
  IR valid detections         4,163      3,714
  Frames represented         10,450      4,166
  Mean RGB reliability     0.384799   0.008717
  Mean IR reliability      0.128696   0.333116

The output schema, finite values, consecutive frame sequence and
reliability range checks passed.

------------------------------------------------------------------------

## 4. Day 10 --- Alpha/beta grid search

**Implementation:** `src/fusion/alpha_beta_search.py`

A grid search evaluated:

-   (`\alpha `{=tex}`\in `{=tex}{1.0,1.5,2.0})
-   (`\beta `{=tex}`\in `{=tex}{0.75,1.0,1.25})
-   (W=30)

The separation metrics compare the mean reliability of the expected
dominant modality in each scene:

\[
sep\_{light}=`\overline{R}`{=tex}*{RGB,light}-`\overline{R}`{=tex}*{IR,light}
\]

\[
sep\_{dark}=`\overline{R}`{=tex}*{IR,dark}-`\overline{R}`{=tex}*{RGB,dark}
\]

The selection score was:

\[ Score=sep\_{light}+2`\times `{=tex}sep\_{dark} \]

A penalty of (-1) was applied if either separation was negative. For
near-ties within 0.05 of the highest score, the selection rule preferred
the lower (`\alpha`{=tex}).

### 4.1 Search results

    Alpha   Beta   Light separation   Dark separation    Total score
  ------- ------ ------------------ ----------------- --------------
      1.0   0.75           0.288819          0.403478   **1.095775**
      1.0   1.00           0.244006          0.332783       0.909572
      1.0   1.25           0.206537          0.276584       0.759706
      1.5   0.75           0.303382          0.392682       1.088745
      1.5   1.00           0.256103          0.324399       0.904901
      1.5   1.25           0.216604          0.270041       0.756685
      2.0   0.75           0.312149          0.384004       1.080156
      2.0   1.00           0.263447          0.317642       0.898732
      2.0   1.25           0.222765          0.264755       0.752274

### 4.2 Selected parameters

The selected values were:

\[ `\boxed{\alpha=1.0,\quad\beta=0.75,\quad W=30}`{=tex} \]

They are stored in:

``` text
outputs/calibration/selected_alpha_beta.json
```

The stored selection metrics include:

-   `sep_light`: 0.2888190513
-   `sep_dark`: 0.4034781084
-   `base_score`: 1.0957752682
-   `penalty`: 0.0
-   `total_score`: 1.0957752682

These parameters were subsequently used for the reliability calculations
and adaptive fusion in this experiment.

------------------------------------------------------------------------

## 5. Day 11 --- Parameter validation

The selected parameters were used to regenerate the Light and Dark
reliability time series.

### 5.1 Selected-parameter results

  Metric                      Light       Dark
  ---------------------- ---------- ----------
  RGB input rows             31,621      4,167
  RGB valid detections       29,270        135
  IR input rows              11,575      4,611
  IR valid detections         4,163      3,714
  Frames represented         10,450      4,166
  Mean RGB reliability     0.466817   0.015244
  Mean IR reliability      0.177998   0.418722

Validation confirmed:

-   Selected (`\alpha`{=tex}), (`\beta`{=tex}) and (W) matched the
    parameter file.
-   Light RGB mean reliability exceeded Light IR mean reliability.
-   Dark IR mean reliability exceeded Dark RGB mean reliability.
-   The reliability values remained finite and within (\[0,1\]).

The selected parameters were locked for the current experiment. This
does not establish that they are globally optimal for every scene or
dataset; it records the selection made using the specified two-scene
search procedure.

------------------------------------------------------------------------

## 6. Day 12 --- Reliability-aware adaptive fusion

### 6.1 Thermal bounding-box transformation

**Implementation:** `src/fusion/warp_thermal_boxes.py`

Thermal detections were transformed using the saved thermal-to-RGB
homography for each scene:

``` text
outputs/calibration/day6_dark/thermal_to_rgb_homography.npy
outputs/calibration/day6_light/thermal_to_rgb_homography.npy
```

The transformation maps the four corners of each thermal bounding box.
The transformed axis-aligned bounding rectangle is then calculated.

The transformed box dimensions are updated while the detection centre
coordinates remain unchanged in the output representation used by the
fusion workflow.

Outputs:

``` text
outputs/IR_dark_detection_warped.csv
outputs/IR_light_detection_warped.csv
```

Ten overlay samples were saved per scene under:

``` text
outputs/day12_box_overlays/dark/
outputs/day12_box_overlays/light/
```

### 6.2 Box transformation results

  Metric                        Dark    Light
  ------------------------- -------- --------
  Input thermal rows           4,611   11,575
  Valid boxes transformed      3,714    4,163
  Median width ratio          1.0141   1.0611
  Mean width ratio            1.0138   1.0687
  Median height ratio         1.4671   1.4228
  Mean height ratio           1.4666   1.4271

The overlays revealed that some detector boxes were imperfect: some
people were missed in Light samples, and an oversized thermal box around
a non-person was observed in Dark. Homography transforms existing boxes;
it does not correct detector classification or localization errors.

### 6.3 Matching

The adaptive fusion implementation uses greedy nearest-centroid matching
between RGB and transformed thermal detections within each frame.

The configured matching threshold is:

``` python
MATCH_THRESHOLD = 50.0
```

A candidate pair is rejected when its centroid distance is greater than
or equal to the threshold. Matching is one-to-one within a frame.

### 6.4 Matched detection weights

For a matched RGB and IR pair:

\[ w\_{RGB}=R\_{RGB}`\times `{=tex}C\_{RGB} \]

\[ w\_{IR}=R\_{IR}`\times `{=tex}C\_{IR} \]

where (R) is modality reliability and (C) is detector confidence.

The fused centroid is:

\[ x_f=`\frac{w_{RGB}x_{RGB}+w_{IR}x_{IR}}{w_{RGB}+w_{IR}}`{=tex} \]

\[ y_f=`\frac{w_{RGB}y_{RGB}+w_{IR}y_{IR}}{w_{RGB}+w_{IR}}`{=tex} \]

The implemented matched confidence calculation is:

\[ C_f=`\frac{w_{RGB}+w_{IR}}{R_{RGB}+R_{IR}}`{=tex} \]

If the total matched weight is zero, the implementation uses the
arithmetic mean fallback for the centroid and confidence.

For matched detections, the selected bounding-box dimensions come from
the RGB source box.

### 6.5 Unmatched detections

The unmatched detection score is:

\[
S\_{single}=R\_{modality}`\times `{=tex}C\_{detector}`\times `{=tex}UNMATCHED_PENALTY
\]

The configured penalty is:

``` python
UNMATCHED_PENALTY = 0.7
```

An unmatched detection is retained only when its score is strictly
greater than its scene- and modality-specific threshold.

The current thresholds are:

``` python
MIN_THRESHOLDS = {
    "dark": {
        "rgb_only": 0.15,
        "ir_only": 0.20,
    },
    "light": {
        "rgb_only": 0.15,
        "ir_only": 0.08,
    },
}
```

These are experimental scene-specific gates, not a single global
`MIN_THRESHOLD`.

For retained unmatched detections, the fused confidence is the
detector's own confidence. RGB-only detections use RGB box dimensions;
IR-only detections use the transformed thermal box dimensions.

### 6.6 Output schema

The first nine columns preserve the fused detection compatibility
format:

``` text
frame,
fused_x,
fused_y,
fused_confidence,
source,
rgb_x,
rgb_y,
thermal_x,
thermal_y
```

Additional diagnostic columns are appended:

``` text
fused_x1,
fused_y1,
fused_x2,
fused_y2,
rgb_conf,
ir_conf,
rgb_reliability,
ir_reliability,
fusion_type,
single_weight,
match_distance
```

Source labels are:

-   `both`
-   `rgb_only`
-   `thermal_only`

Fusion type labels are:

-   `matched`
-   `rgb_only`
-   `ir_only`

The adaptive output is clipped to the 1920 × 1080 frame bounds.

### 6.7 Day 12 fusion results

  Fusion type                       Light      Dark
  -------------------------- ------------ ---------
  Matched (`both`)                  2,845       109
  RGB-only                         16,333         2
  IR-only (`thermal_only`)            807       771
  **Total output rows**        **19,985**   **882**

Unmatched candidate filtering:

  -----------------------------------------------------------------------
  Candidate            Light  Light dropped           Dark   Dark dropped
  type            candidates                    candidates 
  ----------- -------------- -------------- -------------- --------------
  RGB-only            26,425         10,092             26    24 (92.31%)
                                   (38.19%)                

  IR-only              1,318   511 (38.77%)          3,605 2,834 (78.61%)
  -----------------------------------------------------------------------

The output CSVs passed the reported confidence-range and
compatibility-schema checks.

### 6.8 Threshold changes and their effect

The scene-specific unmatched gates were adjusted during the experiment:

-   Dark IR-only threshold: raised to 0.20.
-   Light IR-only threshold: lowered to 0.08.
-   RGB-only threshold: 0.15 in both scenes.

These changes affected unmatched detection retention. They did not
change the selected (`\alpha`{=tex}), (`\beta`{=tex}), (W), matched
centroid weighting formula, matching threshold or unmatched penalty.

The thresholds are experimental settings. The observed candidate-drop
counts describe their effect on these recordings; they do not prove that
every dropped detection was a false positive.

------------------------------------------------------------------------

## 7. ByteTrack compatibility

The existing script was retained:

``` text
src/tracking/prepare_bytetrack_input.py
```

It was run using the adaptive fused outputs, the corresponding RGB
detections and the warped thermal detections.

### 7.1 Audit commands

Dark:

``` powershell
python src/tracking/prepare_bytetrack_input.py `
    --fused outputs/fused_detections_adaptive_dark.csv `
    --rgb outputs/rgb_dark_detection.csv `
    --thermal outputs/IR_dark_detection_warped.csv `
    --output outputs/bytetrack_detections_adaptive_dark_audit.csv
```

Light:

``` powershell
python src/tracking/prepare_bytetrack_input.py `
    --fused outputs/fused_detections_adaptive_light.csv `
    --rgb outputs/rgb_light_detection.csv `
    --thermal outputs/IR_light_detection_warped.csv `
    --output outputs/bytetrack_detections_adaptive_light_audit.csv
```

### 7.2 Required ByteTrack columns

The generated output contains the expected eight columns:

``` text
frame,x1,y1,x2,y2,confidence,class_id,source
```

### 7.3 Uploaded audit CSV results

Both uploaded audit files were inspected.

  Validation                         Dark      Light
  --------------------------- ----------- ----------
  Rows                                882     19,985
  Unique frames                       838      7,076
  Frame range                   642--2631   0--10443
  Expected eight columns             PASS       PASS
  Missing values                        0          0
  Invalid boxes                         0          0
  Coordinates within frame           PASS       PASS
  Confidence within \[0,1\]          PASS       PASS
  Class ID                              0          0

Source distribution:

  Source             Dark    Light
  ---------------- ------ --------
  `both`              109    2,845
  `rgb_only`            2   16,333
  `thermal_only`      771      807

The uploaded ByteTrack audit CSVs passed structural and data-integrity
checks. The existing preparation script was not modified.

**Known minor discrepancy from the earlier box audit:** Light fused-box
`y2` differed by at most 1 pixel in the earlier comparison because
adaptive fusion clips at 1079 while the preparation script permits 1080.
This was reviewed and explicitly accepted; it was not changed.

------------------------------------------------------------------------

## 8. Day 13 --- Additional video validation

Additional paired videos from LLVIP data were tested.

The reported qualitative behaviour was:

-   RGB contribution was dominant in light conditions.
-   IR contribution was dominant in dark conditions.

The user reported satisfactory behaviour on these additional videos.

Exact filenames, per-video row counts and numerical modality statistics
were not supplied for this validation. Therefore, this section records
qualitative validation only and does not claim numerical results for
those additional LLVIP clips.

------------------------------------------------------------------------

## 9. Day 14 --- Scene behaviour and matched-weight analysis

The roadmap referred to Light, Dark-13 and Dark-15 recordings. The exact
Dark-13 and Dark-15 recordings were not available for this experiment,
so the previously used Light and Dark recordings were used instead.

This is a documented substitution, not a claim that the exact requested
clips were tested.

### 9.1 Fusion distribution

  Fusion type      Light   Dark
  ------------- -------- ------
  Matched          2,845    109
  RGB-only        16,333      2
  IR-only            807    771

The distribution shows RGB-heavy output in Light and IR-heavy output in
Dark.

### 9.2 Reliability means in Dark

  Modality     Mean reliability
  ---------- ------------------
  RGB                  0.015244
  IR                   0.418722

The Dark reliability direction checks passed:

-   RGB mean reliability was below 0.05.
-   IR mean reliability was above 0.3.

### 9.3 Light matched-weight sample

Twenty distinct matched frames were sampled across the Light recording.

Mean contribution:

  Modality     Mean weight contribution
  ---------- --------------------------
  RGB                 0.628287 (62.83%)
  IR                  0.371713 (37.17%)

The Light RGB contribution exceeded 50%, so the Light dominance check
passed.

### 9.4 Dark matched-weight sample

Twenty distinct matched frames were sampled across the available 109
matched Dark frames.

Mean contribution:

  Modality     Mean weight contribution
  ---------- --------------------------
  RGB                 0.147794 (14.78%)
  IR                  0.852206 (85.22%)

IR contributed substantially more than RGB. Its mean contribution was
approximately 5.76 times the RGB contribution.

However, the strict numerical target of less than 10% mean RGB
contribution was **not met**. The measured RGB contribution was 14.78%.

This was retained as a documented deviation. No fusion formula or
threshold was changed to force the sample below 10%.

### 9.5 Day 14 interpretation and limitation

The available recordings support the intended qualitative direction:

-   RGB dominates the sampled matched weights in Light.
-   IR dominates the sampled matched weights in Dark.

The Dark numerical target was not fully achieved. The exact Dark-15
target cannot be claimed as verified because the Dark-15 recording was
unavailable. Similarly, the exact expected Dark-15 fusion counts were
not established.

------------------------------------------------------------------------

## 10. Day 15 --- Final audit

The final audit script was run:

``` text
scripts/audit_day15.py
```

### 10.1 Parameter checks

  Parameter     Expected Audit result
  ----------- ---------- --------------
  Alpha              1.0 PASS
  Beta              0.75 PASS
  Window W            30 PASS

The selected parameter JSON also contained the recorded selection reason
and search metrics.

### 10.2 Light CSV checks

The audit passed:

-   Fused CSV exists and is non-empty.
-   First nine compatibility columns are in the expected order.
-   Required fields are present.
-   Fused coordinates and confidence are finite.
-   Fused confidence lies in (\[0,1\]).
-   Fusion labels are valid.
-   Reliability fields are present, finite and within (\[0,1\]).

Reported Light mean reliability:

-   RGB: 0.4668172740
-   IR: 0.1779982227

Reported Light distribution:

-   RGB-only: 16,333
-   Matched: 2,845
-   IR-only: 807

### 10.3 Dark CSV checks

The audit passed:

-   Fused CSV exists and is non-empty.
-   First nine compatibility columns are in the expected order.
-   Required fields are present.
-   Fused coordinates and confidence are finite.
-   Fused confidence lies in (\[0,1\]).
-   Fusion labels are valid.
-   Reliability fields are present, finite and within (\[0,1\]).
-   Mean RGB reliability is below 0.05.
-   Mean IR reliability is above 0.3.

Reported Dark mean reliability:

-   RGB: 0.0152435465
-   IR: 0.4187216549

Reported Dark distribution:

-   IR-only: 771
-   Matched: 109
-   RGB-only: 2

The script reported:

``` text
DAY 15 CSV/PARAMETER AUDIT: PASS
```

### 10.4 ByteTrack final compatibility audit

The two uploaded audit CSVs were checked after the parameter/CSV audit.
Both passed row-count, schema, finite-value, coordinate-boundary,
box-validity, confidence and class-ID checks.

### 10.5 Current implementation constants

The current adaptive implementation is:

``` text
src/fusion/fusion.py
```

The user's source search output confirmed:

``` python
MATCH_THRESHOLD = 50.0
UNMATCHED_PENALTY = 0.7
```

It also confirmed that alpha, beta and the rolling window are loaded
from the selected parameter JSON, and that `MIN_THRESHOLDS` is defined
in this implementation.

The exact `MIN_THRESHOLDS` dictionary was established earlier in the
implementation:

``` python
MIN_THRESHOLDS = {
    "dark": {
        "rgb_only": 0.15,
        "ir_only": 0.20,
    },
    "light": {
        "rgb_only": 0.15,
        "ir_only": 0.08,
    },
}
```

### 10.6 Startup-window limitation

At the beginning of a sequence, the rolling window has fewer than 30
observed frames. The estimator still divides detection rate by (W=30),
so the first estimates are based on partial history and may be lower or
less stable than estimates after the window is full.

For a real-time deployment, one possible mitigation is to pre-populate
the reliability history using a calibration recording before beginning
live inference. This is a documented deployment consideration, not an
implemented feature in this phase.

------------------------------------------------------------------------

## 11. Final status and limitations

### Verified

-   Rolling reliability estimator implemented and validated.
-   Alpha/beta grid search completed.
-   Selected parameters validated and stored.
-   Thermal bounding-box transformation implemented.
-   Reliability-aware adaptive fusion implemented.
-   Scene-specific unmatched thresholds applied.
-   Light and Dark fused CSVs passed the Day 15 CSV/parameter audit.
-   Fused confidence and coordinates passed finite/range checks.
-   ByteTrack preparation completed using the existing, unchanged
    script.
-   Both uploaded ByteTrack audit CSVs passed structural and
    data-integrity checks.
-   Additional LLVIP paired videos were qualitatively reported as
    satisfactory.

### Documented deviations and limitations

1.  **Day 14 clip substitution:** the available Light and Dark
    recordings were used instead of the exact Light, Dark-13 and Dark-15
    clips specified in the roadmap.
2.  **Dark matched-weight target:** mean RGB contribution was 14.78%,
    not below the strict 10% target. IR contribution was 85.22%.
3.  **LLVIP validation evidence:** additional-video validation was
    qualitative; exact filenames and numerical outputs were not
    supplied.
4.  **Startup window:** initial reliability estimates use fewer than 30
    historical frames while retaining a denominator of 30.
5.  **Threshold tuning:** scene-specific unmatched gates are
    experimental settings. Candidate-drop counts do not prove all
    dropped detections were false positives.
6.  **Bounding-box geometry:** the earlier Light box audit found a
    maximum 1-pixel `y2` discrepancy, which was reviewed and accepted.

### Overall conclusion

The reliability-aware adaptive fusion implementation is operational on
the available Light and Dark recordings. The final CSV/parameter audit
passed, and both generated ByteTrack audit files passed structural and
data-integrity checks.

The results demonstrate the intended qualitative modality preference:
RGB is dominant in Light, while IR is dominant in Dark. The Dark
matched-weight sample did not meet the strict below-10% RGB target, and
the exact Dark-13/Dark-15 validation remains unverified because those
clips were unavailable.

Accordingly, the implementation and available-data audit are documented
as passed, with the above experimental deviations retained
transparently. This report does not claim that every original roadmap
target was met.

------------------------------------------------------------------------

## 12. Suggested Git commit

``` text
feat(fusion): implement reliability-aware adaptive RGB-IR fusion
```

A more descriptive alternative:

``` text
feat(fusion): add rolling reliability, adaptive weighting and validation
```
