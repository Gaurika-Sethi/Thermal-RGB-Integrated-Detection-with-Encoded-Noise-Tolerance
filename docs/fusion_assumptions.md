
# Phase 3: RGB-Thermal Fusion Assumptions

## 1. Objective

Combine RGB and thermal person detections into fused centroid
measurements using confidence-weighted position fusion.

## 2. Input data

Both detection CSVs use the following schema:

- frame
- x
- y
- width
- height
- confidence

Each valid detection represents one detected person in one frame.
Multiple detections per frame are supported.

## 3. Frame alignment

RGB and thermal detections are grouped by frame number.

Frame-number alignment assumes both videos are synchronized so
that corresponding frame numbers represent the same scene time.

Matching frame numbers alone does not establish person identity.

## 4. Person association

Within each frame, the pipeline calculates Euclidean distances
between RGB and thermal detection centroids.

Candidate pairs are sorted by increasing distance. A greedy
one-to-one matching procedure accepts pairs within 150 pixels,
provided neither detection has already been matched.

This is a heuristic. It does not guarantee globally optimal
assignment or correct identity association.

## 5. Confidence-weighted fusion

For a matched pair:

fused_x = (rgb_conf * rgb_x + thermal_conf * thermal_x)
          / (rgb_conf + thermal_conf)

fused_y = (rgb_conf * rgb_y + thermal_conf * thermal_y)
          / (rgb_conf + thermal_conf)

fused_confidence = (rgb_conf + thermal_conf) / 2

If only one modality detects a person, its position and
confidence are retained directly.

If neither modality detects a person, the fused position remains
missing. No artificial coordinates are generated.

## 6. Coordinate assumptions

The fusion calculation assumes RGB and thermal centroids are
expressed in comparable image coordinate systems.

Similar centroid values suggest compatibility but do not prove
geometric calibration or pixel-level alignment.

The distance threshold must therefore be evaluated against the
actual footage.

## 7. Output CSV

The fused output preserves:

- frame
- fused_x
- fused_y
- fused_confidence
- source
- match_distance
- rgb_x
- rgb_y
- rgb_confidence
- thermal_x
- thermal_y
- thermal_confidence

The source column records whether a row came from a matched pair,
an RGB-only detection, a thermal-only detection, or no detection.

## 8. Known limitations

- Greedy matching can associate the wrong people in crowded scenes.
- A fixed distance threshold may not work equally well across all
  parts of an image.
- Detection confidence is treated as a fusion weight, not as a
  calibrated probability of correctness.
- This phase does not maintain persistent person identities.
- This phase does not implement Kalman tracking.

## 9. Validation

Validation includes CSV consistency checks, formula verification,
source distribution inspection, and manual visual inspection of
representative frames.

Passing these checks does not establish ground-truth accuracy
for every detection or association.
