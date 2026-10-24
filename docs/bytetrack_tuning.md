# Phase 4 — ByteTrack Parameter Tuning

## Objective

Reduce track fragmentation in the 280-frame RGB video while preserving the original tracking results and evaluating continuity rather than optimizing only for the number of track IDs.

## Baseline

| Parameter / Metric | Value |
|---|---:|
| track_thresh | 0.45 |
| track_buffer | 25 |
| match_thresh | 0.80 |
| frame_rate | 30 |
| Input detections | 5,598 |
| Tracked rows | 3,518 |
| Unique track IDs | 77 |
| Tracks under 5 observed frames | 12 |
| Tracks with internal gaps | 25 |
| Internal missing frames | 250 |

## Experiments

| Run | track_thresh | track_buffer | match_thresh | Tracked rows | Unique IDs | Tracks under 5 frames | Tracks with gaps | Missing frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.45 | 25 | 0.80 | 3,518 | 77 | 12 | 25 | 250 |
| Test A | 0.35 | 50 | 0.70 | 3,861 | 132 | 42 | 58 | 691 |
| Test B | 0.35 | 50 | 0.65 | 3,631 | 174 | 86 | 61 | 558 |
| Test C | 0.45 | 50 | 0.80 | 3,534 | 73 | 10 | 28 | 493 |
| Test D | 0.45 | 50 | 0.85 | 3,583 | 67 | 11 | 26 | 488 |
| Test E | 0.45 | 50 | 0.90 | 3,623 | 64 | 10 | 30 | 472 |

## Visual Evaluation

The baseline and Test E videos were generated from the same 280-frame source video. Visual review preferred Test E.

## Selected Candidate

**Test E**

- track_thresh = 0.45
- track_buffer = 50
- match_thresh = 0.90
- frame_rate = 30

Compared with baseline, Test E reduced unique IDs from 77 to 64 (approximately 16.9%) and increased tracked rows from 3,518 to 3,623.

## Interpretation and Limitations

Test E is the preferred candidate from the tested configurations, but tracking continuity is not fully resolved. Internal missing frames increased from 250 to 472, and tracks with internal gaps increased from 25 to 30.

The experiment results do not establish that every track ID corresponds consistently to one real person. The fewer-than-10-ID target was not achieved.

The baseline files remain unchanged. Test E is a selected candidate pending the remaining Phase 4 closure checks, not a claim that tracking is fully solved.

## Artifacts

- `outputs/tracked_detections.csv`
- `outputs/tracked_detections_test_a.csv` through `outputs/tracked_detections_test_e.csv`
- `outputs/bytetrack_tuning_summary.csv`
- `outputs/bytetrack_comparison_baseline.mp4`
- `outputs/bytetrack_comparison_test_e.mp4`

Scripts:
- `src/tracking/tune_bytetrack.py`
- `src/tracking/compare_tuning_visualizations.py`
