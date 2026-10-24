# ByteTrack Validation --- Short-Track Check Correction

## Purpose

This change fixes a contradiction in the ByteTrack validation report.
The earlier report used a different short-track threshold in its
validation logic and wording. The corrected implementation uses one
consistent definition: a track is short when it has **fewer than 5
observed frames**.

## Definition used

The count is based on the number of unique observed frame numbers for
each `track_id`, not the span between its first and last frame. A track
observed in four frames is short; a track observed in five frames is
not.

## Logic change

The validation script uses one threshold for both the hard check and the
displayed statistic:

``` python
SHORT_TRACK_THRESHOLD = 5

short_tracks = int(
    (track_summary["observed_frames"] < SHORT_TRACK_THRESHOLD).sum()
)

check(
    short_tracks == 0,
    (
        "No very short tracks "
        f"(fewer than {SHORT_TRACK_THRESHOLD} observed frames); "
        f"found {short_tracks}"
    ),
)
```

The summary output and text report use the same `SHORT_TRACK_THRESHOLD`
value to avoid inconsistent thresholds or contradictory wording.

## Verified local results

The user ran the corrected validation script locally and supplied both
the terminal output and the regenerated text report. The results agree:

  Metric                                            Verified result
  ---------------------------------------- ------------------------
  Input detection rows                                        5,598
  Tracked output rows                                         3,518
  Output/input row ratio                                     62.84%
  Input unique frames                                           280
  Tracked unique frames                                         280
  Unique track IDs                                               77
  Duplicate frame/ID rows                                         0
  Tracks observed in fewer than 5 frames                         12
  Failed hard checks                                              1
  Validation result                          INVESTIGATION REQUIRED

The local report is saved at:

`outputs/bytetrack_validation_report.txt`

Supporting summaries are saved at:

-   `outputs/track_summary.csv` --- per-track lifetime and continuity
    statistics.
-   `outputs/track_frame_summary.csv` --- per-frame tracking statistics.

## Interpretation

**The reporting correction is verified locally.** The short-track check
still fails because 12 tracks meet the defined condition. This is an
honest tracking-quality warning, not a remaining reporting
inconsistency. Do not suppress the warning or change the threshold
merely to make the report pass.

The count differs from the earlier count of four because the earlier
check used a threshold of fewer than 3 observed frames. The corrected
threshold also includes tracks with 3 or 4 observed frames.

This correction does not improve tracking quality by itself. The 12
short tracks may warrant further investigation. Structural validation
and visual identity continuity are separate: passing other checks does
not prove that each person retains the correct identity across the
video.

## Files

-   `src/tracking/validate_bytetrack.py` --- canonical validation
    script.
-   `outputs/bytetrack_validation_report.txt` --- locally regenerated
    validation report.
-   `outputs/track_summary.csv` --- per-track statistics.
-   `outputs/track_frame_summary.csv` --- per-frame statistics.

Reference artifacts produced during the correction may be kept as
reference copies, but the verified result above is based on the user's
local execution and local report.

## Completion checklist

-   [x] Define one threshold: fewer than 5 observed frames.
-   [x] Use the same threshold in the short-track count and report
    wording.
-   [x] Regenerate the report and supporting summaries.
-   [x] Run the corrected script in the local project.
-   [x] Confirm the local output reports 12 tracks with fewer than 5
    observed frames.
-   [x] Confirm the local text report records the same count and one
    failed hard check.
-   [x] Preserve the `INVESTIGATION REQUIRED` result honestly.
