import numpy as np
import pytest

from squat_metrics import analyze_squat, bar_path_analysis

FPS = 30


def vertical_bar(n_frames):
    return np.column_stack([np.full(n_frames, 250.0), np.linspace(150, 370, n_frames)])


def test_counts_reps_and_tempo(make_squat):
    xy, conf = make_squat()
    result = analyze_squat(xy, conf, vertical_bar(len(xy)), FPS)

    assert result["total_reps"] == 3
    assert [r["bottom_frame"] for r in result["reps"]] == [45, 135, 225]
    np.testing.assert_allclose(result["tempo_per_rep"], [3.0, 3.0])


def test_video_ending_mid_descent_does_not_add_a_rep(make_squat):
    #300 frames stops a third of the way into a 4th descent
    xy, conf = make_squat(n_frames=300)
    assert analyze_squat(xy, conf, vertical_bar(300), FPS)["total_reps"] == 3


def test_rep_bottom_near_end_of_video_still_counts(make_squat):
    #the last bottom is at frame 225, only 14 frames before the video ends
    xy, conf = make_squat(n_frames=240)
    assert analyze_squat(xy, conf, vertical_bar(240), FPS)["total_reps"] == 3


def test_shaking_while_standing_is_not_a_rep(make_squat):
    xy, conf = make_squat(bottom_y=200, tremor=30)
    assert analyze_squat(xy, conf, vertical_bar(len(xy)), FPS)["total_reps"] == 0


def test_shaking_during_reps_does_not_add_reps(make_squat):
    #270 frames ends standing, so only the prominence filter is being tested
    xy, conf = make_squat(n_frames=270, tremor=15)
    assert analyze_squat(xy, conf, vertical_bar(len(xy)), FPS)["total_reps"] == 3


@pytest.mark.xfail(strict=True, reason=(
    "known bug: rep_count only drops a bottom on the very last frame, so shake in a video "
    "that ends mid-descent moves the high point a few frames earlier and it counts as a rep"))
def test_shaky_video_ending_mid_descent_does_not_add_a_rep(make_squat):
    xy, conf = make_squat(n_frames=300, tremor=15)
    assert analyze_squat(xy, conf, vertical_bar(len(xy)), FPS)["total_reps"] == 3


@pytest.mark.parametrize("bottom_y, expected", [
    (420, "below"),
    (391, "parallel"),
    (340, "partial"),
])
def test_depth_grade(make_squat, bottom_y, expected):
    xy, conf = make_squat(bottom_y=bottom_y)
    reps = analyze_squat(xy, conf, vertical_bar(len(xy)), FPS)["reps"]
    assert [r["depth"] for r in reps] == [expected] * 3


def test_bar_path_deviation():
    n = 30
    straight = vertical_bar(n)
    swaying = straight.copy()
    swaying[:, 0] += 20 * np.sin(np.linspace(0, 2 * np.pi, n))
    missing = np.full((n, 2), np.nan)

    assert bar_path_analysis(straight, 0, n, FPS) == 0
    assert bar_path_analysis(swaying, 0, n, FPS) > 10
    assert np.isnan(bar_path_analysis(missing, 0, n, FPS))
