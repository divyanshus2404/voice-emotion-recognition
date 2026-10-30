import numpy as np
import pytest

from emotion import EmotionPredictor


@pytest.fixture(scope="module")
def predictor():
    return EmotionPredictor()


def tone(seconds=3.0, sr=44_100):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32), sr


def test_scores_are_probabilities(predictor):
    scores = predictor.predict(*tone())
    assert set(scores) == set(predictor.labels)
    assert abs(sum(scores.values()) - 1) < 1e-4


def test_handles_stereo_int16_mic_input(predictor):
    audio, sr = tone()
    stereo = (np.stack([audio, audio], axis=1) * 20_000).astype(np.int16)
    assert len(predictor.predict(stereo, sr)) == len(predictor.labels)


def test_timeline_windows(predictor):
    timeline = predictor.timeline(*tone(seconds=5))
    assert len(timeline) >= 3
    assert timeline[1][0] == pytest.approx(1.0)
