from tennis_autoedit import segmentation

def test_smooth_vlm_predictions():
    # Test bridging [True, False, True]
    preds = [True, False, True, False, False, True, False, True]
    smoothed = segmentation.smooth_vlm_predictions(preds)
    assert smoothed == [True, True, True, False, False, True, True, True]

    # Test no change if no [True, False, True]
    preds2 = [True, False, False, True]
    assert segmentation.smooth_vlm_predictions(preds2) == [True, False, False, True]

def test_filter_short_segments():
    segments = [(0.0, 5.0), (10.0, 11.0), (20.0, 25.0)]
    filtered = segmentation.filter_short_segments(segments, min_duration=2.0)
    assert filtered == [(0.0, 5.0), (20.0, 25.0)]

if __name__ == "__main__":
    test_smooth_vlm_predictions()
    test_filter_short_segments()
    print("Smoke tests passed!")
