from firephantom.evaluation import annotation_accuracy, box_iou, detection_map, physics_error


def test_evaluation_metrics_report_acceptance_quantities():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    annotations = annotation_accuracy(
        [{"objects": [{"category": "flame", "bbox": [0, 0, 10, 10]}]}],
        [{"objects": [{"category": "flame", "bbox": [1, 1, 10, 10]}]}],
    )
    assert annotations["precision"] == 1.0 and annotations["recall"] == 1.0
    detection = detection_map(
        [{"boxes": [[0, 0, 10, 10]], "labels": ["flame"], "scores": [0.9]}],
        [{"boxes": [[0, 0, 10, 10]], "labels": ["flame"]}],
    )
    assert detection["map"] == 1.0
    error = physics_error(
        {"temperature": [1.0, 2.0], "smoke": [1.0], "flow": [1.0, 2.0]},
        {"temperature": [1.1, 2.0], "smoke": [1.0], "flow": [1.0, 2.0]},
    )
    assert error["mean_percent"] > 0.0


def test_detection_map_counts_a_ground_truth_class_without_predictions():
    result = detection_map(
        [{"boxes": [[0, 0, 10, 10]], "labels": ["flame"], "scores": [0.9]}],
        [{"boxes": [[0, 0, 10, 10], [20, 20, 5, 5]], "labels": ["flame", "smoke"]}],
    )
    assert result["classes"] == ["flame", "smoke"]
    assert result["average_precision"]["flame"] == 1.0
    assert result["average_precision"]["smoke"] == 0.0
    assert result["map"] == 0.5


def test_detection_metrics_reject_malformed_inputs():
    try:
        detection_map(
            [{"boxes": [[0, 0, 1, 1]], "labels": [], "scores": [1.0]}],
            [{"boxes": [[0, 0, 1, 1]], "labels": ["flame"]}],
        )
    except ValueError as exc:
        assert "one label" in str(exc)
    else:
        raise AssertionError("mismatched prediction labels should be rejected")
