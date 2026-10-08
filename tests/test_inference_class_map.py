"""Unit tests for the inference contract that do not require real trained weights."""
import json

import pytest

from src.inference.predictor import load_class_names


def test_class_map_sorted_by_training_index(tmp_path):
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"healthy": 1, "diseased": 0}), encoding="utf-8")
    assert load_class_names(path) == ["diseased", "healthy"]


@pytest.mark.parametrize("mapping", [
    {},
    {"healthy": 1},
    {"a": 0, "b": 0},
    {"a": True},
    {"": 0},
    {"a": 0.0},
    ["healthy"],
])
def test_invalid_class_mapping_fails_closed(tmp_path, mapping):
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError):
        load_class_names(path)


def test_missing_class_map_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_class_names(tmp_path / "not-found.json")
