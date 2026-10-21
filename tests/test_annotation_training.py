import yaml

from firephantom.annotation.evaluate import _stratum, sample_strata
from firephantom.annotation.train import normalize_data_yaml


def test_training_data_yaml_resolves_parent_relative_roboflow_paths(tmp_path):
    dataset = tmp_path / "FASDD_CV"
    for split in ("train", "valid", "test"):
        (dataset / split / "images").mkdir(parents=True)
    source = dataset / "data.yaml"
    source.write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\n",
        encoding="utf-8",
    )

    normalized = normalize_data_yaml(source, tmp_path / "run" / "data.firephantom.yaml")
    config = yaml.safe_load(normalized.read_text(encoding="utf-8"))

    assert config["train"] == str((dataset / "train" / "images").resolve())
    assert config["val"] == str((dataset / "valid" / "images").resolve())
    assert config["test"] == str((dataset / "test" / "images").resolve())


def test_annotation_evaluation_stratifies_and_samples_deterministically(tmp_path):
    items = []
    for index, class_ids in enumerate(({0}, {1}, {0, 1}, set())):
        items.append((tmp_path / f"{index}.jpg", tmp_path / f"{index}.txt", set(class_ids)))
    sampled = sample_strata(items * 2, per_stratum=1, seed=7)

    assert [_stratum(item[2]) for item in sampled] == ["fire", "smoke", "both", "background"]
    assert sample_strata(items * 2, per_stratum=1, seed=7) == sampled
