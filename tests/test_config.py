from pathlib import Path

import pytest
import yaml

from config.config import CfgNode


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_yaml_with_base_merges_nested_values(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    child = tmp_path / "child.yml"

    write_yaml(
        base,
        {
            "MODEL": {"NAME": "baseline", "DIM": 256},
            "TRAIN": {"BATCH_SIZE": 8},
        },
    )
    write_yaml(
        child,
        {
            "_BASE_": "base.yml",
            "MODEL": {"DIM": 512},
        },
    )

    loaded = CfgNode.load_yaml_with_base(str(child))

    assert loaded["MODEL"]["NAME"] == "baseline"
    assert loaded["MODEL"]["DIM"] == 512
    assert loaded["TRAIN"]["BATCH_SIZE"] == 8


def test_computed_attributes_are_insertion_only() -> None:
    cfg = CfgNode({"MODEL": {"DIM": 256}})

    cfg.COMPUTED_FEATURE_DIM = 512
    cfg.COMPUTED_FEATURE_DIM = 512

    assert cfg.COMPUTED_FEATURE_DIM == 512

    with pytest.raises(KeyError):
        cfg.COMPUTED_FEATURE_DIM = 1024
