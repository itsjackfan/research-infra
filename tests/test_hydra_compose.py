from research_infra.hydra_compose import compose
from conftest import CONFIG


def test_compose(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG)
    (tmp_path / "eval.yaml").write_text("split: test\n")
    assert compose(tmp_path, overrides=["lr=0.2"]).lr == 0.2
    assert compose(tmp_path, "eval").split == "test"
