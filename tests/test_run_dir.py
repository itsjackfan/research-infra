import pytest
from omegaconf import OmegaConf

from research_infra.run_dir import mark_done, prepare_run_dir

YAML = OmegaConf.to_yaml(OmegaConf.create({"name": "base", "lr": 0.1, "git": {"commit": "abc"}}), resolve=True)


def test_new_resume_done(tmp_path):
    run_dir = tmp_path / "exp/outputs/base"
    assert prepare_run_dir(run_dir, YAML) == "new"
    assert (run_dir / "config.yaml").read_text() == YAML
    assert prepare_run_dir(run_dir, YAML) == "resume"
    mark_done(run_dir)
    assert prepare_run_dir(run_dir, YAML) == "done"


def test_job_side_roundtrip_resumes(tmp_path):
    prepare_run_dir(tmp_path, YAML)
    loaded = OmegaConf.to_yaml(OmegaConf.load(tmp_path / "config.yaml"), resolve=True)
    assert prepare_run_dir(tmp_path, loaded) == "resume"


def test_different_config_refused_unless_forced(tmp_path):
    prepare_run_dir(tmp_path / "r", YAML)
    (tmp_path / "r/latest.pt").write_text("ckpt")
    other = YAML.replace("lr: 0.1", "lr: 0.2")
    with pytest.raises(SystemExit, match="--force-override"):
        prepare_run_dir(tmp_path / "r", other)
    assert (tmp_path / "r/config.yaml").read_text() == YAML
    assert prepare_run_dir(tmp_path / "r", other, force_override=True) == "new"
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == ["config.yaml"]
    assert (tmp_path / "r/config.yaml").read_text() == other
