from pathlib import Path

import pytest
from hydra import compose, initialize

from tests.conftest import ModelSetup


def get_m2_cfg(overrides, devices):
    overrides.update({"gpus": devices.gpus, "paths.root": Path(__file__).parents[3].as_posix()})
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path="."):
        cfg = compose("m2", overrides=overrides)
    return cfg


class M2ModelSetup(ModelSetup):
    def get_cfg(self, overrides):
        return get_m2_cfg(overrides, self.devices)


@pytest.fixture(scope="session")
def m2_model_setup(devices):
    return M2ModelSetup(devices)
