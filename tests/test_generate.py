import os

from hydra import initialize

from bliss import generate
from bliss.utils import path_resolved_compose


def test_generate_run(devices, paths):
    overrides = {
        "dataset.generate_device": "cuda:0" if devices.use_cuda else "cpu",
        "generate.file": f"{paths['root']}/example.pt",
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = path_resolved_compose("config", overrides=overrides)
        generate.generate(cfg)

    os.remove(f"{paths['root']}/example.pt")
    os.remove(f"{paths['root']}/example_images.pdf")
