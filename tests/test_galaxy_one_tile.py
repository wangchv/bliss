import pytest
import torch


class TestGalaxyEncoder:
    @pytest.fixture(scope="class")
    def trained_encoder(self, get_dataset, get_trained_encoder, devices):
        overrides = ["model=basic_sleep_galaxy", "training=tests"]
        use_cuda = devices.use_cuda
        batch_size = 32 if use_cuda else 2
        n_batches = 10 if use_cuda else 2
        n_epochs = 120 if use_cuda else 1
        dataset = get_dataset(batch_size, n_batches, overrides)
        trained_encoder = get_trained_encoder(n_epochs, dataset, overrides)
        return trained_encoder.to(devices.device)

    @pytest.mark.parametrize("n_galaxies", ["2"])
    def test_n_sources_and_locs(self, trained_encoder, n_galaxies, paths, devices):
        use_cuda = devices.use_cuda
        device = devices.device

        test_galaxy = torch.load(paths["data"].joinpath(f"{n_galaxies}_galaxy_test.pt"))
        test_image = test_galaxy["images"]

        with torch.no_grad():
            # get the estimated params
            trained_encoder.eval()
            (
                n_sources,
                locs,
                galaxy_params,
                log_fluxes,
                galaxy_bool,
            ) = trained_encoder.map_estimate(test_image.to(device))

        if not use_cuda:
            return

        # check number of galaxies is correct
        assert test_galaxy["n_galaxies"].item() == n_sources.item()

        # check locations are accurate.
        diff_locs = test_galaxy["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
        diff_locs *= test_image.size(-1)
        assert diff_locs.abs().max() <= 2.5