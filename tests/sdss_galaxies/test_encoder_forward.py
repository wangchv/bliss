import numpy as np
import torch

from bliss.models.location_encoder import (
    LocationEncoder,
    get_is_on_from_n_sources,
    get_images_in_tiles,
)


class TestSourceEncoder:
    def test_forward(self, devices):
        """Tests forward function of source encoder.

        Arguments:
            devices: GPU device information.

        Notes:
            * Test that forward returns the correct pattern of zeros.
            * Test that variational parameters inside h agree with those returned from forward.
        """
        device = devices.device

        n_image_tiles = 30
        max_detections = 4
        ptile_slen = 10
        n_bands = 2
        tile_slen = 2

        # get encoder
        star_encoder = LocationEncoder(
            channel=8,
            dropout=0,
            hidden=64,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            n_bands=n_bands,
            max_detections=max_detections,
        ).to(device)

        with torch.no_grad():
            star_encoder.eval()

            # simulate image padded tiles
            image_ptiles = (torch.randn(n_image_tiles, n_bands, ptile_slen, ptile_slen) + 10.0).to(
                device
            )

            n_star_per_tile = (
                torch.from_numpy(np.random.choice(max_detections, n_image_tiles))
                .type(torch.LongTensor)
                .to(device)
            )

            var_params = star_encoder.encode(image_ptiles)
            pred = star_encoder.encode_for_n_sources(var_params, n_star_per_tile)

            assert torch.all(pred["loc_mean"] >= 0.0)
            assert torch.all(pred["loc_mean"] <= 1.0)

            # test we have the correct pattern of zeros
            assert ((pred["loc_mean"] != 0).sum(1)[:, 0] == n_star_per_tile).all()
            assert ((pred["loc_mean"] != 0).sum(1)[:, 1] == n_star_per_tile).all()

            assert ((pred["loc_logvar"] != 0).sum(1)[:, 0] == n_star_per_tile).all()
            assert ((pred["loc_logvar"] != 0).sum(1)[:, 1] == n_star_per_tile).all()

            for i in range(2):
                assert ((pred["loc_mean"][:, :, i] != 0).sum(1) == n_star_per_tile).all()

            for b in range(n_bands):
                assert ((pred["log_flux_mean"][:, :, b] != 0).sum(1) == n_star_per_tile).all()

            # check pattern of zeros
            is_on_array = get_is_on_from_n_sources(n_star_per_tile, star_encoder.max_detections)
            loc_mean = pred["loc_mean"] * is_on_array.unsqueeze(2).float()
            log_flux_mean = pred["log_flux_mean"] * is_on_array.unsqueeze(2).float()
            assert torch.all(loc_mean == pred["loc_mean"])
            assert torch.all(log_flux_mean == pred["log_flux_mean"])

            # we check the variational parameters against the hidden parameters
            # one by one
            h_out = star_encoder.encode(image_ptiles)

            # get index matrices
            locs_mean_indx_mat = star_encoder.loc_mean_indx
            locs_var_indx_mat = star_encoder.loc_logvar_indx
            log_flux_mean_indx_mat = star_encoder.log_flux_mean_indx
            log_flux_var_indx_mat = star_encoder.log_flux_logvar_indx
            prob_n_source_indx_mat = star_encoder.prob_n_source_indx

            for i in range(n_image_tiles):
                if n_star_per_tile[i] == 0:
                    assert torch.all(pred["loc_mean"][i] == 0)
                    assert torch.all(pred["loc_logvar"][i] == 0)
                    assert torch.all(pred["log_flux_mean"][i] == 0)
                    assert torch.all(pred["log_flux_logvar"][i] == 0)
                else:
                    n_stars_i = int(n_star_per_tile[i])

                    assert torch.all(
                        pred["loc_mean"][i, :n_stars_i].flatten()
                        == torch.sigmoid(h_out)[
                            i,
                            locs_mean_indx_mat[n_stars_i][: (2 * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["loc_logvar"][i, :n_stars_i].flatten()
                        == h_out[
                            i,
                            locs_var_indx_mat[n_stars_i][: (2 * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["log_flux_mean"][i, :n_stars_i].flatten()
                        == h_out[
                            i,
                            log_flux_mean_indx_mat[n_stars_i][: (n_bands * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["log_flux_logvar"][i, :n_stars_i].flatten()
                        == h_out[
                            i,
                            log_flux_var_indx_mat[n_stars_i][: (n_bands * n_stars_i)],
                        ]
                    )

                    assert torch.all(
                        pred["n_source_log_probs"][i].flatten()
                        == star_encoder.log_softmax(h_out[:, prob_n_source_indx_mat])[i]
                    )

    def test_sample(self, devices):
        device = devices.device

        max_detections = 4
        ptile_slen = 10
        n_bands = 2
        tile_slen = 2
        n_samples = 5

        images = torch.randn(1, n_bands, 4 * ptile_slen, 4 * ptile_slen).to(device)

        star_encoder = LocationEncoder(
            channel=8,
            dropout=0,
            hidden=64,
            ptile_slen=ptile_slen,
            tile_slen=tile_slen,
            n_bands=n_bands,
            max_detections=max_detections,
        ).to(device)
        image_ptiles = get_images_in_tiles(images, tile_slen, ptile_slen)
        var_params = star_encoder.encode(image_ptiles)
        star_encoder.sample(var_params, n_samples)