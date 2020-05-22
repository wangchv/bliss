import numpy as np
import torch
import json
from astropy.io import fits

from celeste.datasets import simulated_datasets
from celeste import utils

psf_r = fits.getdata(utils.data_path.joinpath("sdss-002583-2-0136-psf-r.fits"))
psf_i = fits.getdata(utils.data_path.joinpath("sdss-002583-2-0136-psf-i.fits"))
psf_og = torch.Tensor(np.array([psf_r, psf_i])).to(utils.device)  # waiting for new push

param_file = utils.config_path.joinpath("dataset_params/default_star_parameters.json")
with open(param_file, "r") as fp:
    data_params = json.load(fp)

data_params["slen"] = 50
data_params["mean_stars"] = 40


# TODO: Test galaxy dataset and batchsize=1 separately.
class TestSDSSDataset:
    def test_fresh_data(self):
        # this checks that we are actually drawing fresh data
        # at each epoch (or not)

        n_images = 80
        # get dataset
        background = (
            torch.ones(psf_og.shape[0], data_params["slen"], data_params["slen"])
            * 686.0
        )
        star_dataset = simulated_datasets.StarDataset.load_dataset_from_params(
            n_images=n_images,
            data_params=data_params,
            psf=psf_og,
            background=background,
            transpose_psf=False,
            add_noise=True,
        )

        #############################################
        # Check: by creating new batches fresh data
        # should be drawn
        #############################################
        num_epoch = 5
        images_vec = torch.zeros(num_epoch).to(utils.device)
        locs_vec = torch.zeros(num_epoch).to(utils.device)
        fluxes_vec = torch.zeros(num_epoch).to(utils.device)
        n_sources_vec = torch.zeros(num_epoch).to(utils.device)

        # get batch
        batchsize = 8
        num_batches = int(len(star_dataset) / batchsize)

        assert len(star_dataset) == n_images
        assert (
            len(star_dataset) >= num_epoch * batchsize * 2
        ), "Dataset repeats after n_images "

        for i in range(num_epoch):
            images_mean = 0
            true_locs_mean = 0
            true_fluxes_mean = 0
            true_n_sources_mean = 0

            for j in range(num_batches):
                batch = star_dataset.get_batch(batchsize)

                true_fluxes = batch["fluxes"]
                true_locs = batch["locs"]
                true_n_sources = batch["n_sources"]
                images = batch["images"]

                images_mean += images.mean()
                true_locs_mean += true_locs.mean()
                true_fluxes_mean += true_fluxes.mean()
                true_n_sources_mean += true_n_sources.float().mean()

            images_vec[i] = images_mean
            locs_vec[i] = true_locs_mean
            fluxes_vec[i] = true_fluxes_mean
            n_sources_vec[i] = true_n_sources_mean

        assert len(images_vec.unique()) == num_epoch
        assert len(locs_vec.unique()) == num_epoch
        assert len(fluxes_vec.unique()) == num_epoch
        assert len(n_sources_vec.unique()) == num_epoch