"""Implements sleep-phase training of BLISS models.

This module contains the SleepPhase class, which implements the sleep-phase training using
pytorch-lightning framework. Users should use this class to construct the sleep-phase
model.

"""

import math
from itertools import permutations

import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from bliss.models.decoder import ImageDecoder
from bliss.models.location_encoder import (
    LocationEncoder,
    get_full_params_from_tiles,
    get_images_in_tiles,
    get_is_on_from_n_sources,
    get_params_in_batches,
)
from bliss.reporting import DetectionMetrics, plot_image_and_locs

plt.switch_backend("Agg")


def _get_log_probs_all_perms(
    locs_log_probs_all,
    star_params_log_probs_all,
    true_galaxy_bool,
    is_on_array,
):
    # get log-probability under every possible matching of estimated source to true source
    n_ptiles = star_params_log_probs_all.size(0)
    max_detections = star_params_log_probs_all.size(-1)

    n_permutations = math.factorial(max_detections)
    locs_log_probs_all_perm = torch.zeros(
        n_ptiles, n_permutations, device=locs_log_probs_all.device
    )
    star_params_log_probs_all_perm = locs_log_probs_all_perm.clone()

    for i, perm in enumerate(permutations(range(max_detections))):
        # note that we multiply is_on_array, we only evaluate the loss if the source is on.
        locs_log_probs_all_perm[:, i] = (
            locs_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

        # if star, evaluate the star parameters,
        # hence the multiplication by (1 - true_galaxy_bool)
        # the diagonal is a clever way of selecting the elements of each permutation (first index
        # of mean/var with second index of true_param etc.)
        star_params_log_probs_all_perm[:, i] = (
            star_params_log_probs_all[:, perm].diagonal(dim1=1, dim2=2)
            * is_on_array
            * (1 - true_galaxy_bool)
        ).sum(1)

    return locs_log_probs_all_perm, star_params_log_probs_all_perm


def get_min_perm_loss(
    locs_log_probs_all,
    star_params_log_probs_all,
    true_galaxy_bool,
    is_on_array,
):
    # get log-probability under every possible matching of estimated star to true star
    locs_log_probs_all_perm, star_params_log_probs_all_perm = _get_log_probs_all_perms(
        locs_log_probs_all,
        star_params_log_probs_all,
        true_galaxy_bool,
        is_on_array,
    )

    # find the permutation that minimizes the location losses
    locs_loss, indx = torch.min(-locs_log_probs_all_perm, dim=1)
    indx = indx.unsqueeze(1)

    # get the star losses according to the found permutation.
    star_params_loss = -torch.gather(star_params_log_probs_all_perm, 1, indx).squeeze()
    return locs_loss, star_params_loss


def get_params_logprob_all_combs(true_params, param_mean, param_logvar):
    # return shape (n_ptiles x max_detections x max_detections)
    assert true_params.shape == param_mean.shape == param_logvar.shape

    n_ptiles = true_params.size(0)
    max_detections = true_params.size(1)

    # view to evaluate all combinations of log_prob.
    true_params = true_params.view(n_ptiles, 1, max_detections, -1)
    param_mean = param_mean.view(n_ptiles, max_detections, 1, -1)
    param_logvar = param_logvar.view(n_ptiles, max_detections, 1, -1)

    sd = (param_logvar.exp() + 1e-5).sqrt()
    return Normal(param_mean, sd).log_prob(true_params).sum(dim=3)


class SleepPhase(pl.LightningModule):
    """Implementation of sleep-phase training using pytorch-lightning framework.

    Example:
        In python script, set up the sleep-phase model and pytorch-lightning trainer::

            import pytorch_lightning as pl
            from bliss.sleep import SleepPhase

            model = SleepPhase(**cfg.model.kwargs)
            trainer = pl.Trainer()
            trainer.fit(model, data=dataset)
    """

    def __init__(
        self,
        encoder: LocationEncoder,
        decoder: ImageDecoder,
        annotate_probs: bool = False,
        slack=1.0,
        optimizer_params: dict = None,
    ):
        """Initializes SleepPhase class.

        Args:
            encoder: keyword arguments to instantiate ImageEncoder
            decoder: keyword arguments to instantiate ImageDecoder
            annotate_probs: Should probabilities be annotated on plot? Defaults to False.
            slack: Threshold distance in pixels for matching objects.
            optimizer_params: Parameters passed to optimizer. Defaults to None.
        """
        super().__init__()
        self.image_encoder = encoder
        self.image_decoder = decoder
        self.image_decoder.requires_grad_(False)
        self.optimizer_params = optimizer_params

        # consistency
        assert self.image_decoder.tile_slen == self.image_encoder.tile_slen
        assert self.image_decoder.border_padding == self.image_encoder.border_padding

        # plotting
        self.annotate_probs = annotate_probs

        # metrics
        self.val_detection_metrics = DetectionMetrics(slack)
        self.test_detection_metrics = DetectionMetrics(slack)

    def forward(self, image_ptiles, tile_n_sources):
        """Encodes parameters from image tiles."""
        return self.image_encoder(image_ptiles, tile_n_sources)

    def tile_map_estimate(self, batch):
        images = batch["images"]

        batch_size = images.shape[0]
        image_ptiles = get_images_in_tiles(
            images, self.image_encoder.tile_slen, self.image_encoder.ptile_slen
        )
        var_params = self.image_encoder.encode(image_ptiles)
        tile_map = self.image_encoder.max_a_post(var_params)
        tile_est = get_params_in_batches(tile_map, batch_size)
        tile_est["galaxy_params"] = batch["galaxy_params"]

        # FIXME: True galaxy params are not necessarily consistent with other MAP estimates
        # need to do some matching to ensure correctness of residual images?
        # maybe doesn't matter because only care about detection if not estimating
        # galaxy_parameters.
        max_sources = tile_est["locs"].shape[2]
        tile_est["galaxy_params"] = tile_est["galaxy_params"][:, :, :max_sources]
        tile_est["galaxy_params"] = tile_est["galaxy_params"].contiguous()
        return tile_est

    def _get_loss(self, batch):
        """Private method to evaluate loss on the input minibatch.

        Arguments:
            batch: Batch of training data (described below).

        Returns:
            A tuple with the following components:
                loss:
                counter_loss:
                locs_loss:
                star_params_loss:

        Notes:
            loc_mean shape = (n_ptiles x max_detections x 2)
            log_flux_mean shape = (n_ptiles x max_detections x n_bands)

            the *_logvar inputs should the same shape as their respective means
            the true_tile_* inputs, except for true_tile_is_on_array,
            should have same shape as their respective means, e.g.
            true_locs should have the same shape as loc_mean

            In true_locs, the off sources must have parameter value = 0

            true_is_on_array shape = (n_ptiles x max_detections)
                Indicates if sources is on (1) or off (0)

            true_galaxy_bool shape = (n_ptiles x max_detections x 1)
                indicating whether each source is a galaxy (1) or star (0)

            prob_galaxy shape = (n_ptiles x max_detections)
                are probabilities for each source to be a galaxy

            n_source_log_probs shape = (n_ptiles x (max_detections + 1))
                are log-probabilities for the number of sources (0, 1, ..., max_detections)
        """
        (
            images,
            true_tile_locs,
            true_tile_log_fluxes,
            true_tile_galaxy_bool,
            true_tile_n_sources,
        ) = (
            batch["images"],
            batch["locs"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
            batch["n_sources"],
        )

        # some constants
        batch_size = images.shape[0]
        n_tiles_per_image = self.image_decoder.n_tiles_per_image
        n_ptiles = batch_size * n_tiles_per_image
        max_sources = self.image_encoder.max_detections

        # clip decoder output since constraint is: max_detections <= max_sources (per tile)
        true_tile_locs = true_tile_locs[:, :, 0:max_sources]
        true_tile_log_fluxes = true_tile_log_fluxes[:, :, 0:max_sources]
        true_tile_galaxy_bool = true_tile_galaxy_bool[:, :, 0:max_sources]
        true_tile_n_sources = true_tile_n_sources.clamp(max=max_sources)

        # flatten so first dimension is ptile
        # b: batch, s: n_tiles_per_image
        true_tile_locs = rearrange(true_tile_locs, "b n s xy -> (b n) s xy", xy=2)
        true_tile_log_fluxes = rearrange(true_tile_log_fluxes, "b n s bands -> (b n) s bands")
        true_tile_galaxy_bool = rearrange(true_tile_galaxy_bool, "b n s 1 -> (b n) s")
        true_tile_n_sources = rearrange(true_tile_n_sources, "b n -> (b n)")
        true_tile_is_on_array = get_is_on_from_n_sources(true_tile_n_sources, max_sources)

        # extract image tiles
        image_ptiles = get_images_in_tiles(
            images, self.image_encoder.tile_slen, self.image_encoder.ptile_slen
        )
        var_params = self.image_encoder.encode(image_ptiles)
        pred = self.image_encoder.encode_for_n_sources(var_params, true_tile_n_sources)

        # the loss for estimating the true number of sources
        n_source_log_probs = pred["n_source_log_probs"].view(n_ptiles, max_sources + 1)
        cross_entropy = CrossEntropyLoss(reduction="none").requires_grad_(False)
        counter_loss = cross_entropy(n_source_log_probs, true_tile_n_sources)

        # the following two functions computes the log-probability of parameters when
        # each estimated source i is matched with true source j.
        # enforce large error if source is off
        loc_mean, loc_logvar = pred["loc_mean"], pred["loc_logvar"]
        loc_mean = loc_mean + (true_tile_is_on_array == 0).float().unsqueeze(-1) * 1e16
        locs_log_probs_all = get_params_logprob_all_combs(true_tile_locs, loc_mean, loc_logvar)
        star_params_log_probs_all = get_params_logprob_all_combs(
            true_tile_log_fluxes, pred["log_flux_mean"], pred["log_flux_logvar"]
        )

        # inside _get_min_perm_loss is where the matching happens:
        # we construct a bijective map from each estimated source to each true source
        (locs_loss, star_params_loss) = get_min_perm_loss(
            locs_log_probs_all,
            star_params_log_probs_all,
            true_tile_galaxy_bool,
            true_tile_is_on_array,
        )

        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float() + counter_loss + star_params_loss
        loss = loss_vec.mean()

        return loss, counter_loss, locs_loss, star_params_loss

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        return Adam(self.image_encoder.parameters(), **self.optimizer_params)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step (pytorch lightning)."""
        batch_size = len(batch["n_sources"])
        loss = self._get_loss(batch)[0]
        self.log("train/loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        batch_size = len(batch["images"])
        (
            detection_loss,
            counter_loss,
            locs_loss,
            star_params_loss,
        ) = self._get_loss(batch)

        # log all losses
        self.log("val/loss", detection_loss, batch_size=batch_size)
        self.log("val/counter_loss", counter_loss.mean(), batch_size=batch_size)
        self.log("val/locs_loss", locs_loss.mean(), batch_size=batch_size)
        self.log("val/star_params_loss", star_params_loss.mean(), batch_size=batch_size)

        # get metrics and log on full image parameters.
        true_params, est_params, _ = self._get_full_params(batch)
        metrics = self.val_detection_metrics(true_params, est_params)
        self.log("val/precision", metrics["precision"], batch_size=batch_size)
        self.log("val/recall", metrics["recall"], batch_size=batch_size)
        self.log("val/f1", metrics["f1"], batch_size=batch_size)
        self.log("val/avg_distance", metrics["avg_distance"], batch_size=batch_size)
        return batch

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method."""
        if self.current_epoch > 0 and self.image_decoder.n_bands == 1:
            self._make_plots(outputs[-1], kind="validation")

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        true_params, est_params, _ = self._get_full_params(batch)
        metrics = self.test_detection_metrics(true_params, est_params)
        batch_size = len(batch["images"])
        self.log("precision", metrics["precision"], batch_size=batch_size)
        self.log("recall", metrics["recall"], batch_size=batch_size)
        self.log("f1", metrics["f1"], batch_size=batch_size)
        self.log("avg_distance", metrics["avg_distance"], batch_size=batch_size)

        return batch

    def _get_full_params(self, batch):
        # true
        exclude = {"images", "slen", "background"}
        slen = int(batch["slen"].unique().item())
        true_tile_params = {k: v for k, v in batch.items() if k not in exclude}
        true_params = get_full_params_from_tiles(true_tile_params, self.image_encoder.tile_slen)

        # estimate
        tile_estimate = self.tile_map_estimate(batch)
        est_params = get_full_params_from_tiles(tile_estimate, self.image_encoder.tile_slen)
        return true_params, est_params, slen

    # pylint: disable=too-many-statements
    def _make_plots(self, batch, kind="validation", n_samples=16):
        # add some images to tensorboard for validating location/counts.
        # 'batch' is a batch from simulated dataset (all params are tiled)

        assert n_samples ** (0.5) % 1 == 0
        if n_samples > len(batch["n_sources"]):  # do nothing if low on samples.
            return
        nrows = int(n_samples ** 0.5)  # for figure

        true_params, est_params, slen = self._get_full_params(batch)

        # setup figure and axes.
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(n_samples):
            plot_image_and_locs(
                i,
                fig,
                axes[i],
                batch["images"],
                slen,
                true_params,
                estimate=est_params,
                labels=None if i > 0 else ("t. gal", "p. source", "t. star"),
                annotate_axis=True,
                add_borders=True,
            )

        fig.tight_layout()
        if self.logger:
            if kind == "validation":
                title = f"Epoch:{self.current_epoch}/Validation Images"
                self.logger.experiment.add_figure(title, fig)
            elif kind == "testing":
                self.logger.experiment.add_figure("Test Images", fig)
            else:
                raise NotImplementedError()
        plt.close(fig)
