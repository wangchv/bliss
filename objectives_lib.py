import torch
import numpy as np
from hungarian_alg import linear_sum_assignment

from torch.distributions import normal

from star_datasets_lib import get_is_on_from_n_stars

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def isnan(x):
    return x != x


#############################
# functions to get loss for training the counter
############################
def get_one_hot_encoding_from_int(z, n_classes):
    z = z.long()

    assert len(torch.unique(z)) <= n_classes

    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot

def get_categorical_loss(log_probs, true_n_stars):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == len(true_n_stars)
    max_detections = log_probs.shape[1]

    return torch.sum(
        -log_probs * \
        get_one_hot_encoding_from_int(true_n_stars,
                                        max_detections), dim = 1)

def eval_star_counter_loss(star_counter, train_loader,
                            optimizer = None, train = True):

    avg_loss = 0.0
    max_detections = torch.Tensor([star_counter.max_detections])

    for _, data in enumerate(train_loader):
        images = data['image'].to(device)
        true_n_stars = data['n_stars'].to(device)

        if train:
            star_counter.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            star_counter.eval()

        # evaluate log q
        log_probs = star_counter(images)
        loss = get_categorical_loss(log_probs, true_n_stars).mean()

        assert not isnan(loss)

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss * images.shape[0] / len(train_loader.dataset)

    return avg_loss

#############################
# functions to get loss for training the encoder
############################
def _logit(x, tol = 1e-8):
    return torch.log(x + tol) - torch.log(1 - x + tol)

def eval_normal_logprob(x, mu, log_var):
    return - 0.5 * log_var - 0.5 * (x - mu)**2 / torch.exp(log_var)

def eval_logitnormal_logprob(x, mu, log_var):
    logit_x = _logit(x)
    return eval_normal_logprob(logit_x, mu, log_var)

def eval_lognormal_logprob(x, mu, log_var):
    log_x = torch.log(x)
    return eval_normal_logprob(log_x, mu, log_var)

def run_hungarian_alg(log_probs_all, n_stars):
    # log_probs_all should be a tensor of size
    # (batchsize x estimated_param x true param)
    # giving for each N, the log prob of the estimated parameter
    # against the target parameter

    batchsize = log_probs_all.shape[0]
    max_detections = log_probs_all.shape[1]
    perm = np.zeros((batchsize, max_detections))

    for i in range(batchsize):
        n_stars_i = int(n_stars[i])
        if(n_stars_i > 0):
            col_indx = linear_sum_assignment(\
                                    -log_probs_all[i, 0:n_stars_i, 0:n_stars_i])

            perm[i, :] = torch.cat((col_indx,
                torch.LongTensor([i for i in range(n_stars_i, max_detections)])))
        else:
            perm[i, :] = torch.LongTensor([i for i in range(max_detections)])

    return perm

def _permute_losses_mat(losses_mat, perm):
    batchsize = losses_mat.shape[0]
    max_stars = losses_mat.shape[1]

    assert losses_mat.shape[2] == max_stars
    assert perm.shape[0] == batchsize
    assert perm.shape[1] == max_stars

    mat_perm = torch.zeros((batchsize, max_stars)).to(device)
    seq_tensor = torch.LongTensor([i for i in range(batchsize)])

    for i in range(max_stars):
        i_tensor = torch.LongTensor([i for j in range(batchsize)])
        mat_perm[:, i] = losses_mat[seq_tensor, i_tensor, perm[:, i]]

    return mat_perm

def get_locs_logprob_all_combs(true_locs, logit_loc_mean, logit_loc_log_var):
    batchsize = true_locs.shape[0]
    max_stars = true_locs.shape[1]

    # get losses for locations
    _logit_loc_mean = logit_loc_mean.view(batchsize, max_stars, 1, 2)
    _logit_loc_log_var = logit_loc_log_var.view(batchsize, max_stars, 1, 2)
    _true_locs = true_locs.view(batchsize, 1, max_stars, 2)

    # this is batchsize x (n_stars x n_stars)
    # the log prob for each mean x observed location
    locs_log_probs_all = eval_logitnormal_logprob(_true_locs,
                            _logit_loc_mean, _logit_loc_log_var).sum(dim = 3)

    return locs_log_probs_all

def get_fluxes_logprob_all_combs(true_fluxes, log_flux_mean, log_flux_log_var):
    batchsize = true_fluxes.shape[0]
    max_stars = true_fluxes.shape[1]

    _log_flux_mean = log_flux_mean.view(batchsize, max_stars, 1)
    _log_flux_log_var = log_flux_log_var.view(batchsize, max_stars, 1)
    _true_fluxes = true_fluxes.view(batchsize, 1, max_stars)

    # this is batchsize x (n_stars x n_stars)
    # the log prob for each mean x observed flux
    flux_log_probs_all = eval_lognormal_logprob(_true_fluxes,
                                _log_flux_mean, _log_flux_log_var)
    assert list(flux_log_probs_all.shape) == [batchsize, max_stars, max_stars]

    flux_log_probs_all = eval_lognormal_logprob(_true_fluxes,
                                _log_flux_mean, _log_flux_log_var)

    return flux_log_probs_all

def get_encoder_loss(star_encoder, images, true_locs,
                        true_fluxes, true_n_stars):

    # get variational parameters
    logit_loc_mean, logit_loc_log_var, \
            log_flux_mean, log_flux_log_var = star_encoder(images, true_n_stars)

    # get losses for all estimates stars against all true stars

    # this is batchsize x (n_stars x n_stars)
    # the log prob for each mean x observed location
    locs_log_probs_all = \
        get_locs_logprob_all_combs(true_locs, logit_loc_mean, logit_loc_log_var)

    flux_log_probs_all = get_fluxes_logprob_all_combs(true_fluxes, \
                                log_flux_mean, log_flux_log_var)

    # for my sanity
    batchsize = images.shape[0]
    max_stars = true_locs.shape[1]
    assert list(locs_log_probs_all.shape) == [batchsize, max_stars, max_stars]
    assert list(flux_log_probs_all.shape) == [batchsize, max_stars, max_stars]

    # get permutation
    perm = run_hungarian_alg(locs_log_probs_all, true_n_stars)

    # only count those stars that are on
    is_on = get_is_on_from_n_stars(true_n_stars, max_stars)

    # get losses
    locs_loss = -(_permute_losses_mat(locs_log_probs_all, perm) * is_on).sum(dim = 1)
    fluxes_loss = -(_permute_losses_mat(flux_log_probs_all, perm) * is_on).sum(dim = 1)


    loss = (locs_loss + fluxes_loss).mean()

    return loss, locs_loss, fluxes_loss, perm

def eval_star_encoder_loss(star_encoder, train_loader,
                optimizer = None, train = False):

    avg_loss = 0.0

    for _, data in enumerate(train_loader):
        true_fluxes = data['fluxes'].to(device)
        true_locs = data['locs'].to(device)
        true_n_stars = data['n_stars'].to(device)
        images = data['image'].to(device)

        if train:
            star_encoder.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            star_encoder.eval()

        # evaluate log q
        loss = get_encoder_loss(star_encoder, images, true_locs,
                                true_fluxes, true_n_stars)[0]

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss * images.shape[0] / len(train_loader.dataset)

    return avg_loss
