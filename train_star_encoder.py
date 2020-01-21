import numpy as np

import torch
import torch.optim as optim

import fitsio

import sdss_psf
import simulated_datasets_lib
import starnet_vae_lib
import inv_kl_objective_lib as objectives_lib

import time

import json

from torch.distributions import normal

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device: ', device)

print('torch version: ', torch.__version__)

# set seed
np.random.seed(65765)
_ = torch.manual_seed(3453453)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# data parameters
with open('./data/default_star_parameters.json', 'r') as fp:
    data_params = json.load(fp)

print(data_params)

# load psf
psf_dir = './data/'
psf_r = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-r.fits')[0].read()
psf_i = fitsio.FITS(psf_dir + 'sdss-002583-2-0136-psf-i.fits')[0].read()
psf_og = np.array([psf_r])

# sky intensity: for the r and i band
# sky_intensity = torch.Tensor([686., 1123.]).to(device)
sky_intensity = torch.Tensor([686.]).to(device)

# draw data
print('generating data: ')
n_images = 200
t0 = time.time()
star_dataset = \
    simulated_datasets_lib.load_dataset_from_params(psf_og,
                            data_params,
                            sky_intensity = sky_intensity,
                            n_images = n_images,
                            transpose_psf = False,
                            add_noise = True)

print('data generation time: {:.3f}secs'.format(time.time() - t0))
# get loader
batchsize = 20

loader = torch.utils.data.DataLoader(
                 dataset=star_dataset,
                 batch_size=batchsize,
                 shuffle=True)

# define VAE
star_encoder = starnet_vae_lib.StarEncoder(full_slen = data_params['slen'],
                                           stamp_slen = 7,
                                           step = 2,
                                           edge_padding = 2,
                                           n_bands = psf_og.shape[0],
                                           max_detections = 2)

star_encoder.to(device)

# define optimizer
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = optim.Adam([
                    {'params': star_encoder.parameters(),
                    'lr': learning_rate}],
                    weight_decay = weight_decay)

n_epochs = 201
print_every = 20
print('training')

test_losses = np.zeros((4, n_epochs // print_every + 1))

avg_loss_old = 1e16
for epoch in range(n_epochs):
    t0 = time.time()

    avg_loss, counter_loss, locs_loss, fluxes_loss \
        = objectives_lib.eval_star_encoder_loss(star_encoder, loader,
                                                    optimizer, train = True)

    elapsed = time.time() - t0
    print('[{}] loss: {:0.4f}; counter loss: {:0.4f}; locs loss: {:0.4f}; fluxes loss: {:0.4f} \t[{:.1f} seconds]'.format(\
                    epoch, avg_loss, counter_loss, locs_loss, fluxes_loss, elapsed))

    # draw fresh data
    loader.dataset.set_params_and_images()

    if (epoch % print_every) == 0:
        _ = \
            objectives_lib.eval_star_encoder_loss(star_encoder,
                                            loader, train = True)

        loader.dataset.set_params_and_images()
        test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss = \
            objectives_lib.eval_star_encoder_loss(star_encoder,
                                            loader, train = False)

        print('**** test loss: {:.3f}; counter loss: {:.3f}; locs loss: {:.3f}; fluxes loss: {:.3f} ****'.format(\
            test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss))

        outfile = './fits/results_2020-01-21/starnet_r'
        print("writing the encoder parameters to " + outfile)
        torch.save(star_encoder.state_dict(), outfile)

        test_losses[:, epoch // print_every] = np.array([test_loss, test_counter_loss, test_locs_loss, test_fluxes_loss])
        np.savetxt('./fits/results_2020-01-21/test_losses-starnet_r', test_losses)

print('done')
