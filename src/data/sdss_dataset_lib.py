import pathlib
import os

import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.utils.data import Dataset
import fitsio

from astropy.io import fits
from astropy.wcs import WCS


def _get_mgrid2(slen0, slen1):
    offset0 = (slen0 - 1) / 2
    offset1 = (slen1 - 1) / 2
    x, y = np.mgrid[-offset0:(offset0 + 1), -offset1:(offset1 + 1)]
    # return torch.Tensor(np.dstack((x, y))) / offset
    return torch.Tensor(np.dstack((y, x))) / torch.Tensor([[[offset1, offset0]]])

class SloanDigitalSkySurvey(Dataset):

    # this is adapted from
    # https://github.com/jeff-regier/celeste_net/blob/935fbaa96d8da01dd7931600dee059bf6dd11292/datasets.py#L10
    # to run on a specified run, camcol, field, and band
    # returns one 1 x 1489 x 2048 image
    def __init__(self, sdssdir = '../sdss_stage_dir/',
                 run = 3900, camcol = 6, field = 269, bands = [2]):

        super(SloanDigitalSkySurvey, self).__init__()
        self.sdss_path = pathlib.Path(sdssdir)

        self.rcfgs = []

        self.bands = bands

        # meta data for the run + camcol
        pf_file = "photoField-{:06d}-{:d}.fits".format(run, camcol)
        camcol_path = self.sdss_path.joinpath(str(run), str(camcol))
        pf_path = camcol_path.joinpath(pf_file)

        pf_fits = fitsio.read(pf_path)

        fieldnums = pf_fits["FIELD"]
        fieldgains = pf_fits["GAIN"]

        # get desired field
        for i in range(len(fieldnums)):
            _field = fieldnums[i]
            gain = fieldgains[i]
            if _field == field:
                self.rcfgs.append((run, camcol, field, gain))

        self.items = [None] * len(self.rcfgs)

    def __len__(self):
        return len(self.rcfgs)

    def __getitem__(self, idx):
        if not self.items[idx]:
            self.items[idx] = self.get_from_disk(idx)
        return self.items[idx]

    def get_from_disk(self, idx):
        run, camcol, field, gain = self.rcfgs[idx]

        camcol_dir = self.sdss_path.joinpath(str(run), str(camcol))
        field_dir = camcol_dir.joinpath(str(field))

        image_list = []
        background_list = []
        nelec_per_nmgy_list = []
        calibration_list = []
        gain_list = []

        cache_path = field_dir.joinpath("cache.pkl")
        # if cache_path.exists():
        #     print('loading cached sdss image from ', cache_path)
        #     return pickle.load(cache_path.open("rb"))

        for b, bl in enumerate("ugriz"):
            if not(b in self.bands):
                continue

            frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(bl, run, camcol, field)
            frame_path = str(field_dir.joinpath(frame_name))
            print("loading sdss image from", frame_path)
            frame = fitsio.FITS(frame_path)

            calibration = frame[1].read()
            nelec_per_nmgy = gain[b] / calibration

            sky_small, = frame[2]["ALLSKY"].read()
            sky_x, = frame[2]["XINTERP"].read()
            sky_y, = frame[2]["YINTERP"].read()

            small_rows = np.mgrid[0:sky_small.shape[0]]
            small_cols = np.mgrid[0:sky_small.shape[1]]
            sky_interp = RegularGridInterpolator((small_rows, small_cols), sky_small, method="nearest")

            sky_y = sky_y.clip(0, sky_small.shape[0] - 1)
            sky_x = sky_x.clip(0, sky_small.shape[1] - 1)
            large_points = np.stack(np.meshgrid(sky_y, sky_x)).transpose()
            large_sky = sky_interp(large_points)
            large_sky_nelec = large_sky * gain[b]

            pixels_ss_nmgy = frame[0].read()
            pixels_ss_nelec = pixels_ss_nmgy * nelec_per_nmgy
            pixels_nelec = pixels_ss_nelec + large_sky_nelec

            image_list.append(pixels_nelec)
            background_list.append(large_sky_nelec)

            gain_list.append(gain[b])
            nelec_per_nmgy_list.append(nelec_per_nmgy)
            calibration_list.append(calibration)

            frame.close()

        ret = {'image': np.stack(image_list),
               'background': np.stack(background_list),
               'nelec_per_nmgy': np.stack(nelec_per_nmgy_list),
               'gain': np.stack(gain_list),
               'calibration': np.stack(calibration_list)}
        pickle.dump(ret, field_dir.joinpath("cache.pkl").open("wb+"))

        return ret

# def _tile_image(image, subimage_slen, return_tile_coords = False):
#     # breaks up a large image into smaller patches,
#     # of size subimage_slen x subimage_slen
#     # NOTE: input and output are torch tensors, not numpy arrays
#     #       (need the unfold command from torch)
#
#     assert len(image.shape) == 2
#
#     image_xlen = image.shape[0]
#     image_ylen = image.shape[1]
#
#     assert (image.shape[0] % subimage_slen) == 0
#     assert (image.shape[1] % subimage_slen) == 0
#
#     image_unfold = image.unfold(0, subimage_slen, subimage_slen).unfold(1, subimage_slen, subimage_slen)
#
#     batchsize = int(image.shape[0] * image.shape[1] / subimage_slen**2)
#
#     if return_tile_coords:
#         num_y_tiles = int(image.shape[1] / subimage_slen)
#         num_x_tiles = int(image.shape[0] / subimage_slen)
#         tile_coords = torch.LongTensor([[i * subimage_slen, j * subimage_slen] \
#                                         for i in range(num_x_tiles) for j in range(num_y_tiles)])
#
#         return image_unfold.contiguous().view(batchsize, 1, subimage_slen, subimage_slen), tile_coords
#     else:
#         return image_unfold.contiguous().view(batchsize, 1, subimage_slen, subimage_slen)

def convert_mag_to_nmgy(mag):
    return 10**((22.5 - mag) / 2.5)

def convert_nmgy_to_mag(nmgy):
    return 22.5 - 2.5 * torch.log10(nmgy)

class SDSSHubbleData(Dataset):

    def __init__(self, sdssdir = '../../celeste_net/sdss_stage_dir/',
                        hubble_cat_file = '../hubble_data/NCG7089/' + \
                                         'hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt.txt',
                        slen = 100,
                        run = 2583,
                        camcol = 2,
                        field = 136,
                        bands = [2],
                        x0 = 630,
                        x1 = 310,
                        fudge_conversion = 1.0,
                        align_bands = True):

        super(SDSSHubbleData, self).__init__()

        assert os.path.exists(sdssdir)

        self.slen = slen
        self.x0 = x0
        self.x1 = x1

        # get sdss data
        self.run = run
        self.camcol = camcol
        self.field = field

        # must use at least the r band
        assert 2 in bands
        self.bands = np.array(bands)

        self.which_r = int(np.argwhere(self.bands == 2))
        if len(bands) > 1:
            self.which_other = int(np.argwhere(self.bands != 2))

        # only handles two bands at the moment
        assert len(bands) <= 2

        # get sdss data
        self.sdss_dir = sdssdir
        self.sdss_data = SloanDigitalSkySurvey(self.sdss_dir,
                                           run = run,
                                           camcol = camcol,
                                           field = field,
                                           bands = bands)

        self.sdss_path = pathlib.Path(self.sdss_dir)

        # save PSF filename; dont actually need to load it though
        # self.psf_file = "psField-{:06d}-{:d}-{:04d}.fit".format(run, camcol, field)
        # self.psf_file = self.sdss_path.joinpath(str(run), str(camcol), \
        #                                     str(field), self.psf_file)
        # print('loading psf from ', self.psf_file)
        # self.psf_full = sdss_psf.psf_at_points(0, 0, psf_fit_file = self.psf_file)
        # self.psf = _trim_psf(self.psf_full, slen)

        # the full SDSS image, ~1500 x 2000 pixels
        self.sdss_image_full = torch.Tensor(self.sdss_data[0]['image'])
        self.sdss_background_full = torch.Tensor(self.sdss_data[0]['background'])

        # load hubble data
        print('loading hubble data from ', hubble_cat_file)
        HTcat = np.loadtxt(hubble_cat_file, skiprows=True)

        # hubble magnitude
        self.hubble_rmag = HTcat[:,9]
        # right ascension and declination
        self.hubble_ra = HTcat[:,21]
        self.hubble_dc = HTcat[:,22]

        # color information
        self.hubble_color = HTcat[:,9] - HTcat[:,10]

        # convert hubble r.a and declination to pixel coordinates
        # (0, 0) is top left of self.sdss_image_full
        frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format('r', run, camcol, field)
        field_dir = pathlib.Path(sdssdir).joinpath(str(run), str(camcol), str(field))
        frame_path = str(field_dir.joinpath(frame_name))
        print('getting sdss coordinates from: ', frame_path)
        hdulist = fits.open(str(frame_path))
        self.wcs = WCS(hdulist['primary'].header)
        # NOTE: pix_coordinates are (column x row), i.e. pix_coord[0] corresponds to a column
        pix_coordinates = \
            self.wcs.wcs_world2pix(self.hubble_ra, self.hubble_dc, 0, ra_dec_order = True)

        self.locs_full_x0 = pix_coordinates[1] # the row of pixel
        self.locs_full_x1 = pix_coordinates[0] # the column of pixel

        if (len(self.bands) > 1) and align_bands:
            self._align_images()

        # convert hubble magnitude to n_electron count
        # only take r band
        self.nelec_per_nmgy_full = self.sdss_data[0]['nelec_per_nmgy'][self.bands == 2].squeeze()
        self.nelec_per_nmgy = self.nelec_per_nmgy_full[x1:(x1 + slen)]
        which_cols = np.floor(self.locs_full_x1 / len(self.nelec_per_nmgy_full)).astype(int)
        hubble_nmgy = convert_mag_to_nmgy(self.hubble_rmag)

        self.fudge_conversion = fudge_conversion
        self.fluxes_full = hubble_nmgy * self.nelec_per_nmgy[which_cols] * self.fudge_conversion # / self.psf_max

        assert len(self.fluxes_full) == len(self.locs_full_x0)
        assert len(self.fluxes_full) == len(self.locs_full_x1)

        # get parameters in our subimage
        which_locs = (self.locs_full_x0 > x0) & (self.locs_full_x0 < (x0 + self.slen - 1)) & \
                        (self.locs_full_x1 > x1) & (self.locs_full_x1 < (x1 + self.slen - 1))

        self.locs = np.array([self.locs_full_x0[which_locs] - x0,
                                self.locs_full_x1[which_locs] - x1]).transpose()

        r_fluxes = self.fluxes_full[which_locs]

        print('\n returning image at x0 = {}, x1 = {}'.format(x0, x1))
        # just a subset
        self.sdss_image = self.sdss_image_full[:, x0:(x0 + slen), x1:(x1 + slen)]
        self.sdss_background = self.sdss_background_full[:, x0:(x0 + slen), x1:(x1 + slen)]

        # convert to torch.Tensor
        self.locs = torch.Tensor(self.locs) / (self.slen - 1)
        self.r_fluxes = torch.Tensor(r_fluxes)
        self.sdss_image = torch.Tensor(self.sdss_image)
        self.sdss_background = torch.Tensor(self.sdss_background)

        self.hubble_color = torch.Tensor(self.hubble_color[which_locs])

        if len(self.bands) > 1:
            self._estimate_colors()
        else:
            self.fluxes = self.r_fluxes.unsqueeze(1)

    def _align_images(self):
        indx = self.bands[self.which_other]
        other_band = 'ugriz'[indx]
        frame_name = "frame-{}-{:06d}-{:d}-{:04d}.fits".format(other_band, self.run, self.camcol, self.field)
        field_dir = pathlib.Path(self.sdss_dir).joinpath(str(self.run), str(self.camcol), str(self.field))
        frame_path = str(field_dir.joinpath(frame_name))
        print('\n aligning images. \n Getting sdss coordinates from: ', frame_path)
        hdu = fits.open(str(frame_path))
        wcs_other = WCS(hdu['primary'].header)

        # get pixel coords
        pix_coordinates_other = wcs_other.wcs_world2pix(self.hubble_ra,
                                        self.hubble_dc, 0, ra_dec_order = True)

        self.shift_x0 = np.median(self.locs_full_x0 - pix_coordinates_other[1])
        self.shift_x1 = np.median(self.locs_full_x1 - pix_coordinates_other[0])

        grid = _get_mgrid2(self.sdss_image_full.shape[-2],
                            self.sdss_image_full.shape[-1]).unsqueeze(0) - \
                torch.Tensor([[[[self.shift_x1 / (self.sdss_image_full.shape[-1] - 1),
                                self.shift_x0 / (self.sdss_image_full.shape[-2] - 1)]]]]) * 2

        self.sdss_image_full[self.which_other] = \
            torch.nn.functional.grid_sample(self.sdss_image_full[self.which_other].unsqueeze(0).unsqueeze(0),
                                            grid, align_corners=True).squeeze()

    def _estimate_colors(self):
        locs_indx = torch.round(self.locs * (self.slen - 1)).type(torch.long)

        # pixels in the r-band
        image_r_pixels = self.sdss_image[self.which_r][locs_indx[:, 0], locs_indx[:, 1]]

        # pixels in the other band
        image_other_pixels = self.sdss_image[self.which_other][locs_indx[:, 0], locs_indx[:, 1]]

        # flux ratio
        flux_ratio = (image_other_pixels / image_r_pixels)

        # set fluxes
        self.fluxes = torch.zeros(len(self.r_fluxes), len(self.bands))

        self.fluxes[:, int(np.argwhere(self.bands == 2))] = self.r_fluxes
        self.fluxes[:, int(np.argwhere(self.bands != 2))] = (self.r_fluxes * flux_ratio)

    # only one image in a dataset right now;
    # __len__ and __getitem__ are unnecessary atm ...




    #     # break up the full sdss image into tiles, and create a batch
    #     self.images, self.tile_coords = \
    #         _tile_image(torch.Tensor(self.sdss_image_full), slen, return_tile_coords = True)
    #     self.images = self.images.numpy()
    #     self.tile_coords = self.tile_coords.numpy()
    #     self.backgrounds = _tile_image(torch.Tensor(self.sdss_background_full), slen).numpy()
    #
    #     # get counts of stars in each tile
    #     x = np.histogram2d(self.locs_x1[self.which_bright],
    #                     self.locs_x0[self.which_bright],
    #                   bins = (np.arange(-0.5, self.sdss_image_full.shape[1] + slen - 0.5, step = slen),
    #                           np.arange(-0.5, self.sdss_image_full.shape[0] + slen - 0.5, step = slen)));
    #     self.counts_mat = np.transpose(x[0])
    #
    #     # keep only those images with no more than max_detection
    #     # number of stars
    #     which_keep = (self.counts_mat.flatten() > 0) & (self.counts_mat.flatten() < max_detections)
    #
    #     self.images = self.images[which_keep]
    #     self.backgrounds = self.backgrounds[which_keep]
    #     self.tile_coords = self.tile_coords[which_keep]
    #     self.n_stars = self.counts_mat.flatten()[which_keep]
    #
    # def __len__(self):
    #     return self.images.shape[0]
    #
    # def __getitem__(self, idx):
    #
    #     x0 = self.tile_coords[idx, 0]
    #     x1 = self.tile_coords[idx, 1]
    #
    #     locs, fluxes, locs_border, fluxes_border, locs_dim, fluxes_dim = \
    #         self._get_hubble_params_in_patch(x0, x1, self.slen)
    #
    #     assert self.n_stars[idx] == (locs.shape[0] + locs_border.shape[0])
    #
    #     n_stars = locs.shape[0] # int(self.n_stars[idx])
    #     # have a 0.5 pixel border around each image; discard those
    #     #  outside this border
    #     # which_keep = (locs[:, 0] > 0) & (locs[:, 0] < 1) & \
    #     #                 (locs[:, 1] > 0) & (locs[:, 1] < 1)
    #
    #     # locs = locs[which_keep, :]
    #     # fluxes = fluxes[which_keep]
    #     #
    #     # n_stars = sum(which_keep)
    #
    #     if(len(fluxes) < self.max_detections):
    #         # append "empty" stars
    #         n_null = self.max_detections - len(fluxes)
    #         locs = np.concatenate((locs, np.zeros((n_null, 2))), 0)
    #         fluxes = np.concatenate((fluxes, np.zeros(n_null)))
    #
    #         n_null_border = self.max_detections - len(fluxes_border)
    #         locs_border = np.concatenate((locs_border, np.zeros((n_null_border, 2))), 0)
    #         fluxes_border = np.concatenate((fluxes_border, np.zeros(n_null_border)))
    #
    #     return {'image': self.images[idx],
    #             'background': self.backgrounds[idx],
    #             'locs': locs,
    #             'fluxes': fluxes,
    #             'locs_border': locs_border,
    #             'fluxes_border': fluxes_border,
    #             # TODO: locs_dim and fluxes_dim are currently of varying size;
    #             # how will if affect the dataloader?
    #             # 'locs_dim': locs_dim,
    #             # 'fluxes_dim': fluxes_dim,
    #             'n_stars': n_stars}
    #
    # def _get_hubble_params_in_patch(self, x0, x1, slen, return_dim_stars = False):
    #     # x0 and x1 correspond to matrix indices
    #     # x0 is row of pixel, x1 is column of pixel
    #
    #     # get all locations in the square
    #     which_pixels = (self.locs_x0 > x0 - 0.5) & \
    #                             (self.locs_x0 < (x0 + slen - 0.5)) & \
    #                      (self.locs_x1 > x1 - 0.5) & \
    #                             (self.locs_x1 < (x1 + slen - 0.5))
    #
    #     locs = np.transpose(np.array([(self.locs_x0[which_pixels * self.which_bright] - x0) / (slen - 1),
    #                                   (self.locs_x1[which_pixels * self.which_bright] - x1) / (slen - 1)]))
    #
    #     fluxes = self.fluxes[which_pixels * self.which_bright]
    #
    #     # those locs on the border; we don't detect these
    #     which_border = (locs[:, 0] > 1) | (locs[:, 0] < 0) | \
    #                         (locs[:, 1] > 1) | (locs[:, 1] < 0)
    #     locs_border = locs[which_border]
    #     locs = locs[(1 - which_border).astype(bool)]
    #
    #     fluxes_border = fluxes[which_border]
    #     fluxes = fluxes[(1 - which_border).astype(bool)]
    #
    #     if return_dim_stars:
    #         which_dim = (1 - self.which_bright).astype(bool)
    #         locs_dim = np.transpose(np.array([(self.locs_x0[which_pixels * which_dim] - x0) / (slen - 1),
    #                                       (self.locs_x1[which_pixels * which_dim] - x1) / (slen - 1)]))
    #
    #         fluxes_dim = self.fluxes[which_pixels * which_dim]
    #     else:
    #         locs_dim = 0
    #         fluxes_dim = 0
    #
    #     return locs, fluxes, locs_border, fluxes_border, locs_dim, fluxes_dim
    #
    # def plot_image_patch(self, x0, x1, slen, flip_y = True, plot_dim_stars = False):
    #
    #     image = self.sdss_image_full[x0:(x0 + slen), x1:(x1 + slen)]
    #
    #     if flip_y:
    #         # to match with portillo's plots
    #         image = image[slen::-1]
    #
    #     plt.matshow(image)
    #
    #     # get hubble parameters
    #     locs, fluxes, locs_border, fluxes_border, locs_dim, fluxes_dim = \
    #         self._get_hubble_params_in_patch(x0, x1, slen,
    #                                         return_dim_stars = plot_dim_stars)
    #
    #     x_locs = locs[:, 1] * (slen - 1)
    #     y_locs = locs[:, 0] * (slen - 1)
    #
    #     if flip_y:
    #         y_locs = (slen - 1) - y_locs
    #
    #     plt.plot(x_locs, y_locs, 'o', color = 'b')
    #
    #     if plot_dim_stars:
    #         # plot dim stars, too
    #         x_locs_dim = locs_dim[:, 1] * (slen - 1)
    #         y_locs_dim = locs_dim[:, 0] * (slen - 1)
    #
    #         if flip_y:
    #             y_locs_dim = (slen - 1) - y_locs_dim
    #
    #         plt.plot(x_locs_dim, y_locs_dim, '.', color = 'g')