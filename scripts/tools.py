import numpy as np
import pandas as pd
import math
from PIL import Image, ImageOps, ImageMath

import torch
import matplotlib.pyplot as plt
from astropy.utils.data import download_file

# Mpol utilities
from mpol.__init__ import zenodo_record
from mpol import coordinates, gridding, fourier, losses, precomposed, utils
from mpol.images import ImageCube


####################################################################################################
def make_power_spectrum(visibility_file, n_bins):
    '''------------------------------------------------------------------
    Create a power spectrum from the visibilities in the input .npz file.
    Save the power spectrum data and alsop plot it.
    ------------------------------------------------------------------'''

    # load the mock visibilities from the .npz file
    d = np.load(visibility_file)
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data = d["data"]
    data_re = np.real(data)
    data_im = np.imag(data)

    # calculate the uv distance (baseline separations in klambda, calculated as sqrt(u*u+v*v))
    uvdist = np.hypot(uu, vv)

    # get the number of visibilities
    nvis = data.shape[0]

    print(f'Loaded visibilities from {visibility_file}.')
    print(f'The dataset has {nvis} visibilities.\n')

    re_and_im_df = pd.DataFrame({'uvdist': uvdist, 'V_re': data_re, 'V_im': data_im})
    re_and_im_df.sort_values(by='uvdist', inplace=True, ignore_index=True)

    # bin the visibilities by uv distance
    re_and_im_df['uvdist_bins'], bin_edges = pd.cut(re_and_im_df['uvdist'], bins=n_bins, retbins=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # square the averaged real and imaginary parts in each bin and sum the squares to get the power
    bin_averaged_V_re = re_and_im_df.groupby('uvdist_bins')['V_re'].mean()
    bin_averaged_V_im = re_and_im_df.groupby('uvdist_bins')['V_im'].mean()
    bin_averaged_power = bin_averaged_V_re**2 + bin_averaged_V_im**2

    # calclulate the scales corresponding to the uv distances in arcseconds
    scales = (1/(bin_centers*1000))*(180*60*60/math.pi)

    # make a dataframe of the power spectrum and save it to a .csv file
    power_spectrum_df = pd.DataFrame({'uvdist_klambda': bin_centers, 'scales_arcsec':scales, 'V_re': bin_averaged_V_re.values, 'V_im':bin_averaged_V_im.values, 'power': bin_averaged_power.values})
    power_spectrum_df.to_csv('RML_loop_outputs/power_spectrum.csv', sep='\t', header=True, index_label='bin')
    print(f'Power spectrum data saved to: RML_loop_outputs/power_spectrum.csv\n')

    return power_spectrum_df
####################################################################################################

####################################################################################################
def make_visibilities_from_image(image_file, cell_size):

    # load the image (of the sky brightness distribution model) using the pillow library
    #im_raw = Image.open('../data/star_images/model_star_delta.jpeg')
    #im_raw = Image.open('../data/star_images/RScl.jpg')
    im_raw = Image.open(image_file)

    # convert the image to single axis greyscale from RGB, etc.
    im_grey = ImageOps.grayscale(im_raw)
    print(im_grey.mode)

    # get image dimensions
    xsize, ysize = im_grey.size
    print(xsize, ysize)

    # Additional tasks: optional
    ####################################################################################################
    xhann = np.hanning(xsize)
    yhann = np.hanning(ysize)
    # each is already normalized to a max of 1
    # so hann is also normed to a max of 1
    # broadcast to 2D
    hann = np.outer(yhann, xhann)

    # now convert the numpy array to a Pillow object
    # scale to 0 - 255 and then convert to uint8
    hann8 = np.uint8(hann * 255)
    im_apod = Image.fromarray(hann8)

    im_res = ImageMath.eval("a * b", a=im_grey, b=im_apod)#.rotate(90)

    max_dim = np.maximum(xsize, ysize)
    im_pad = ImageOps.pad(im_res, (max_dim, max_dim))

    npix = 128
    im_grey = im_pad.resize((npix,npix))
    ####################################################################################################

    # obtain the pixel values of the greyscale image as a numpy array
    im_array = np.array(im_grey)
    im_array

    # convert this array to a float64 type and normalize its max value to 1
    im_array = im_array.astype("float64")
    im_array = im_array/im_array.max()

    # flipping the image to prevent it from being flipped when visualized with imshow(origin='lower')
    im_array = np.flipud(im_array)

    # visualise the image
    plt.imshow(im_array, origin='lower')
    plt.colorbar()
    plt.show()

    # convert single channel image to a 3D imager cube (add a dimension of size 1)
    im_cube = np.expand_dims(im_array, axis=0)
    im_cube

    # choose how big we want our mock sky brightness to be on the sky (setting cell size and npix)
    #cell_size = 0.01 # arcsec
    print(cell_size, "arcsec")

    # calculate the number of pixels per image axis
    npix = im_array.shape[0]
    print(npix)

    # calculate the area per pixel in the image
    pixel_area = cell_size**2 # arcsec
    print(pixel_area, "arcsec^2")

    # calculate the total flux in the original image
    original_flux = np.sum(im_cube * pixel_area)
    print(original_flux, "Jy")

    # scale the image so that the total flux becomes 'new_flux' Jy
    new_flux = 5 # Jy
    im_cube_flux_scaled = im_cube * (new_flux/original_flux)
    scaled_flux = np.sum(im_cube_flux_scaled * pixel_area)
    print(scaled_flux, "Jy")
    print(im_cube_flux_scaled, "Jy")

    # convert the image cube array to a torch tensor
    img_tensor = torch.tensor(im_cube_flux_scaled.copy())
    img_tensor

    # shift the tensor from a “Sky Cube” to a “Packed Cube” as the input to mpol.images.ImageCube() which will be FFT’ed to get the visibilities is a “Packed Cube” object
    img_tensor_packed = utils.sky_cube_to_packed_cube(img_tensor)

    # create an MPol "Image Cube" object
    image = ImageCube.from_image_properties(cell_size=cell_size, npix=npix, nchan=1, cube=img_tensor_packed)

    # double check if the image cube is as expected: check if it has the same scaled flux
    print(np.sum(np.squeeze(utils.packed_cube_to_sky_cube(image()).detach().numpy()) * pixel_area))
    # double check if the image cube is as expected: convert back to numpy array and visualise
    plt.imshow(np.squeeze(utils.packed_cube_to_sky_cube(image()).detach().numpy()), origin="lower")
    plt.colorbar()
    plt.show()

    # obtain a (u,v) distribution (and weights) on which to calculate the visibilities of the image (download from MPol ALMA logo tutorial)

    # download the ALMA logo mock visibility dataset
    fname = download_file(
        f"https://zenodo.org/record/{zenodo_record}/files/logo_cube.noise.npz",
        cache=True,
        show_progress=True,
        pkgname="mpol",
    )

    # select the components for a single channel
    chan = 4

    # extract the (u,v) distribution and weights from the downloaded data
    d = np.load(fname)
    uu = d["uu"][chan]
    vv = d["vv"][chan]
    weight = d["weight"][chan]

    nvis = uu.shape[0]
    print(f"Dataset has {nvis} visibilities")

    print(uu)
    print(vv)
    print(weight)

    max_uv = np.max(np.array([uu,vv]))
    max_cell_size = utils.get_maximum_cell_size(max_uv)
    print("The maximum cell_size that will still Nyquist sample the spatial frequency represented by the maximum u,v value is {:.2f} arcseconds".format(max_cell_size))
    assert cell_size < max_cell_size

    # create the mock visibilities corresponding to the image (having the same shape as the uu, vv, and weight inputs)
    data_noise, data_noiseless = fourier.make_fake_data(image, uu, vv, weight)

    print(data_noise.shape)
    print(data_noiseless.shape)
    print(data_noise)

    # save the mock visibilities to a .npz file
    data = np.squeeze(data_noise)
    #data = np.squeeze(data_noiseless)
    #np.savez("../data/visibilities/mock_visibilities_model_star_delta.npz", uu=uu, vv=vv, weight=weight, data=data)
    np.savez("../data/visibilities/mock_visibilities_RScl.npz", uu=uu, vv=vv, weight=weight, data=data)
####################################################################################################

def calcvals(uvbin, baseline, weight,data):
    fluxmean_real=np.zeros(len(uvbin))
    fluxstd_real=np.zeros(len(uvbin))
    fluxmean_imag=np.zeros(len(uvbin))
    fluxstd_imag=np.zeros(len(uvbin))
    for i in range(len(uvbin)-1):
        uvmax=uvbin[i+1]
        uvmin=uvbin[i]
        data_where = np.logical_and(baseline>uvmin, baseline<=uvmax)        # boolean array to locate the baselines within the bin
        data_where2 = data_where
        # data_where2 = np.logical_and(data_where, mask)                      # remove any flagged visibilities
        weight_n = weight[data_where2]
        data_n = data[data_where2]
        real_n = data_n.real
        imag_n = data_n.imag

        vis_real=0.
        vis_imag=0.
        variance_real = 0.
        variance_imag = 0.
        if (data_where2.sum()>0):
            vis_real = np.average(real_n,weights=weight_n)
            variance_real = np.sqrt(np.average((real_n-vis_real)**2.,weights=weight_n))
            vis_imag = np.average(imag_n,weights=weight_n)
            variance_imag = np.sqrt(np.average((imag_n-vis_imag)**2.,weights=weight_n))
            num1 = data_where.sum()
            variance_real /= (np.sqrt(num1)-1)
            variance_imag /= (np.sqrt(num1)-1)
        fluxmean_real[i]=vis_real
        fluxstd_real[i]=variance_real
        fluxmean_imag[i]=vis_imag
        fluxstd_imag[i]=variance_imag
        print(uvmin, uvmax)

    psd = fluxmean_real**2.+fluxmean_imag**2.
    variance_psd = 2.*fluxmean_real*fluxstd_real + 2.*fluxmean_imag*fluxstd_imag
    return fluxmean_real,fluxstd_real,fluxmean_imag,fluxstd_imag, psd, variance_psd