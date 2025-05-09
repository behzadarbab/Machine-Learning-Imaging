import numpy as np
import pandas as pd
import math

import torch
import matplotlib.pyplot as plt

# Mpol utilities
from mpol import fourier, utils
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
def image_to_mock_visibilities(image, pixel_size_arcsec, array_config_file, outfile, add_noise=False, plot=True):
    """
    Function to convert an image to mock visibilities.
    'image' should be a 2D numpy array, and 'array_config_file' should be a .npz file containing the u, v coordinates and weights.
    '{outfile}.npz' is the name of the output file to which the mock visibilities will be saved.
    'add_noise' is a boolean flag to indicate whether to add noise to the mock visibilities or not.
    """


    ### load the array configuration from the input .npz file
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    # NOTE: Do not read in visibility data from this file even if it exists, only load the baseline positions and weights, i.e., the uu, vv, and weight arrays
    d = np.load(array_config_file)
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    print('Loaded array configuration from', array_config_file, '\n')
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


    ### plot the loaded baseline configuration
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    print('Plotting the loaded baseline configuration...')
    fig, ax = plt.subplots(nrows=1)
    ax.scatter(uu, vv, s=1.5, rasterized=True, linewidths=0.0, c="k") # plot the u,v coordinates
    ax.scatter(-uu, -vv, s=1.5, rasterized=True, linewidths=0.0, c="k")  # and their Hermitian conjugates
    ax.set_xlabel(r"$u$") # [k$\lambda$]? [$\lambda$]?
    ax.set_ylabel(r"$v$")
    ax.set_title("Baseline configuration to be used for generating mock visibilities")
    ax.invert_xaxis()
    plt.show()
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


    ### pre-process the data to formats needed for MPoL modeules
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    print('Processing the image to MPoL/torch formats...\n')
    # convert single channel image to a 3D imager cube (add a dimension of size 1)
    im_cube = np.expand_dims(image, axis=0)
    # choose how big we want our mock sky brightness to be on the sky (setting cell size and npix)
    mod_image_cell_size = pixel_size_arcsec # arcsec
    print('Model image cell size:', mod_image_cell_size, 'arcsec\n')
    npix = image.shape[0] # number of pixels along one side of the image
    # convert the image cube array to a torch tensor
    img_tensor = torch.tensor(im_cube.copy())
    # shift the tensor from a “Sky Cube” to a “Packed Cube” as the input to mpol.images.ImageCube() which will be FFT’ed to get the visibilities is a “Packed Cube” object
    img_tensor_packed = utils.sky_cube_to_packed_cube(img_tensor)
    # create an MPol "Image Cube" object
    mod_image = ImageCube.from_image_properties(cell_size=mod_image_cell_size, npix=npix, nchan=1, cube=img_tensor_packed)
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


    ### plot the processed image to make sure that it is the same as the user-input image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    print('Plotting the processed image to check that it is the same as the original user-input...')
    plt.imshow(np.squeeze(utils.packed_cube_to_sky_cube(mod_image()).detach().numpy()), cmap="gray", origin="lower")
    plt.colorbar()
    plt.show()
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


    ### some checks
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    config_max_uv = np.max(np.array([uu,vv]))
    config_max_cell_size = utils.get_maximum_cell_size(config_max_uv)
    print("The maximum cell_size that will still Nyquist sample the spatial frequency represented by the maximum u,v value is {:.2f} arcseconds".format(config_max_cell_size))
    print("Pixel size in input image is: ", pixel_size_arcsec, '\n')
    assert pixel_size_arcsec < config_max_cell_size
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


    ### create the mock visibilities corresponding to the image (having the same shape as the uu, vv, and weight inputs)
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    print('Creating the mock visibilities from the image...\n')
    mod_image_data_noise, mod_image_data_noiseless = fourier.make_fake_data(mod_image, uu, vv, weight)
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


    ### save the mock visibilities
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    if add_noise:
        # add noise to the mock visibilities
        print('Using mock visibilities with noise\n')
        mod_image_data = np.squeeze(mod_image_data_noise)
    else:
        # save the noiseless mock visibilities
        print('Using mock visibilities without noise\n')
        mod_image_data = np.squeeze(mod_image_data_noiseless)

    # save the mock visibilities to a .npz file
    np.savez(f"{outfile}.npz", uu=uu, vv=vv, weight=weight, data=mod_image_data)
    print(f"Mock visibilities saved to {outfile}.npz\n")
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    if plot:
        ### re-load and plot the mock visibilities from the .npz file to check if they were saved correctly
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
        print(f'Re-loading the mock visibilities from {outfile}.npz file and plotting array configuration...')
        d = np.load(f"{outfile}.npz")
        uu = d["uu"]
        vv = d["vv"]
        weight = d["weight"]
        data = d["data"]
        data_re = np.real(data)
        data_im = np.imag(data)
        # Plot the downloaded (u,v) distribution
        fig, ax = plt.subplots(nrows=1)
        ax.scatter(uu, vv, s=1.5, rasterized=True, linewidths=0.0, c="k") # plot the u,v coordinates
        ax.scatter(-uu, -vv, s=1.5, rasterized=True, linewidths=0.0, c="k")  # and their Hermitian conjugates
        ax.set_xlabel(r"$u$")
        ax.set_ylabel(r"$v$")
        ax.invert_xaxis()
        plt.show()
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


        ### calculate the amplitude and phase of the visibilities and make plots
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
        print('Calculating the amplitude and phase of the visibilities and making plots...')
        amp = np.abs(data)
        phase = np.angle(data)
        # calculate the uv distance (baseline separations in meters, calculated as sqrt(u*u+v*v))
        uvdist = np.hypot(uu, vv)
        fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
        pkw = {"s":1, "rasterized":True, "linewidths":0.0, "c":"k"}
        ax[0].scatter(uvdist, data_re, **pkw)
        ax[0].set_ylabel("Re(V) [Jy]")
        ax[1].scatter(uvdist, data_im, **pkw)
        ax[1].set_ylabel("Im(V) [Jy]")
        ax[2].scatter(uvdist, amp, **pkw)
        ax[2].set_ylabel("amplitude [Jy]")
        ax[3].scatter(uvdist, phase, **pkw)
        ax[3].set_ylabel("phase [radians]")
        ax[3].set_xlabel(r"$uv$dist [k$\lambda$]")
        plt.tight_layout()
        plt.show()
        ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


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