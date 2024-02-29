import numpy as np
import pandas as pd
import math

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