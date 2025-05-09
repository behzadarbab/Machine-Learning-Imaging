import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy.utils.data import download_file
import tempfile
import tarfile
import os


direc = '../data/visibilities/'
fname = direc+'R_Dor.continuumsc_fixvis_ID0.spw1.ms'
#fname = direc+'R_Dor.continuumsc_fixvis_ID1.spw1.ms'
#fname = direc+'R_Dor.continuumsc_fixvis_ID2.spw1.ms'

# query the data
tb.open(fname)
ant1 = tb.getcol("ANTENNA1")  # array of int with shape [nvis]
ant2 = tb.getcol("ANTENNA2")  # array of int with shape [nvis]
uvw = tb.getcol("UVW")  # array of float64 with shape [3, nvis]
weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
data = tb.getcol("DATA")  # array of complex128 with shape [npol, nchan, nvis]
#data = tb.getcol("CORRECTED_DATA")  # array of complex128 with shape [npol, nchan, nvis]
tb.close()

# get the channel information
tb.open(fname + "/SPECTRAL_WINDOW")
chan_freq = tb.getcol("CHAN_FREQ")
num_chan = tb.getcol("NUM_CHAN")
tb.close()
chan_freq = chan_freq.flatten()  # Hz
nchan = len(chan_freq)

if (nchan > 1) and (chan_freq[1] > chan_freq[0]):
    # reverse channels
    chan_freq = chan_freq[::-1]
    data = data[:, ::-1, :]
    flag = flag[:, ::-1, :]

xc = np.where(ant1 != ant2)[0]
data = data[:, :, xc]
flag = flag[:, :, xc]
uvw = uvw[:, xc]
weight = weight[:, xc]

# average the polarizations
data = np.sum(data * weight[:, np.newaxis, :], axis=0) / np.sum(weight, axis=0)
flag = np.any(flag, axis=0)
weight = np.sum(weight, axis=0)

# After this step, ``data`` should be shape ``(nchan, nvis)`` and weights should be shape ``(nvis,)``
print(data.shape)
print(flag.shape)
print(weight.shape)


# when indexed with mask, returns valid visibilities
mask = ~flag

# convert uu and vv to kilolambda
uu, vv, ww = uvw  # unpack into len nvis vectors
# broadcast to the same shape as the data
# stub to broadcast uu,vv, and weights to all channels
broadcast = np.ones((nchan, 1))
uu = uu * broadcast
vv = vv * broadcast
weight = weight * broadcast

# calculate wavelengths in meters
wavelengths = c.value / chan_freq[:, np.newaxis]  # m

# calculate baselines in klambda
uu = 1e-3 * uu / wavelengths  # [klambda]
vv = 1e-3 * vv / wavelengths  # [klambda]

frequencies = chan_freq * 1e-9  # [GHz]

baseline = np.sqrt(uu*uu+vv*vv)

nbins=25
uvbin=np.linspace(baseline.min(),baseline.max(),num=nbins,endpoint=True)
duv = uvbin[1]-uvbin[0]

def calcvals(uvbin, baseline, weight,data):
    fluxmean_real=np.zeros(len(uvbin))
    fluxstd_real=np.zeros(len(uvbin))
    fluxmean_imag=np.zeros(len(uvbin))
    fluxstd_imag=np.zeros(len(uvbin))
    for i in range(len(uvbin)-1):
        uvmax=uvbin[i+1]
        uvmin=uvbin[i]
        data_where = np.logical_and(baseline>uvmin, baseline<=uvmax)        # boolean array to locate the baselines within the bin
        data_where2 = np.logical_and(data_where, mask)                      # remove any flagged visibilities
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

mean_real1,var_real1,mean_imag1,var_imag1,mean_psd1,var_psd1 = calcvals(uvbin, baseline, weight,data)

mas1 = 1/(uvbin+0.5*duv)*180./np.pi*3600
uvd1 = (uvbin+0.5*duv)

direc = '/disk/aop5_2/wouterv/ALMA/2022.1.01071.S/RDor_B7/continuum_fits/'
fname = direc+'R_Dor.continuumsc_fixvis_offpos.ID0.ms'

# query the data
tb.open(fname)
ant1 = tb.getcol("ANTENNA1")  # array of int with shape [nvis]
ant2 = tb.getcol("ANTENNA2")  # array of int with shape [nvis]
uvw = tb.getcol("UVW")  # array of float64 with shape [3, nvis]
weight = tb.getcol("WEIGHT")  # array of float64 with shape [npol, nvis]
flag = tb.getcol("FLAG")  # array of bool with shape [npol, nchan, nvis]
data = tb.getcol("DATA")  # array of complex128 with shape [npol, nchan, nvis]
#data = tb.getcol("CORRECTED_DATA")  # array of complex128 with shape [npol, nchan, nvis]
tb.close()

# get the channel information
tb.open(fname + "/SPECTRAL_WINDOW")
chan_freq = tb.getcol("CHAN_FREQ")
num_chan = tb.getcol("NUM_CHAN")
tb.close()
chan_freq = chan_freq.flatten()  # Hz
nchan = len(chan_freq)

if (nchan > 1) and (chan_freq[1] > chan_freq[0]):
    # reverse channels
    chan_freq = chan_freq[::-1]
    data = data[:, ::-1, :]
    flag = flag[:, ::-1, :]

xc = np.where(ant1 != ant2)[0]
data = data[:, :, xc]
flag = flag[:, :, xc]
uvw = uvw[:, xc]
weight = weight[:, xc]

# average the polarizations
data = np.sum(data * weight[:, np.newaxis, :], axis=0) / np.sum(weight, axis=0)
flag = np.any(flag, axis=0)
weight = np.sum(weight, axis=0)

# After this step, ``data`` should be shape ``(nchan, nvis)`` and weights should be shape ``(nvis,)``
print(data.shape)
print(flag.shape)
print(weight.shape)


# when indexed with mask, returns valid visibilities
mask = ~flag

# convert uu and vv to kilolambda
uu, vv, ww = uvw  # unpack into len nvis vectors
# broadcast to the same shape as the data
# stub to broadcast uu,vv, and weights to all channels
broadcast = np.ones((nchan, 1))
uu = uu * broadcast
vv = vv * broadcast
weight = weight * broadcast

# calculate wavelengths in meters
wavelengths = c.value / chan_freq[:, np.newaxis]  # m

# calculate baselines in klambda
uu = 1e-3 * uu / wavelengths  # [klambda]
vv = 1e-3 * vv / wavelengths  # [klambda]

frequencies = chan_freq * 1e-9  # [GHz]

baseline = np.sqrt(uu*uu+vv*vv)

nbins=25
uvbin=np.linspace(baseline.min(),baseline.max(),num=nbins,endpoint=True)
duv = uvbin[1]-uvbin[0]

def calcvals(uvbin, baseline, weight,data):
    fluxmean_real=np.zeros(len(uvbin))
    fluxstd_real=np.zeros(len(uvbin))
    fluxmean_imag=np.zeros(len(uvbin))
    fluxstd_imag=np.zeros(len(uvbin))
    for i in range(len(uvbin)-1):
        uvmax=uvbin[i+1]
        uvmin=uvbin[i]
        data_where = np.logical_and(baseline>uvmin, baseline<=uvmax)
        data_where2 = np.logical_and(data_where, mask)
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
        # print(uvmin, uvmax)

    psd = fluxmean_real**2.+fluxmean_imag**2.
    variance_psd = 2.*fluxmean_real*fluxstd_real + 2.*fluxmean_imag*fluxstd_imag
    return fluxmean_real,fluxstd_real,fluxmean_imag,fluxstd_imag, psd, variance_psd

mean_realo,var_realo,mean_imago,var_imago,mean_psdo,var_psdo = calcvals(uvbin, baseline, weight,data)

maso = 1/(uvbin+0.5*duv)*180./np.pi*3600
uvdo = (uvbin+0.5*duv)

fig , ax = plt.subplots(nrows=3, figsize=(8,8), sharex=True)

ax[0].errorbar(mas1,1e3*mean_real1,yerr=1e3*var_real1,fmt='.', label='On-source')
ax[0].errorbar(maso,1e3*mean_realo,yerr=1e3*var_realo,fmt='.', label='Off-source')
ax[0].set_xscale('log')
ax[0].set_ylabel('Real mJy')
ax[0].legend(loc='lower right', numpoints=1)
ax[0].axhline(color='black')
ax[1].errorbar(mas1,1e3*mean_imag1,yerr=1e3*var_imag1,fmt='.')
ax[1].errorbar(maso,1e3*mean_imago,yerr=1e3*var_imago,fmt='.')
ax[1].set_xscale('log')
ax[1].set_ylabel('Imag mJy')
ax[1].axhline(color='black')
ax[2].errorbar(mas1,1e6*mean_psd1,yerr=1e6*var_psd1,fmt='.')
ax[2].errorbar(maso,1e6*mean_psdo,yerr=1e6*var_psdo,fmt='.')
ax[2].set_xscale('log')
ax[2].set_xlim([10.8,54])
#ax[2].set_ylim([-18,79])
ax[2].set_ylabel('mJy$^2$')
ax[2].set_xlabel('mas')
ax[2].axhline(color='black')
ax[2].set_xticks([11,15,20,25,30,35,40,45,50])
ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

np.savetxt('R_Dor_ID0_onoff.txt',(mas1,1e6*mean_psd1,1e6*var_psd1,maso,1e6*mean_psdo,1e6*var_psdo))

plt.savefig('R_Dor_ID0.onoff.pdf')
