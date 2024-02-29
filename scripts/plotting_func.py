import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################################################################
def plot_final_image(rml, img_cube):
    fig, ax = plt.subplots(nrows=1)
    im = ax.imshow(
        np.squeeze(img_cube),
        origin="lower",
        interpolation="none",
        extent=rml.icube.coords.img_ext,
    )
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
    ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
    ax.set_title("Maximum likelihood image")
    plt.colorbar(im, label=r"Jy/$\mathrm{arcsec}^2$")
    plt.savefig('RML_loop_outputs/maximum_likelihood_image.pdf', format='pdf', bbox_inches='tight')
    print(f'Maximum likelihood image plot saved to: RML_loop_outputs/maximum_likelihood_image.pdf\n')
    # plt.close()
    return fig, ax
############################################################################################################


############################################################################################################
def plot_loss_per_iter(hyperparams_config, loss_tracker):
    fig, ax = plt.subplots(nrows=1)
    ax.plot(np.arange(hyperparams_config["epochs"]), loss_tracker, marker=".", color="k", linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_xscale("log")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per iteration")
    plt.savefig('RML_loop_outputs/loss_per_iteration.pdf', format='pdf', bbox_inches='tight')
    print(f'Loss per iteration plot saved to: RML_loop_outputs/loss_per_iteration.pdf\n')
    plt.close()
############################################################################################################


############################################################################################################
def plot_dirty(imager, img, beam, chan):
    kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3))
    bmplot = ax[0].imshow(beam[chan], **kw)
    #plt.colorbar(bmplot, ax=ax[0])
    ax[0].set_title("Dirty beam")
    imgplot = ax[1].imshow(img[chan], **kw)
    #plt.colorbar(imgplot, ax=ax[1])
    ax[1].set_title("Dirty image")
    for a in ax:
        a.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
        a.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
    plt.tight_layout()
    plt.savefig('RML_loop_outputs/dirty_beam_and_dirty_image.pdf', format='pdf', bbox_inches='tight')
    print(f'Dirty beam and dirty image plot saved to: RML_loop_outputs/dirty_beam_and_dirty_image.pdf')
    plt.close()
    print(f'The dirty image contains {np.sum(img < 0)} negative pixels.\n')
############################################################################################################


############################################################################################################
def plot_uv_distribution(uu, vv):
    fig, ax = plt.subplots(nrows=1)
    ax.scatter(uu, vv, s=1, rasterized=True, linewidths=0.0, c="k")
    ax.set_xlabel(r"$u$ [k$\lambda$]")
    ax.set_ylabel(r"$v$ [k$\lambda$]")
    ax.set_title("uv distribution")
    plt.savefig('RML_loop_outputs/uv_distribution.pdf', format='pdf', bbox_inches='tight')
    print(f'(u,v) distribution plot saved to: RML_loop_outputs/uv_distribution.pdf\n')
    plt.close()
############################################################################################################


############################################################################################################
def plot_visibilities_vs_uvdist(visibility_file):
    '''----------------------------------------------
    Plot the amplitude, phase, and real and imaginary
    parts of the visibilities vs uv distance
    ----------------------------------------------'''

    # load the mock visibilities from the .npz file
    d = np.load(visibility_file)
    uu = d["uu"]
    vv = d["vv"]
    # weight = d["weight"]
    data = d["data"]
    data_re = np.real(data)
    data_im = np.imag(data)

    #calculate the amplitude and phase of the visibilities
    amp = np.abs(data)
    phase = np.angle(data)

    # calculate the uv distance (baseline separations in meters, calculated as sqrt(u*u+v*v))
    uvdist = np.hypot(uu, vv)

    # plot the visibilities vs uv distance
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
    ax[3].set_xlabel(r"$q$ [k$\lambda$]")
    plt.tight_layout()
    plt.savefig('RML_loop_outputs/visibility_vs_uvdist_plots.pdf', format='pdf', bbox_inches='tight')
    print(f'Visibility vs uv distance plots saved to: RML_loop_outputs/visibility_vs_uvdist_plots.pdf\n')
    plt.close()
############################################################################################################


############################################################################################################
def plot_power_spectrum(power_spectrum_file):
    '''---------------------------------------------
    Plot the power spectrum from the input .csv file
    ---------------------------------------------'''

    # load the power spectrum data
    power_spectrum_df = pd.read_csv(power_spectrum_file, sep='\t', index_col='bin')

    # plot the power spectrum
    fig, ax = plt.subplots()
    ax.plot(power_spectrum_df['scales_arcsec'], power_spectrum_df['power'], marker='.', color='k')
    ax.set_xlabel(r"scale [arcsec]")
    ax.set_ylabel("Power")
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('RML_loop_outputs/power_spectrum.pdf', format='pdf', bbox_inches='tight')
    print(f'Power spectrum plot saved to: RML_loop_outputs/re_and_im_vis_vs_uvdist.pdf\n')
    plt.close()
############################################################################################################