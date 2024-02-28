import numpy as np
import matplotlib.pyplot as plt


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

def plot_uv_distribution(uu, vv):
    fig, ax = plt.subplots(nrows=1)
    ax.scatter(uu, vv, s=1, rasterized=True, linewidths=0.0, c="k")
    ax.set_xlabel(r"$u$ [k$\lambda$]")
    ax.set_ylabel(r"$v$ [k$\lambda$]")
    ax.set_title("uv distribution")
    plt.savefig('RML_loop_outputs/uv_distribution.pdf', format='pdf', bbox_inches='tight')
    print(f'(u,v) distribution plot saved to: RML_loop_outputs/uv_distribution.pdf\n')
    plt.close()