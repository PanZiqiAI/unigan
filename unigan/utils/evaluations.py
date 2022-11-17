
import numpy as np
from custom_pkg.basic.visualizers import plt


def vis_singular_values(sv_ema, sv, save_path):
    """
    :param sv_ema: (nz, )
    :param sv: (n_samples, nz)
    :param save_path:
    """
    if sv_ema is not None and len(sv_ema) < sv.shape[1]: sv_ema = None
    # 1. Init figure.
    plt.figure(dpi=700)
    plt.title("Jacobian singular values")
    # 2. Plot.
    n_rows = 3 if sv_ema is not None else 2
    # (1) EMA.
    if sv_ema is not None:
        plt.subplot(n_rows, 1, 1)
        for index, value in enumerate(sv_ema):
            plt.bar(x=index, height=value, width=0.75)
        plt.ylabel("ema")
    # (2) Mean.
    plt.subplot(n_rows, 1, n_rows-1)
    for index, value in enumerate(np.swapaxes(sv, 0, 1)):
        plt.bar(x=index, height=value.mean().item(), width=0.75)
    plt.ylabel("mean")
    # (3) Std.
    plt.subplot(n_rows, 1, n_rows)
    for index, value in enumerate(np.swapaxes(sv, 0, 1)):
        plt.bar(x=index, height=value.std(ddof=1).item(), width=0.75)
    plt.ylabel("std")
    """ Saving. """
    plt.xlabel("latent dimension")
    plt.savefig(save_path)
    plt.close()
