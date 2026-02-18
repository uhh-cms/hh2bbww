

from cmsdb.util import DotDict
from collections import defaultdict
from scinum import Number
# from hbw.util import round_sig
from cmsdb.constants import (  # noqa: F401
    br_h,
    br_w,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


br_tau = DotDict(
    had=Number(0.6479),
    lep=Number(0.3521),
)
br_wlep = DotDict(
    e=Number(0.1071, {"br_w_e": 0.0016}),
    mu=Number(0.1063, {"br_w_mu": 0.0015}),
    tau=Number(0.1138, {"br_w_tau": 0.0021}),
)


br_w_full = br_w.copy()
br_w_full.tau_lep = br_wlep.tau * br_tau.lep
br_w_full.tau_had = br_wlep.tau * br_tau.had
br_w_full.lep_no_tau = br_wlep.e + br_wlep.mu

br_w_labels = {
    "had": "q",
    "tau_had": r"$\tau_\mathrm{had}$",
    "tau_lep": r"$\tau_\mathrm{lep}$",
    "lep_no_tau": r"$e/\mu$",
}

br_labels = {
    "bb": "bb",
    "ww": "WW",
    "gluglu": "gg",
    "tt": r"$\tau\tau$",
    "cc": "cc",
    "zz": "ZZ",
    "gg": r"$\gamma\gamma$",
    "zg": r"$Z\gamma$",
    "mm": r"$\mu\mu$",
}

# collect the HH BRs by looping over all combinations of H decay modes and applying the appropriate combinatorial factor
def make_br_dict(brs=br_h, labels=br_labels):
    out_brs = defaultdict(dict)
    for i, (decay1, label1) in enumerate(labels.items()):
        br1 = brs[decay1]
        for j, (decay2, label2) in enumerate(labels.items()):
        # for j, (decay2, label2) in enumerate(reversed(list(labels.items()))):
            br2 = brs[decay2]
            hh_decay = f"{label1}{label2}"
            if decay1 == decay2:
                br_value = br1**2
            else:
                br_value = 2 * br1 * br2
            out_brs[label1][label2] = br_value.nominal
            # if i > j:
            #     out_brs[label1][label2] = -1  # only fill the lower triangle of the matrix to avoid double counting
    return out_brs

# make 2d plot of HH BRs as a function of the two decay modes
hh_brs = make_br_dict(br_h, br_labels)
ww_brs = make_br_dict(br_w_full, br_w_labels)

def make_plot_hh(cmap="viridis"):
    decay_modes = list(br_labels.values())
    br_matrix = np.array([[hh_brs[decay1][decay2] for decay2 in decay_modes] for decay1 in decay_modes])
    plt.figure(figsize=(10, 8))
    # cmap options: "viridis", "plasma", "inferno", "magma", "cividis"
    # None values will be shown as white in the plot, so we can use them to only show the lower triangle of the matrix

    # im = plt.imshow(br_matrix, cmap=cmap)
    # log scale the colorbar
    im = plt.imshow(br_matrix, cmap=cmap)
    im.set_norm(LogNorm(vmin=np.abs(br_matrix[br_matrix > 0]).min(), vmax=1))
    # include values in the plot
    for i in range(len(decay_modes)):
        for j in range(len(decay_modes)):
            if i < j:
                continue
            br = br_matrix[i, j]
            # format number, 2 sig digits, in scientific notation (\times 10^N) if smaller 0.001
            if br < 0.01:
                # plt.text(j, i, f"{br:.2e}", ha="center", va="center", color="white")
                color = "white" if br < 1e-4 else "black"
                mantissa, exponent = f"{br:.2e}".split('e')
                exponent = int(exponent)  # Remove leading zeros
                plt.text(j, i, f"${mantissa}$\n$\\times 10^{{{exponent}}}$", ha="center", va="center", color=color, fontsize=13)
            else:
                plt.text(j, i, f"{br:.4f}", ha="center", va="center", color="black", fontsize=13)

    cbar = plt.colorbar(im)
    cbar.set_label(rf"$\mathcal{{BR}}(\mathrm{{HH \to XXYY}})$", fontsize=26)

    plt.xticks(ticks=np.arange(len(decay_modes)), labels=decay_modes, fontsize=16, rotation=0)
    plt.xlabel(rf"$\mathcal{{BR}}(\mathrm{{H \to XX}})$", fontsize=26)
    plt.ylabel(rf"$\mathcal{{BR}}(\mathrm{{H \to YY}})$", fontsize=26)
    plt.yticks(ticks=np.arange(len(decay_modes)), labels=decay_modes, fontsize=16, rotation=90)
    plt.title("Branching Ratios of HH decay modes", fontsize=24)
    plt.tight_layout()
    plt.savefig("hh_brs.png")
    plt.savefig("hh_brs.pdf")


def make_plot_ww(cmap="viridis"):
    decay_modes1 = list(br_w_labels.values())
    decay_modes2 = list(br_w_labels.values())[::-1]
    br_matrix = np.array([[ww_brs[decay1][decay2] for decay2 in decay_modes2] for decay1 in decay_modes1])
    # mask entries if i>(len(decay_modes1)-j) to only show the lower triangle of the matrix
    br_matrix = np.where(np.arange(len(decay_modes1))[:, None] >= np.arange(len(decay_modes2))[None, ::-1], br_matrix, -1)

    plt.figure(figsize=(10, 8))
    # cmap options: "viridis", "plasma", "inferno", "magma", "cividis"
    # None values will be shown as white in the plot, so we can use them to only show the lower triangle of the matrix

    # im = plt.imshow(br_matrix, cmap=cmap)
    # log scale the colorbar
    im = plt.imshow(br_matrix, cmap=cmap)
    im.set_norm(LogNorm(vmin=np.abs(br_matrix[br_matrix > 0]).min(), vmax=1))
    # include values in the plot
    for i in range(len(decay_modes1)):
        for j in range(len(decay_modes2)):
            # if i < j:
            #     continue
            br = br_matrix[i, j]
            # format number, 2 sig digits, in scientific notation (\times 10^N) if smaller 0.001
            color = "white" if br < 1e-1 else "black"
            if br < 0.01:
                # plt.text(j, i, f"{br:.2e}", ha="center", va="center", color="white")
                mantissa, exponent = f"{br:.2e}".split('e')
                exponent = int(exponent)  # Remove leading zeros
                plt.text(j, i, f"${mantissa}$\n$\\times 10^{{{exponent}}}$", ha="center", va="center", color=color, fontsize=13)
            else:
                plt.text(j, i, f"{br:.4f}", ha="center", va="center", color=color, fontsize=13)

    cbar = plt.colorbar(im)
    cbar.set_label(rf"$\mathcal{{BR}}(\mathrm{{WW \to XXYY}})$", fontsize=26)

    plt.xticks(ticks=np.arange(len(decay_modes2)), labels=decay_modes2, fontsize=16, rotation=0)
    plt.xlabel(rf"$\mathcal{{BR}}(\mathrm{{W \to XX}})$", fontsize=26)
    plt.ylabel(rf"$\mathcal{{BR}}(\mathrm{{W \to YY}})$", fontsize=26)
    plt.yticks(ticks=np.arange(len(decay_modes1)), labels=decay_modes1, fontsize=16, rotation=90)
    plt.title("Branching Ratios of WW decay modes", fontsize=24)
    plt.tight_layout()
    plt.savefig("ww_brs.png")
    plt.savefig("ww_brs.pdf")

make_plot_ww()

from IPython import embed; embed()
