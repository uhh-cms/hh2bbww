

"""
Usage:
cf_sandbox venv_columnar "python3 hh_brs.py"
"""


from cmsdb.util import DotDict
from collections import defaultdict
from scinum import Number
# from hbw.util import round_sig
from cmsdb.constants import (  # noqa: F401
    br_hh,
    br_h,
    br_w,
    br_z,
)

from cmsdb.processes import (
    hh_ggf_kl1_kt1,
    hh_vbf_kv1_k2v1_kl1,
    hhh_ggf,
)
com = 13.6
lumi = 62.4
xs_hh = hh_ggf_kl1_kt1.xsecs[com] + hh_vbf_kv1_k2v1_kl1.xsecs[com]
xs_hhh = hhh_ggf.xsecs[com]

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

br_zlep = DotDict(
    e=Number(0.033632, {"br_z_e": 0.000042}),
    mu=Number(0.033662, {"br_z_mu": 0.000066}),
    tau=Number(0.033696, {"br_z_tau": 0.000083}),
)


br_w_full = br_w.copy()
br_w_full.tau_lep = br_wlep.tau * br_tau.lep
br_w_full.tau_had = br_wlep.tau * br_tau.had
br_w_full.lep_no_tau = br_wlep.e + br_wlep.mu
br_w_lep_no_tau_had = br_wlep.e + br_wlep.mu + br_wlep.tau * br_tau.lep


br_bbww_dl_no_tau_had = br_hh.bbww * br_w_lep_no_tau_had ** 2

br_z_full = br_z.copy()
br_z_full.tau_leplep = br_zlep.tau * br_tau.lep ** 2
br_z_full.tau_hadhad = br_zlep.tau * br_tau.had ** 2
br_z_full.tau_hadlep = br_zlep.tau * br_tau.had * br_tau.lep * 2
br_z_full.no_tau = br_zlep.e + br_zlep.mu

br_tau_labels = {
    "had": r"$\tau\rightarrow qq$",
    "lep": r"$\tau\rightarrow \ell\nu_\ell$",
}
br_w_labels = {
    "had": "qq",
    # "tau_had": r"$\tau_\mathrm{had}$",
    # "tau_lep": r"$\tau_\mathrm{lep}$",
    "tau_had": r"$\tau_\mathrm{qq}$",
    "tau_lep": r"$\tau_{\ell\nu}$",
    "lep_no_tau": r"$e/\mu$",
}
br_w_labels_details = {
    "had": "qq",
    "tau_had": r"$\tau\nu_\tau \rightarrow qq\nu_\tau$",
    "tau_lep": r"$\tau\nu_\tau \rightarrow \ell\nu_\ell\nu_\tau$",
    "lep_no_tau": r"$e\nu_e/\mu\nu_\mu$",
}

br_z_labels = {
    "tau_hadhad": r"$\tau_\mathrm{qq}\tau_\mathrm{qq}$",
    "tau_hadlep": r"$\tau_\mathrm{qq}\tau_{\ell\nu}$",
    "tau_leplep": r"$\tau_{\ell\nu}\tau_{\ell\nu}$",
    # "tau_hadhad": r"$\tau_\mathrm{had}\tau_\mathrm{had}$",
    # "tau_hadlep": r"$\tau_\mathrm{had}\tau_\mathrm{lep}$",
    # "tau_leplep": r"$\tau_\mathrm{lep}\tau_\mathrm{lep}$",
    "no_tau": r"$ee/\mu\mu$",
    "nunu": r"$\nu\nu$",
    "qq": "qq",
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
br_labels_reduced = {
    "bb": "bb",
    "ww": "WW",
    "gluglu": "gg",
    "tt": r"$\tau\tau$",
    "cc": "cc",
    "zz": "ZZ",
    "gg": r"$\gamma\gamma$",
}


def make_br_tabular(brs=br_h, labels=br_labels):
    from tabulate import tabulate
    toplabels = ["Decay mode", "Branching ratio"]
    table = []
    sum_brs = 0.
    for decay_mode, label in labels.items():
        sum_brs += brs[decay_mode].nominal
        br = brs[decay_mode]
        table.append([label, f"${br.str(format='pdg', style='latex', combine_uncs='all'  )}$"])

    tabular = tabulate(table, headers=toplabels, tablefmt="latex_raw")
    print(tabular)
    print(f"sum of BRs: {sum_brs:.8f}")
    return tabular


def make_table_from_tabular(tabular, caption="", label="tab:"):
    table = rf"""
\begin{{table}}[!htbp]
  \centering
  \caption{{{caption}}}%
  \label{{{label}}}
  \renewcommand{{\arraystretch}}{{1.3}}
  \begin{{small}}{tabular}
  \end{{small}}
  \renewcommand{{\arraystretch}}{{1.0}}
\end{{table}}
"""
    return table


tab = make_br_tabular()
tab = make_table_from_tabular(tab, caption="Branching ratios of Higgs decay modes.", label="tab:higgs_brs")
print(tab)


# collect the HH BRs by looping over all combinations of H decay modes and applying the appropriate combinatorial factor
def make_br_dict(brs=br_h, labels=br_labels, base_br: float = 1.0):
    out_brs = defaultdict(dict)
    for i, (decay1, label1) in enumerate(labels.items()):
        br1 = brs[decay1]
        # for j, (decay2, label2) in enumerate(reversed(list(labels.items()))):
        for j, (decay2, label2) in enumerate(labels.items()):
            br2 = brs[decay2]
            if decay1 == decay2:
                br_value = br1**2
            else:
                br_value = 2 * br1 * br2
            out_brs[label1][label2] = br_value.nominal * base_br
            # if i > j:
            #     out_brs[label1][label2] = -1  # only fill the lower triangle of the matrix to avoid double counting
    return out_brs


def make_hhh_br_dict(brs=br_h, labels=br_labels, decay3: str = "bb", base_br: float = 1.0):
    br3 = brs[decay3]
    out_brs = defaultdict(dict)
    for i, (decay1, label1) in enumerate(labels.items()):
        br1 = brs[decay1]
        # for j, (decay2, label2) in enumerate(reversed(list(labels.items()))):
        for j, (decay2, label2) in enumerate(labels.items()):
            br2 = brs[decay2]
            if (decay1 == decay2) and (decay2 == decay3):
                br_value = br1**3
            elif (decay1 != decay2) and (decay2 != decay3) and (decay1 != decay3):
                br_value = 6 * br1 * br2 * br3
            else:
                br_value = 3 * br1 * br2 * br3
            out_brs[label1][label2] = br_value.nominal * base_br
            # if i > j:
            #     out_brs[label1][label2] = -1  # only fill the lower triangle of the matrix to avoid double counting
    return out_brs


def make_plot(
    brs_dict,
    labels_dict,
    title="Branching Ratios",
    cbar_label=r"$\mathcal{BR}$",
    xlabel=r"$\mathcal{BR}(X)$",
    ylabel=r"$\mathcal{BR}(Y)$",
    outfile_base="br_plot",
    cmap="viridis",
    reverse_x=False,
    reverse_y=False,
    upper_quadrant=False,
    color_threshold_small=None,
    figsize=(10, 8),
):
    """
    Generic function to plot branching ratio matrices.

    Parameters:
    - brs_dict: dictionary of branching ratios
    - labels_dict: dictionary of labels
    - title: plot title
    - cbar_label: colorbar label
    - xlabel, ylabel: axis labels
    - outfile_base: base name for output files (png/pdf will be added)
    - cmap: colormap
    - reverse_x: reverse the x-axis order
    - reverse_y: reverse the y-axis order
    - upper_quadrant: if True, show only the upper triangle of the matrix; if False, show only the lower triangle
    - color_threshold_small: threshold for small value color (white vs black)
    - figsize: figure size
    """
    text_font_size = {
        2: 32,
        4: 22,
        5: 20,
        6: 18,
        7: 16,
        9: 13,
    }.get(len(labels_dict), 13)  # adjust text size based on number of decay modes

    decay_modes_y = list(labels_dict.values())
    decay_modes_x = list(labels_dict.values())

    if reverse_y:
        decay_modes_y = decay_modes_y[::-1]
    if reverse_x:
        decay_modes_x = decay_modes_x[::-1]

    # Build matrix with the appropriate order
    br_matrix = np.array([
        [brs_dict[labels_dict_key_y][labels_dict_key_x]
         for labels_dict_key_x in (list(labels_dict.values())[::-1] if reverse_x else list(labels_dict.values()))]
        for labels_dict_key_y in (list(labels_dict.values())[::-1] if reverse_y else list(labels_dict.values()))
    ])

    # set threshold for white color to last 20% of the colorbar range if not provided
    if color_threshold_small is None:
        max_log = np.log10(1)
        min_log = np.log10(np.abs(br_matrix[br_matrix > 0]).min())
        color_threshold_small = 10 ** ((max_log + min_log) / 3)  # last 20% of the range in log space

    # choose correct quadrant based on reverse_x and reverse_y
    # (lower left if both False or True, lower right if one of them is true)
    if reverse_x.__xor__(reverse_y):
        # show only the lower right triangle
        n_y, n_x = br_matrix.shape
        row_indices, col_indices = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
        if upper_quadrant:
            br_matrix = np.where(row_indices <= (n_x - 1 - col_indices), br_matrix, -1)
        else:
            br_matrix = np.where(row_indices >= (n_x - 1 - col_indices), br_matrix, -1)
    else:
        # show only the lower left triangle
        n_y, n_x = br_matrix.shape
        row_indices, col_indices = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
        if upper_quadrant:
            br_matrix = np.where(row_indices <= col_indices, br_matrix, -1)
        else:
            br_matrix = np.where(row_indices >= col_indices, br_matrix, -1)

    plt.figure(figsize=figsize)
    im = plt.imshow(br_matrix, cmap=cmap)
    im.set_norm(LogNorm(vmin=np.abs(br_matrix[br_matrix > 0]).min(), vmax=1))

    # Add text annotations
    for i in range(len(decay_modes_y)):
        for j in range(len(decay_modes_x)):
            br = br_matrix[i, j]
            if br < 0:  # Skip masked values
                continue

            color = "white" if br < color_threshold_small else "black"
            if br < 0.01:
                mantissa, exponent = f"{br:.2e}".split("e")
                exponent = int(exponent)
                plt.text(j, i, f"${mantissa}$\n$\\times 10^{{{exponent}}}$",
                        ha="center", va="center", color=color, fontsize=text_font_size)
            else:
                plt.text(j, i, f"{br:.4f}", ha="center", va="center", color=color, fontsize=text_font_size)

    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label, fontsize=26)

    ticks_fontsize = max(min(16, text_font_size), 24)
    plt.xticks(ticks=np.arange(len(decay_modes_x)), labels=decay_modes_x, fontsize=ticks_fontsize, rotation=0)
    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    plt.yticks(ticks=np.arange(len(decay_modes_y)), labels=decay_modes_y, fontsize=ticks_fontsize, rotation=90, va="center")  # noqa: E
    plt.title(title, fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{outfile_base}.png")
    plt.savefig(f"{outfile_base}.pdf")
    plt.close()


def make_plot_hh(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        hh_brs,
        br_labels,
        title=r"Branching Ratios of $\mathrm{HH \to XXYY}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{HH \to XXYY})$",
        xlabel=r"$\mathrm{H \to XX}$",
        ylabel=r"$\mathrm{H \to YY}$",
        outfile_base="hh_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
        # color_threshold_small=1e-4,
    )
    make_plot(
        hh_brs,
        br_labels_reduced,
        # title=r"Branching Ratios of $\mathrm{HH \to XXYY}$ decay modes",
        title="",
        cbar_label=r"$\mathcal{BR}(\mathrm{HH \to XXYY})$",
        xlabel=r"$\mathrm{H \to XX}$",
        ylabel=r"$\mathrm{H \to YY}$",
        outfile_base="hh_brs_reduced",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
        # color_threshold_small=1e-4,
    )


# initialize BR dictionaries for HH, WW, and ZZ decays
hh_brs = make_br_dict(br_h, br_labels)
hh_brs_reduced = make_br_dict(br_h, br_labels_reduced)
ww_brs = make_br_dict(br_w_full, br_w_labels)
zz_brs = make_br_dict(br_z_full, br_z_labels)
tautau_brs = make_br_dict(br_tau, br_tau_labels)


def make_plot_ww(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        ww_brs,
        br_w_labels,
        title=r"Branching Ratios of $\mathrm{WW \to XXYY}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{WW \to XXYY})$",
        xlabel=r"$\mathrm{W \to XX}$",
        ylabel=r"$\mathrm{W \to YY}$",
        outfile_base="ww_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


def make_plot_zz(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        zz_brs,
        br_z_labels,
        title=r"Branching Ratios of $\mathrm{ZZ \to XXYY}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{ZZ \to XXYY})$",
        xlabel=r"$\mathrm{Z \to XX}$",
        ylabel=r"$\mathrm{Z \to YY}$",
        outfile_base="zz_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


def make_plot_tautau(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        tautau_brs,
        br_tau_labels,
        title=r"Branching Ratios of $\mathrm{\tau\tau \to XXYY}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{\tau\tau \to XXYY})$",
        xlabel=r"$\mathrm{\tau \to XX}$",
        ylabel=r"$\mathrm{\tau \to YY}$",
        outfile_base="tautau_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


#
# make plots with multiplying the appropriate HH BR to get the absolute BR for the
# full decay chain (e.g. HH->bbWW->bbXXYY)
#

bbww_brs = make_br_dict(br_w_full, br_w_labels, base_br=br_hh.bbww.nominal)
bbzz_brs = make_br_dict(br_z_full, br_z_labels, base_br=br_hh.bbzz.nominal)
bbtautau_brs = make_br_dict(br_tau, br_tau_labels, base_br=br_hh.bbtt.nominal)


def make_plot_bbww(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        bbww_brs,
        br_w_labels,
        title=r"Branching Ratios of $\mathrm{HH \to bbWW}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{HH \to bbWW \to bbXXYY})$",
        xlabel=r"$\mathrm{W \to XX}$",
        ylabel=r"$\mathrm{W \to YY}$",
        outfile_base="bbww_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


def make_plot_bbzz(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        bbzz_brs,
        br_z_labels,
        title=r"Branching Ratios of $\mathrm{HH \to bbZZ}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{HH \to bbZZ \to bbXXYY})$",
        xlabel=r"$\mathrm{Z \to XX}$",
        ylabel=r"$\mathrm{Z \to YY}$",
        outfile_base="bbzz_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


def make_plot_bbtautau(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        bbtautau_brs,
        br_tau_labels,
        title=r"Branching Ratios of $\mathrm{HH \to bb\tau\tau}$ decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{HH \to bb\tau\tau \to bbXXYY})$",
        xlabel=r"$\mathrm{\tau \to XX}$",
        ylabel=r"$\mathrm{\tau \to YY}$",
        outfile_base="bbtautau_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


hhh_brs = make_hhh_br_dict(br_h, br_labels, decay3="bb")


def make_plot_hhh(cmap="viridis", reverse_x=False, reverse_y=False, upper_quadrant=False):
    make_plot(
        hhh_brs,
        br_labels,
        title="Branching Ratios of HHH decay modes",
        cbar_label=r"$\mathcal{BR}(\mathrm{HHH \to bbXXYY})$",
        xlabel=r"$\mathrm{H \to XX}$",
        ylabel=r"$\mathrm{H \to YY}$",
        outfile_base="hhh_brs",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )
    make_plot(
        hhh_brs,
        br_labels_reduced,
        title="",
        cbar_label=r"$\mathcal{BR}(\mathrm{HHH \to bbXXYY})$",
        xlabel=r"$\mathrm{H \to XX}$",
        ylabel=r"$\mathrm{H \to YY}$",
        outfile_base="hhh_brs_reduced",
        cmap=cmap,
        reverse_x=reverse_x,
        reverse_y=reverse_y,
        upper_quadrant=upper_quadrant,
    )


make_plot_hh(reverse_x=True)
make_plot_hhh(reverse_x=True)

make_plot_ww(reverse_x=True)
make_plot_zz(reverse_x=True)
make_plot_tautau(reverse_x=True)

make_plot_bbww(reverse_x=True)
make_plot_bbzz(reverse_x=True)
make_plot_bbtautau(reverse_x=True)
