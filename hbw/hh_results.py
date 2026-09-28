"""
Run via
cf_sandbox venv_columnar "python3 hh_results.py"
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt


run2_expected_limits = {
    "CMS": {
        "luminosity_fb": 138,
        "wwyy": 54,
        "bbzz_4l": 41,
        "yytt": 26,
        "multilepton": 20,
        "bbww": 18,
        "bbww_dl": 27,
        "bbww_sl": 27,
        "bbyy": 5.6,
        "bbtt": 5.2,
        # improvements w.r.t. combination? https://arxiv.org/pdf/2202.09617
        "bbbb": 4.3,
        "bbbb_resolved": 7.8,
        "bbbb_boosted": 5.1,
    },
    "ATLAS": {
        "luminosity_fb": 140,
        "bbww_dl": 14,
        "multilepton": 11,
        "bbbb": 8.1,
        "bbyy": 5.0,
        "bbtt": 3.3,
    },
}

cms_run2_combination = {
    "url": "https://arxiv.org/abs/2510.07527",
    "ref": "arXiv:2510.07527",
    "label": "CMS Run 2",
    "collaboration": "CMS",
    "energy_TeV": 13.0,
    "limits": {
        "HH": {
            "observed": 3.5,
            "expected": 2.5,
        },
        "kappaL": {
            "best-fit": 1.51,
            "observed": (-1.35, 6.35),
            "expected": (-2.24, 7.89),
        },
        "kappa2V": {
            "best-fit": 1.02,
            "observed": (0.64, 1.40),
            "expected": (0.62, 1.41),
        },
    },
}

atlas_run2_combination = {
    "url": "https://arxiv.org/abs/2406.09971",
    "ref": "Phys. Rev. Lett. 133 (2024) 101801",
    "label": "ATLAS Run 2",
    "collaboration": "ATLAS",
    "energy_TeV": 13.0,
    "limits": {
        "HH": {
            "best-fit": 0.5,
            "observed": 2.9,
            "expected": 2.4,
            "expected_r1": 3.4,
        },
        "kappaL": {
            "best-fit": 3.8,
            "observed": (-1.2, 7.2),
            "expected": (-1.6, 7.2),
        },
        "kappa2V": {
            "best-fit": 1.02,
            "observed": (0.6, 1.5),
            "expected": (0.4, 1.6),
        },
    },
}
cms_atlas_combination = {
    "url": "https://arxiv.org/abs/2602.23991",
    "ref": "arXiv:2602.23991",
    "label": "CMS+ATLAS Run 2",
    "collaboration": "CMS+ATLAS",
    "energy_TeV": 13.0,
    "limits": {
        "HH": {
            "observed": 2.5,
            "expected": 1.7,
            "expected_r0": 1.7,
            "expected_r1": 2.8,
        },
        "kappaL": {
            "best-fit": 1.8,
            "observed": (-0.71, 6.1),
            "expected": (-1.3, 6.7),
        },
        "kappa2V": {
            "best-fit": 1.02,
            "observed": (0.73, 1.3),
            "expected": (0.66, 1.4),
        },
    },
    "CMS": {
        "HH": {
            "best-fit": 1.0,
            "observed": 3.5,
            "expected": 2.6,
            "expected_r0": 2.6,
            "expected_r1": 3.6,
        },
    },
    "ATLAS": {
        "HH": {
            "best-fit": 0.5,
            "observed": 2.9,
            "expected": 2.4,
            "expected_r0": 2.4,
            "expected_r1": 3.5,
        },
    },
}

atlas_bbtt_run3 = {
    "key": "bbtt",
    "url": "https://arxiv.org/abs/2607.26879",
    "ref": "arXiv:2607.26879",
    "label": r"$bb\tau\tau$ Run 2+3",
    "collaboration": "ATLAS",
    "energy_TeV": 13.6,
    "limits": {
        "luminosity_fb": 140 + 56,
        "HH": {
            "best-fit": 2.6,
            "significance_observed": 1.8,
            "significance_expected": 2.7,
            "observed": 5.3,
            "expected": 2.4,
        },
        "kappaL": {
            # "best-fit": -1.5,  # approximately from plot
            "observed": (-3.4, 10.1),
            "observed_excluded": (1.6, 5.5),  # only -3.4,-1.6 and 5.5,10.1 are allowed
            "expected": (-1.7, 8.5),
        },
        "kappa2V": {
            # "best-fit": 1.7,  # approximately from plot
            "observed": (-0.2, 2.4),
            "expected": (0.0, 2.1),
        },
    },
    "run2": {
        "luminosity_fb": 140,
        "HH": {
            "observed": 5.4,
            "expected": 2.6,
        },
    },
    "run3": {
        "luminosity_fb": 56,
        "HH": {
            "observed": 7.9,
            "expected": 4.3,
        },
    },
}

atlas_dummy = {
    "label": "ATLAS",
    "limits": {},
}
cms_dummy = {
    "label": "CMS",
    "limits": {},
}

bbtt_run2 = {
    "key": "bbtt",
    "url": "https://arxiv.org/abs/2206.09401",
    "ref": "Phys. Lett. B 842 (2023) 137531",
    "label": r"$bb\tau\tau$ Run 2",
    "collaboration": "CMS",
    "energy_TeV": 13.0,
    "luminosity_fb": 138,
    "limits": {
        "luminosity_fb": 138,
        "HH": {
            "observed": 3.3,
            "expected": 5.2,
        },
        "qqHH": {
            "observed": 154,
            "expected": 124,
        },
        "kappaL": {
            "best-fit": 3.61,
            "observed": (-1.7, 8.7),
            "expected": (-2.9, 9.8),
        },
        "kappa2V": {
            "best-fit": 1.07,
            "observed": (-0.4, 2.6),
            "expected": (-0.6, 2.8),
        },
    },
}

bbtt_run3 = {
    "key": "bbtt",
    # "url": "https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/",
    "url": "https://cms-alcm.web.cern.ch/analyses/HIG-25-008/HIG-25-008-pas-v4.pdf",
    "ref": "CMS-PAS-HIG-25-008",
    "label": r"$bb\tau\tau$ Run 2+3",
    "collaboration": "CMS",
    "energy_TeV": 13.6,
    "luminosity_fb": 172 + 138,
    "limits": {
        "luminosity_fb": 172 + 138,
        "HH": {
            "observed": 4.0,
            "expected": 3.0,
            "expected_r0": 3.0,
            "expected_r1": 3.9,
        },
        "kappaL": {
            "best-fit": 0.7,
            "observed": (-2.5, 9.4),
            "expected": (-1.5, 8.4),
        },
        "kappa2V": {
            "best-fit": 1.1,
            "observed": (0.02, 2.1),
            "expected": (-0.1, 2.2),
        },
    },
    "run3": {
        "luminosity_fb": 172,
        "HH": {
            "observed": 8.8,
            "expected": 3.9,
            "expected_r0": 3.9,
            "expected_r1": 4.6,
        },
        "kappaL": {
            "best-fit": -1.5,
            "observed": (-4.1, 11.2),
            "expected": (-2.2, 9.2),
        },
        "kappa2V": {
            "best-fit": 1.1,
            "observed": (-0.2, 2.3),
            "expected": (-0.2, 2.4),
        },
    },
    "run2": {
        "luminosity_fb": 138,
        "HH": {
            "observed": 3.5,
            "expected": 5.3,
            "expected_r0": 5.3,
            "expected_r1": 6.2,
        },
        "kappaL": {
            "best-fit": 3.6,
        },
        "kappa2V": {
            "best-fit": 1.1,
        },
    },
}

bbyy_run3 = {
    "key": "bbyy",
    "url": "https://cds.cern.ch/record/2945062",
    "ref": "CMS-PAS-HIG-25-007",
    "report": "CMS-PAS-HIG-25-007",
    "label": r"$bb\gamma\gamma$ Run 3",
    "collaboration": "CMS",
    "energy_TeV": 13.6,
    "luminosity_fb": 62,
    "1d strategy": {
        "label": r"$bb\gamma\gamma$ Run 3",
        "luminosity_fb": 62,
        "HH": {
            "observed": 7.4,
            "expected": 8.7,
        },
        "kappaL": {
            "best-fit": -1.4,
            "observed": (-3.9, 10.1),
            "expected": (-4.5, 11.1),
        },
    },
    "2d strategy": {
        "label": r"$bb\gamma\gamma$ Run 3",
        "luminosity_fb": 62,
        "HH": {
            "observed": 11.0,
            "expected": 7.3,
        },
        "kappaL": {
            "best-fit": -1.4,
            "observed": (-6.1, 12.8),
            "expected": (-3.6, 10.5),
        },
    },
}

atlas_bbyy_run3 = {
    "key": "bbyy",
    "url": "https://arxiv.org/abs/2507.03495",
    "ref": "Phys. Lett. B 876 (2026) 140280",
    # "label": r"ATLAS $bb\gamma\gamma$ Run 2+3",
    "label": r"$bb\gamma\gamma$ Run 2+3",
    "collaboration": "ATLAS",
    # "energy_TeV": 13.6,  # 13, 13.6
    "luminosity_fb": 308,  # 140, 168
    "limits": {
        "key": "bbyy",
        "luminosity_fb": 308,
        "HH": {
            "best-fit": 0.9,
            "observed": 3.7,
            "expected": 2.6,
            "expected_r0": 2.6,
            "expected_r1": 3.7,
        },
        "kappaL": {
            "best-fit": 1.4,
            "observed": (-1.6, 6.6),
            "expected": (-1.8, 6.9),
        },
        "kappa2V": {
            "best-fit": 0.9,
            "observed": (-0.5, 2.8),
            "expected": (-0.4, 2.6),
        },
    },
    "run2": {
        "luminosity_fb": 140,
        "HH": {
            "observed": 4.8,
            "expected": 4.2,
            "expected_r0": 4.2,
            "expected_r1": 5.5,
        },
    },
    "run3": {
        "luminosity_fb": 168,
        "HH": {
            "observed": 5.6,
            "expected": 3.8,
            "expected_r0": 3.8,
            "expected_r1": 5.0,
        },
    },
}

bbbb_run3_PAS = {
    "key": "bbbb",
    "url": "https://cds.cern.ch/record/2947325",
    "report": "CMS-PAS-HIG-24-010",
    "label": "bbbb Run 3",
    "collaboration": "CMS",
    "energy_TeV": 13.6,
    "luminosity_fb": 62,
    "limits": {
        "HH": {
            "observed": 4.4,
            "expected": 4.4,
        },
        "kappaL": {
            "observed": (-3.3, 9.7),
            "expected": (-3.4, 10.0),
        },
        "kappa2V": {
            "observed": (0.63, 1.43),
            "expected": (0.54, 1.51),
        },
    },
}

# in Run3, they quote this paper for bbbb resolved ----> https://arxiv.org/pdf/2202.09617
bbbb_run2 = {
    "resolved": {
        "url": "https://arxiv.org/abs/2202.09617",
        "HH": {
            "observed": 3.9,
            "expected": 7.8,
        },
    },
    "boosted": {
        "url": "https://arxiv.org/pdf/2205.06667",
        "HH": {
            "observed": 9.9,
            "expected": 5.1,
        },
    },
}

bbbb_run3 = {
    "key": "bbbb",
    "url": "https://arxiv.org/abs/2604.27044v1",
    "ref": "arXiv:2604.27044",
    "label": "bbbb Run 2+3",
    "collaboration": "CMS",
    # "energy_TeV": 13.6,
    "luminosity_fb": 138 + 62,
    "limits": {
        "key": "bbbb",
        "luminosity_fb": 138 + 62,
        "HH": {
            "observed": 4.7,
            "expected": 2.8,
        },
        "kappaL": {
            "best-fit": -0.67,
            "observed": (-4.1, 9.6),
            "expected": (-3.2, 9.8),
            # "observed": (-3.8, 10.0),  # from UL scans
            # "expected": (-2.0, 8.7),  # from UL scans
        },
        "kappa2V": {
            "best-fit": 1.02,
            "observed": (0.77, 1.27),
            "expected": (0.69, 1.35),
            # "observed": (0.74, 1.30),  # from UL scans
            # "expected": (0.69, 1.35),  # from UL scans
        },
    },
    "run2": {
        "key": "bbbb",
        "luminosity_fb": 138,
        "HH": {  # Abstract says something different???? ahhh because it's only resolved channel?
            "observed": 7.5,
            "expected": 3.8,
        },
    },
    "run2_resolved": {
        "key": "bbbb_resolved",
        "luminosity_fb": 138,
        "HH": {  # Abstract says something different???? ahhh because it's only resolved channel?
            "observed": 10.0,
            "expected": 5.9,
        },
    },
    "run3": {
        "label": "bbbb Run 3",
        "key": "bbbb",
        "luminosity_fb": 62,
        "HH": {
            "observed": 4.4,
            "expected": 4.4,
        },
        "kappaL": {
            "observed": (-3.4, 9.7),
            "expected": (-4.3, 10.9),
        },
        "kappa2V": {
            "observed": (0.68, 1.38),
            "expected": (0.54, 1.50),
        },
    },
    "resolved": {  # "first approach"
        "label": "bbbb resolved Run 3",
        "key": "bbbb_resolved",
        # "luminosity_fb": 200,
        "luminosity_fb": 62,  # unclear if limits are only from Run-3
        "HH": {
            "best-fit": 0.9,
            "observed": 6.3,
            "expected": 5.3,
        },
        "kappaL": {
            "observed": (-4.4, 11.7),
            "expected": (-3.8, 11.0),
        },
        "kappa2V": {
            "observed": (-0.08, 2.20),
            "expected": (-0.08, 2.19),
        },
    },
    "boosted": {  # "first approach, mass-fit method"
        # "label": "bbbb Run 3 [boosted]",
        "label": "bbbb boosted Run 3",
        "key": "bbbb_boosted",
        # "luminosity_fb": 200,
        "luminosity_fb": 62,  # unclear if limits are only from Run-3
        "HH": {
            "best-fit": -2.6,
            "observed": 7.1,
            "expected": 5.3,
        },
        "kappaL": {
            "observed": (-5.1, 10.9),
            "expected": (-6.8, 12.8),
        },
        "kappa2V": {
            "observed": (0.63, 1.42),
            "expected": (0.54, 1.51),
        },
    },
}

atlas_bbww_dl_run2 = {
    "key": "bbww_dl",
    "label": r"$bb\ell\ell + p_T^{miss}$",
    "collaboration": "ATLAS",
    "url": "https://arxiv.org/abs/2310.11286",
    "ref": "JHEP 02 (2024) 037",
    "energy_TeV": 13.0,
    "luminosity_fb": 140,
    "limits": {
        "luminosity_fb": 140,
        "HH": {  # different in combination...?
            "observed": 9.7,
            "expected": 16.2,
        },
        "kappaL": {
            "best-fit-guessed": True,
            # "best-fit": 3.5,  # approximately from plot
            "observed": (-6.2, 13.3),
            "expected": (-8.1, 15.5),
        },
        "kappa2V": {
            "best-fit-guessed": True,
            # "best-fit": 1.1,  # approximately from plot
            "observed": (-0.17, 2.4),
            "expected": (-0.51, 2.7),
        },
    },
}

bbww_dl_run3 = {
    "key": "bbww_dl",
    # "label": "bbWW(DL) Run 3",
    "label": "bbWW(DL) Run 3",
    "collaboration": "CMS",
    "url": "https://arxiv.org/abs/2604.02127",
    "ref": "arXiv:2604.02127",
    "energy_TeV": 13.6,
    "luminosity_fb": 62,
    "limits": {
        "luminosity_fb": 62,
        "key": "bbww_dl",
        "HH": {
            "observed": 12.0,
            "expected": 18.5,
        },
        "kappaL": {
            "best-fit": 3.4,
            "observed": (-9.1, 15.7),
            "expected": (-13.3, 19.8),
        },
        "kappa2V": {
            "best-fit": 1.02,
            "observed": (-0.20, 2.25),
            "expected": (-0.48, 2.54),
        },
    },
}

bbww_run2 = {
    "key": "bbww",
    "url": "https://inspirehep.net/literature/2768920",
    "label": "bbWW Run 2",
    "ref": "JHEP 07 (2024) 293",
    "collaboration": "CMS",
    "energy_TeV": 13.0,
    "luminosity_fb": 138,
    "limits": {
        "HH": {
            "observed": 16,
            "expected": 18,
        },
        "kappaL": {
            # "best-fit": ,
            "observed": (-7.2, 13.8),
            "expected": (-8.7, 15.2),
        },
        "kappa2V": {
            # "best-fit": ,
            "observed": (-1.4, 3.5),
            "expected": (-1.1, 3.2),
        },
    },
    "SL": {
        "label": "bbWW(SL) Run 2",
        # "ref": "D. Noll (PhD thesis)",
        "ref": "PhD thesis (Dennis Noll)",
        "HH": {
            "observed": 28,
            "expected": 27,
        },
        "kappaL": {  # from Dennis' thesis
            "best-fit": 7.99,
            "observed": (-11.8, 18.4),
            "expected": (-11.4, 17.9),
        },
        "kappa2V": {  # from Dennis' thesis
            "best-fit": 0.98,
            "observed": (-1.5, 3.6),
            "expected": (-1.6, 3.6),
        },
    },
    "DL": {
        "label": "bbWW(DL) Run 2",
        # "ref": "P. Fackeldey (PhD thesis)",
        "ref": "PhD thesis (Peter Fackeldey)",
        "HH": {
            "observed": 19,
            "expected": 27,
        },
        "kappaL": {  # from Peter's thesis
            "best-fit": 3.17,
            "observed": (-8.64, 15.21),
            "expected": (-11.43, 17.83),
        },
        "kappa2V": {  # from Peter's thesis
            "best-fit": 1.04,
            "observed": (-1.71, 3.84),
            "expected": (-2.22, 4.37),
        },
    },
}


hhh_6b_run2 = {
    "url": "https://cds.cern.ch/record/2945361",
    "label": "hhh_6b Run 2",
    "collaboration": "CMS",
    "energy_TeV": 13.0,
    "luminosity_fb": 138,
    "limits": {
        "HHH": {
            "observed": 588,
            "expected": 572,
        },
        "kappaL": {
            "observed": (-7, 12),
            "expected": (-6, 11),
        },
        "kappa4": {
            "observed": (-190, 190),
            "expected": (-190, 190),
        },
    },
}

hhh_4b2y_run2 = {
    "url": "https://cds.cern.ch/record/2937680",
    "label": "hhh_4b2y Run 2",
    "collaboration": "CMS",
    "energy_TeV": 13.0,
    "luminosity_fb": 138,
    "limits": {
        "HHH": {
            "observed": 3400,
            "expected": 2086,
        },
        "kappaL": {
            "best-fit": -10.9,
            "observed": (-16.1, 20.2),
            "expected": (-13.8, 18.0),
        },
        "kappa4": {
            "best-fit": -283,
            "observed": (-533, 541),
            "expected": (-397, 406),
        },
    },
}


def make_limits_summary(*results_dicts):
    summary = {}
    for result_dict in results_dicts:
        for key, value in result_dict["limits"].items():
            if key not in summary:
                summary[key] = []
            summary[key].append({
                "label": result_dict["label"],
                "observed": value.get("observed"),
                "expected": value.get("expected"),
            })
    return summary


def _collect_hh_entries(
    *results_dicts,
    limit_keys: Sequence[str] = ("limits", "limits_2d_strategy"),
):
    entries = []
    for result_dict in results_dicts:
        base_label = result_dict.get("label", "unknown")
        for limit_key in limit_keys:
            limits = result_dict.get(limit_key)
            if not limits or "HH" not in limits:
                continue

            if limits.get("label", None):
                label = limits["label"]
            elif limit_key == "limits":
                label = base_label
            else:
                label = f"{base_label} [{limit_key}]"

            entries.append({
                "label": label,
                "collaboration": result_dict.get("collaboration", "UNKNOWN"),
                "ref": limits.get("ref", result_dict.get("ref", "UNKNOWN")),
                "luminosity_fb": limits.get("luminosity_fb", result_dict.get("luminosity_fb")),
                "limits": limits,
            })
    return entries


def _format_hh_entry_label(entry):
    label = entry.get("label", "unknown")
    collaboration = entry.get("collaboration", "UNKNOWN")
    luminosity = entry.get("luminosity_fb", None)
    ref_idx = entry.get("ref_idx", None)
    ref = entry.get("ref", None)
    if luminosity is not None:
        second_line = f"{collaboration}, {luminosity} fb$^{{-1}}$"
    else:
        second_line = f"{collaboration}"
    if ref:
        return f"{label} [{second_line}|\n{ref}"
        # return f"{label}\n{second_line}\n{ref}"
    elif ref_idx:
        second_line = f"{second_line} [{ref_idx}]"
    return f"{label}\n{second_line}"


def _plot_scalar_limits(ax, entries, key: str, title: str):
    labels = [_format_hh_entry_label(entry) for entry in entries]
    obs_x = []
    obs_y = []
    exp_x = []
    x_tick_label_size = 14
    exp_y = []
    overlap_tol = 0.05
    overlap_offset = 0.06

    for idx, entry in enumerate(entries):
        values = entry["limits"].get(key)
        if not values:
            continue

        observed = values.get("observed")
        expected = values.get("expected")

        needs_offset = (
            observed is not None and
            expected is not None and
            abs(observed - expected) <= overlap_tol
        )

        if observed is not None:
            obs_x.append(observed)
            obs_y.append(idx - overlap_offset if needs_offset else idx)
        if expected is not None:
            exp_x.append(expected)
            exp_y.append(idx + overlap_offset if needs_offset else idx)

    y_values = list(range(len(entries)))
    ax.scatter(obs_x, obs_y, marker="o", color="black", label="Observed", zorder=3)
    ax.scatter(exp_x, exp_y, marker="s", facecolors="white", edgecolors="tab:blue", label="Expected", zorder=3)
    ax.set_yticks(y_values)
    ax.set_yticklabels(labels, fontsize=16)
    ax.tick_params(axis="x", labelsize=x_tick_label_size)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(1.0, color="tab:red", linestyle="--", linewidth=1.2, label="SM")
    # ax.set_title(title, fontsize=20)
    ax.set_xlabel(title, fontsize=20)


def _plot_interval_limits(ax, entries, key: str, title: str):
    labels = [_format_hh_entry_label(entry) for entry in entries]
    has_observed_label = False
    has_expected_label = False
    has_best_fit_label = False
    row_bounds = []
    x_tick_label_size = 14

    for idx, entry in enumerate(entries):
        values = entry["limits"].get(key)
        if not values:
            continue

        observed = values.get("observed")
        expected = values.get("expected")
        fit = values.get("best-fit")

        row_bounds.append({
            "idx": idx,
            "observed": observed,
            "expected": expected,
            "fit": fit,
        })

    x_values = []
    for row in row_bounds:
        for bounds in (row["observed"], row["expected"]):
            if bounds:
                x_values.extend(bounds)
        if row["fit"] is not None:
            x_values.append(row["fit"])

    if x_values:
        x_min = min(x_values)
        x_max = max(x_values)
        x_pad = max(0.5, 0.08 * (x_max - x_min))
        x_left = x_min - x_pad
        x_right = x_max + x_pad
    else:
        x_left = x_right = None

    lane_gap = 0.06
    lane_half_height = 0.24

    for row in row_bounds:
        idx = row["idx"]
        observed = row["observed"]
        expected = row["expected"]
        fit = row["fit"]
        observed_y0 = idx - lane_half_height
        observed_y1 = idx - lane_gap / 2
        expected_y0 = idx + lane_gap / 2
        expected_y1 = idx + lane_half_height

        if x_left is not None and observed:
            ax.fill_betweenx(
                [observed_y0, observed_y1],
                x_left,
                observed[0],
                color="black",
                alpha=0.3,
                linewidth=0,
                zorder=0,
            )
            ax.fill_betweenx(
                [observed_y0, observed_y1],
                observed[1],
                x_right,
                color="black",
                alpha=0.3,
                linewidth=0,
                zorder=0,
            )
        if x_left is not None and expected:
            ax.fill_betweenx(
                [expected_y0, expected_y1],
                x_left,
                expected[0],
                color="tab:blue",
                alpha=0.3,
                linewidth=0,
                zorder=0,
            )
            ax.fill_betweenx(
                [expected_y0, expected_y1],
                expected[1],
                x_right,
                color="tab:blue",
                alpha=0.3,
                linewidth=0,
                zorder=0,
            )

        if observed:
            ax.vlines(
                [observed[0], observed[1]],
                observed_y0,
                observed_y1,
                color="black",
                linewidth=2.2,
                label="Observed range" if not has_observed_label else None,
            )
            has_observed_label = True
        if expected:
            ax.vlines(
                [expected[0], expected[1]],
                expected_y0,
                expected_y1,
                color="tab:blue",
                linewidth=2.0,
                # linestyle="--",
                label="Expected range" if not has_expected_label else None,
            )
            has_expected_label = True

        if fit is not None:
            ax.scatter(
                [fit],
                [idx],
                marker="D",
                color="tab:green",
                zorder=3,
                label="Best fit" if not has_best_fit_label else None,
            )
            has_best_fit_label = True

    y_values = list(range(len(labels)))

    ax.set_yticks(y_values)
    ax.set_yticklabels(labels, fontsize=16)
    ax.tick_params(axis="x", labelsize=x_tick_label_size)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    if x_left is not None and x_right is not None:
        ax.set_xlim(x_left, x_right)
    ax.axvline(1.0, color="tab:red", linestyle="--", linewidth=1.2, label="SM")
    # ax.set_title(title, fontsize=20)
    ax.set_xlabel(title, fontsize=20)


def make_hh_summary_plot(
    *results_dicts,
    output_path: str | None = None,
    # include_limit_keys: Sequence[str] = ("limits",),
    include_limit_keys: Sequence[str] = ("limits", "2d strategy", "resolved", "boosted", "SL", "DL"),
):
    """
    Build a 3-panel summary plot for HH analyses:
    1) HH signal-strength limits
    2) kappaL exclusion ranges
    3) kappa2V exclusion ranges
    """
    entries = _collect_hh_entries(*results_dicts, limit_keys=include_limit_keys)
    if not entries:
        raise ValueError("No HH entries with HH limits were found.")

    # assign numeric footnote indices for unique non-empty refs
    ref_to_idx = {}
    next_idx = 1
    for entry in entries:
        ref = entry.get("ref", "")
        if ref and ref not in ("UNKNOWN", None):
            if ref not in ref_to_idx:
                ref_to_idx[ref] = next_idx
                next_idx += 1
            entry["ref_idx"] = ref_to_idx[ref]
        else:
            entry["ref_idx"] = None
    ref_to_idx = None

    # fig, axes = plt.subplots(ncols=3, figsize=(18, 0.95 * len(entries) + 2), sharey=True)
    fig, axes = plt.subplots(ncols=3, figsize=(18, 0.65 * len(entries) + 2), sharey=True)

    _plot_scalar_limits(
        axes[0],
        entries,
        key="HH",
        # title=r"$\mu_{HH}$ / SM",
        # title=r"95% CL upper limit on $\sigma(HH) / \sigma_{theory}$",
        title=r"95% CL upper limit on $\mu_{HH}$",
    )
    _plot_interval_limits(
        axes[1],
        entries,
        key="kappaL",
        title=r"95% CL exclusion ranges on $\kappa_{\lambda}$",
    )
    _plot_interval_limits(
        axes[2],
        entries,
        key="kappa2V",
        title=r"95% CL exclusion ranges on $\kappa_{2V}$",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[1].get_legend_handles_labels()
    for handle, label in zip(extra_handles, extra_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=len(labels),
        frameon=False,
        fontsize=18,
    )

    # fig.suptitle("HH Analysis Summary")
    if ref_to_idx:
        # leave room at the bottom for numbered references
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])
        # build ordered reference list
        ordered_refs = [ref for ref, idx in sorted(ref_to_idx.items(), key=lambda kv: kv[1])]
        ref_lines = [f"[{i+1}] {ref}" for i, ref in enumerate(ordered_refs)]
        refs_text = "\n".join(ref_lines)
        fig.text(0.01, 0.02, refs_text, ha="left", va="bottom", fontsize=9)
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=200)

    return fig, axes


def make_improvements_table(
        *results_dicts,
        # improvement_factor=lambda run3, run2, lumi, run2_lumi: 1 - (run3 * math.sqrt(lumi / run2_lumi) / run2),
        improvement_factor=lambda run3, run2, lumi, run2_lumi: run2 / (run3 * math.sqrt(lumi / run2_lumi)) - 1,
):
    """
    Build a table that compares for each entry of results_dicts the expected limit
    with the expected run2 limit (taken from run2_expected_limits) and computes the improvement factor.
    The table should have the format:
    | Analysis | Run 2                       | Run 3                       | Improvement Factor |
    |          | luminosity | expected limit | luminosity | expected limit |                    |
    """
    table = []
    for result_dict, key, label in results_dicts:
        limits_dict = result_dict.get(key, {})
        if key == "DUMMY":
            table.append({
                "Analysis": label,
                "Run 2 result": "",
                "New result": "",
                "Luminosity [fb$^{-1}$]": "",
                "Improvement": "",
            })
            continue
        if not label:
            label = result_dict.get("label", "unknown")
        key = limits_dict.get("key", None) or result_dict.get("key", None)
        if not key or not limits_dict:
            continue
        run2_limit = run2_expected_limits.get(result_dict.get("collaboration"), {}).get(key, None)
        if not run2_limit:
            print("Warning: No expected limit for Run 2 in", label, key, ", skipping.")
            continue
        run2_lumi = run2_expected_limits.get(result_dict.get("collaboration"), {}).get("luminosity_fb", None)
        if not run2_lumi:
            print("Warning: No luminosity information for Run 2 in", label, key, ", skipping.")
            continue
        run3_limit = limits_dict.get("HH", {}).get("expected", None)
        if not run3_limit:
            print("Warning: No expected limit for Run 3 in", label, key, ", skipping.")
            continue
        run3_lumi = limits_dict.get("luminosity_fb", None)
        if not run3_lumi:
            print("Warning: No luminosity information for Run 3 in", label, key, ", using main lumi key instead.")
            run3_lumi = result_dict.get("luminosity_fb", None)

        if run2_limit is not None and run3_limit is not None and run3_lumi is not None:
            improvement = improvement_factor(run3_limit, run2_limit, run3_lumi, run2_lumi) * 100
            table.append({
                "Analysis": label,
                # "Run 2 luminosity [fb$^{-1}$]": f"{int(run2_lumi)}",
                "Run 2 result": f"{run2_limit:.1f}",
                "New result": f"{run3_limit:.1f}",
                "Luminosity [fb$^{-1}$]": f"{int(run3_lumi)}",
                "Improvement": rf"{int(improvement)}\%",
            })
    return table


def tabulate_improvements_table(table):
    """
    Print the improvements table in a nice format.
    """
    import tabulate
    headers = list(table[0].keys()) if table else []
    data = [[row[h] for h in headers] for row in table]
    out_table = tabulate.tabulate(data, headers=headers, tablefmt="latex_raw")
    print(out_table)


if __name__ == "__main__":
    _table = make_improvements_table(
        (atlas_dummy, "DUMMY", r"ATLAS"),
        (atlas_bbyy_run3, "limits", r"\bbgg Run 2+3"),
        (atlas_bbyy_run3, "run3", r"\bbgg Run 3"),
        (atlas_bbyy_run3, "run2", r"\bbgg Run 2 re-analysis"),
        (atlas_bbtt_run3, "limits", r"\bbtautau Run 2+3"),
        (atlas_bbtt_run3, "run3", r"\bbtautau Run 3"),
        (atlas_bbtt_run3, "run2", r"\bbtautau Run 2 re-analysis"),
        (cms_dummy, "DUMMY", r"CMS"),
        (bbbb_run3, "limits", r"\bbbb Run 2+3"),
        (bbbb_run3, "run3", r"\bbbb Run 3"),
        (bbbb_run3, "resolved", r"\bbbb resolved Run 3"),
        (bbbb_run3, "boosted", r"\bbbb boosted Run 3"),
        (bbbb_run3, "run2_resolved", r"\bbbb resolved re-analysis"),
        # (bbbb_run3, "run2", r"\bbbb resolved re-analysis (+ boosted)"),
        # (bbtt_run2, "limits", None),  # to be replaced with Run 3
        (bbtt_run3, "limits", r"\bbtautau Run 2+3"),
        (bbtt_run3, "run3", r"\bbtautau Run 3"),
        # (bbyy_run3, "limits", None),
        (bbyy_run3, "2d strategy", r"$\bbgg$ Run 3"),
        (bbww_dl_run3, "limits", r"\bbww Run 3"),
    )
    tabulate_improvements_table(_table)

    make_hh_summary_plot(
        bbbb_run3,
        bbtt_run3,
        bbyy_run3,
        bbww_dl_run3,
        bbww_run2,
        # hhh_6b_run2,
        # hhh_4b2y_run2,
        output_path="hh_summary_plot.pdf",
    )
    make_hh_summary_plot(
        atlas_bbyy_run3,
        atlas_bbtt_run3,
        bbbb_run3,
        bbtt_run3,
        bbyy_run3,
        bbww_dl_run3,
        output_path="hh_summary_plot_run3.pdf",
    )
    make_hh_summary_plot(
        bbww_run2,
        atlas_bbww_dl_run2,
        bbww_dl_run3,
        output_path="hh_summary_plot_bbww.pdf",
    )
    make_hh_summary_plot(
        cms_run2_combination,
        atlas_run2_combination,
        cms_atlas_combination,
        output_path="hh_combination_plot.pdf",
    )
