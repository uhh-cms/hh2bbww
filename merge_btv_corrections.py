# coding: utf-8

"""
Tool for merging BTV correction sets into a single, analysis-application-friendly form.
Taken from Marcel Rieger.

Example output file:
/afs/cern.ch/work/m/mrieger/public/hbt/external_files/custom_btv_files/btag_merged_2024.json.gz
Created with:
/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-01-30/btagging_preliminary.json.gz
"""

from __future__ import annotations

import os
import pathlib
import gzip
import tempfile
import copy
from typing import Generator, Literal

import correctionlib.schemav2


def merge_btv_corrections(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> None:
    """
    Example implementation to merge BTV specific corrections applying to different flavors into a single correction
    that accepts multiple flavors. The strategy is somewhat opinionated for now but could be easily extended for more
    generic merging scenarios. The case handled here uses the b and light correction sets in [1] with additional info
    given in [2-4] about handling of c jets and their uncertainties.

    - [1] https://cms-analysis-corrections.docs.cern.ch/corrections_era/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/BTV/2026-01-30/#btagging_preliminaryjsongz  # noqa
    - [2] https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations
    - [3] https://btv-wiki.docs.cern.ch/PerformanceCalibration/SFUncertaintiesAndCorrelations
    - [4] https://cms-talk.web.cern.ch/t/2024-b-tagging-shape-sfs/141685

    For more info, see `_merge_btv_corrections_inplace` below.
    """
    # check input
    input_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(input_path))))
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input file not existing or not a file: {input_path}")

    # prepare output
    output_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(output_path))))
    if os.path.isfile(output_path):
        os.remove(output_path)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # read the corrections to merge
    if input_path.endswith(".gz"):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            with gzip.open(input_path, "rt") as f_in:
                tmp_file.write(f_in.read().encode())
            cset = correctionlib.schemav2.CorrectionSet.parse_file(tmp_file.name)
    else:
        cset = correctionlib.schemav2.CorrectionSet.parse_file(input_path)

    # perform the actual merging inplace
    _merge_btv_corrections_inplace(cset)

    # write them back to file
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wt") as f:
            f.write(cset.model_dump_json(exclude_unset=True))
    else:
        with open(output_path, "w") as f:
            f.write(cset.model_dump_json(exclude_unset=True))


def _merge_btv_corrections_inplace(
    cset: correctionlib.schemav2.CorrectionSet,
) -> None:
    """
    Implementation of the merging strategy.

    Input and output correction structure:
    correction
    └─ systematic
       └─ working_point
          └─ flavor
             └─ abseta
                └─ pt
                   └─ values

    Merging strategy:
    1) load b and light corrections
    2) append suffixes to systematic names of all b and light corrections to reflect their scope (as in ref. [3])
    3) for every systematic variation that only applies to light jets, register a copy of the central values of b/c jets
       with the name of the light variation and vice versa, allowing for simple evaluation on analysis-level
    4) create c correction
       - start with copy of b
       - change flavor node values to 4 in all systematic variations
       - double uncertainties w.r.t. the central one in all systematic variations
    5) merge b, c and light corrections
       - start with copy of b
       - merge with c under working_point nodes
       - merge with light under working_point nodes, which is trivial as per step 3)
    """
    # 1) get b and light corrections
    b_corr = copy.deepcopy(_get_correction(cset, "UParTAK4_kinfit"))
    l_corr = copy.deepcopy(_get_correction(cset, "UParTAK4_negtagDY"))

    # 2) amend systematic names with suffixes, store central ones for 3)
    b_syst_item_central = None
    l_syst_item_central = None
    for b_syst_item in _iter_btv_correction(b_corr, stop="systematic"):
        if b_syst_item.key != "central":
            b_syst_item.key += "_bc"
        else:
            b_syst_item_central = b_syst_item
    for l_syst_item in _iter_btv_correction(l_corr, stop="systematic"):
        if l_syst_item.key != "central":
            l_syst_item.key += "_light"
        else:
            l_syst_item_central = l_syst_item

    # 3) duplicate non-central variations from b to light with central values and vice versa
    for b_syst_item in _iter_btv_correction(b_corr, stop="systematic"):
        if b_syst_item.key.endswith("_bc"):
            l_syst_item = copy.deepcopy(l_syst_item_central)
            l_syst_item.key = b_syst_item.key
            l_corr.data.content.append(l_syst_item)
    for l_syst_item in _iter_btv_correction(l_corr, stop="systematic"):
        if l_syst_item.key.endswith("_light"):
            b_syst_item = copy.deepcopy(b_syst_item_central)
            b_syst_item.key = l_syst_item.key
            b_corr.data.content.append(b_syst_item)

    # 4) create c correction
    c_corr = copy.deepcopy(b_corr)
    central_sf_values = {}
    for c_syst_item, wp_item, flavor_item, eta_binning, eta_edges, _, sf_values in _iter_btv_correction(c_corr):
        # overwrite flavor
        flavor_item.key = 4
        # save central values during central iteration, or double uncertainty for other systematics
        central_key = (wp_item.key, eta_edges)
        if c_syst_item.key == "central":
            central_sf_values[central_key] = sf_values
        elif c_syst_item.key.endswith("_bc"):
            for i, (syst_value, central_value) in enumerate(zip(sf_values, central_sf_values[central_key])):
                sf_values[i] = 2 * syst_value - central_value

    # 5) merge b, c and light corrections into a single one
    merged_corr = copy.deepcopy(b_corr)
    merged_corr.name = "UParTAK4_merged"
    merged_corr.version = 1
    merged_corr.description = (
        f"Scale factors for b, c and light jets. Created by merging the '{b_corr.name}' and '{l_corr.name}' "
        "corrections, duplicating the b corrections for c jets and doubling their uncertainty. For more info, see the "
        "description of the respective corrections, especially regarding their systematic variations. Note that, "
        "systematic variations ending in '_bc' ('_light') only affect flavor 4/5 (0) and evaluate to 'central' values "
        "otherwise."
    )
    # merge with c at working_point level
    # (pairwise traversal would be faster, but brute-force is perfectly fine to keep it simple)
    for syst_item, wp_item in _iter_btv_correction(merged_corr, stop="working_point"):
        for c_syst_item, c_wp_item in _iter_btv_correction(c_corr, stop="working_point"):
            if (syst_item.key, wp_item.key) == (c_syst_item.key, c_wp_item.key):
                wp_item.value.content += c_wp_item.value.content
    # merge with light at working_point level (possible as per 3)
    for syst_item, wp_item in _iter_btv_correction(merged_corr, stop="working_point"):
        for l_syst_item, l_wp_item in _iter_btv_correction(l_corr, stop="working_point"):
            if (syst_item.key, wp_item.key) == (l_syst_item.key, l_wp_item.key):
                wp_item.value.content += l_wp_item.value.content

    # add back the merged correction
    cset.corrections.append(merged_corr)


def _get_correction(
    cset: correctionlib.schemav2.CorrectionSet,
    name: str,
) -> correctionlib.schemav2.Correction:
    """
    Helper to retrieve a correction by *name* from a schema v2 correction set *cset*. Raises a KeyError if not found.
    """
    for corr in cset.corrections:
        if corr.name == name:
            return corr
    raise KeyError(f"correction '{name}' not found in correction set '{cset.name}'")


def _iter_btv_correction(
    corr: correctionlib.schemav2.Correction,
    *,
    stop: Literal["systematic", "working_point", "flavor"] | None = None,
) -> Generator[tuple, None, None]:
    """
    Iteration helper to loop over all relevant nodes of a typical BTV correction *corr*, yielding the relevant info at
    each level. The iteration order is guaranteed to be such that the "central" systematic variation is always visited
    first if existing.

    When a *stop* level is given, the iteration depth is limited to that level and the yielded info only contains nodes
    up to that level.
    """
    # prepare sorted list of syst_items such that "central" is guaranteed to come first
    syst_items = list(corr.data.content)
    for i, syst_item in enumerate(syst_items):
        if syst_item.key == "central":
            syst_items.insert(0, syst_items.pop(i))
            break

    # iterate
    for syst_item in syst_items:
        if stop == "systematic":
            yield syst_item
            continue
        syst_cat = syst_item.value
        for wp_item in syst_cat.content:
            if stop == "working_point":
                yield (syst_item, wp_item)
                continue
            flavor_cat = wp_item.value
            for flavor_item in flavor_cat.content:
                if stop == "flavor":
                    yield (syst_item, wp_item, flavor_item)
                    continue
                eta_binning = flavor_item.value
                for i, pt_binning in enumerate(eta_binning.content):
                    eta_edges = (eta_binning.edges[i], eta_binning.edges[i + 1])
                    sf_values = pt_binning.content
                    yield (syst_item, wp_item, flavor_item, eta_binning, eta_edges, pt_binning, sf_values)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="merge BTV corrections into a single file")
    parser.add_argument("input_path", help="path to the input correction file to read")
    parser.add_argument("output_path", help="path to the output correction file to generate, overwrites existing files")
    args = parser.parse_args()

    merge_btv_corrections(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
