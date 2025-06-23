# coding: utf-8

"""
Toy script to explore the creation of a correctionlib file with multiple, mixed dependencies (categories, bin ranges,
and formulas) for the DY weight correction in the hh2bbtautau analysis.
"""

from __future__ import annotations

import correctionlib.schemav2 as cs


# helpers ##############################################################################################################

def step_expr(x: float | int, up: bool, steepness: float | int = 1000) -> str:
    """
    Creates a "step" expression using an error function reach a plateau of 1 at *x*. When *up* is *True*, the step
    increases at *x* (zero-ing all values to the left), otherwise it decreases (zero-ing all values to the right).
    """
    sign = "" if up else "-"
    if x < 0:
        shifted_x = f"(x+{-x})"
    elif x > 0:
        shifted_x = f"(x-{x})"
    else:
        shifted_x = "x"
    return f"0.5*(erf({sign}{steepness}*{shifted_x})+1)"


def expr_in_range(expr: str, lower_bound: float | int, upper_bound: float | int) -> str:
    """
    Multiplies an expression with step functions so that it evaluates to zero outside a range given by *lower_bound* and
    *upper_bound*.
    """
    assert lower_bound < upper_bound, "Lower bound must be smaller than upper bound."
    # distinguish cases
    inf = float("inf")
    bounds = (lower_bound, upper_bound)
    # TODO: fix erf string!
    if bounds == (-inf, inf):
        return expr
    if bounds == (-inf, 0):
        return expr
        # return f"({expr})*{step_expr(0, up=False)}"
    if bounds == (0, inf):
        return expr
        # return f"({expr})*{step_expr(0, up=True)}"
    if lower_bound == -inf:
        return expr
        # return f"({expr})*{step_expr(upper_bound, up=False)}"
    if upper_bound == inf:
        return expr
        # return f"({expr})*{step_expr(lower_bound, up=True)}"
    return expr
    # return f"({expr})*{step_expr(lower_bound, up=True)}*{step_expr(upper_bound, up=False)}"


def create_dy_weight_correction(dy_weight_data: dict) -> cs.Correction:
    # create the correction object
    dy_weight_correction = cs.Correction(
        name="dy_correction_weight",
        description="DY weights derived in the phase space of the hh2bbtautau analysis, supposed to correct njet and "
        "ptll distributions, as well as correlated quantities.",
        version=1,
        inputs=[
            cs.Variable(name="era", type="string", description="Era name."),
            cs.Variable(name="njets", type="int", description="Number of jets in the event. To be capped at 10."),
            cs.Variable(name="ptll", type="real", description="pT of the reconstructed dilepton system in GeV."),
            cs.Variable(name="syst", type="string", description="Systematic variation."),
        ],
        output=cs.Variable(name="weight", type="real", description="DY event weight."),
        data=cs.Category(
            nodetype="category",
            input="era",
            content=[],
        ),
    )
    # dynamically fill it
    for era, era_data in dy_weight_data.items():
        era_category_content = []
        for syst, syst_data in era_data.items():
            binning_content = []
            for (min_njet, max_njet), formulas in syst_data.items():
                # create a joined expression for all formulas
                expr = "+".join(
                    expr_in_range(formula, lower_bound, upper_bound)
                    for lower_bound, upper_bound, formula in formulas
                )
                # add formula object to binning content
                binning_content.append(
                    cs.Formula(
                        nodetype="formula",
                        variables=["ptll"],
                        parser="TFormula",
                        expression=expr,
                    ),
                )
            # add a new category item for the jet bins
            njet_edges = sorted(set(sum(map(list, syst_data.keys()), [])))
            era_category_content.append(
                cs.CategoryItem(
                    key=syst,
                    value=cs.Binning(
                        nodetype="binning",
                        input="njets",
                        flow="error",
                        edges=njet_edges,
                        content=binning_content,
                    ),
                ),
            )
        # add a new category item for the era
        dy_weight_correction.data.content.append(
            cs.CategoryItem(
                key=era,
                value=cs.Category(
                    nodetype="category",
                    input="syst",
                    content=era_category_content,
                ),
            ),
        )
    return dy_weight_correction


if __name__ == "__main__":
    # import gzip

    # nested structure of formulas
    # year -> syst -> (min_njet, max_njet) -> [(lower_bound, upper_bound, formula), ...]
    inf = float("inf")
    """
    dy_weight_data = {
        "2022": {
            "nominal": {
                (0, 1): [
                    (0.0, 50.0, "1.0"),
                    (50, inf, "1.1"),
                ],
                (1, 11): [
                    (0.0, 50.0, "1.0"),
                    (50.0, inf, "1.1"),
                ],
            },
            "up1": {
                (0, 1): [
                    (0.0, inf, "1.05"),
                ],
                (1, 11): [
                    (0.0, inf, "1.05"),
                ],
            },
            "down1": {
                (0, 1): [
                    (0.0, inf, "0.95"),
                ],
                (1, 11): [
                    (0.0, inf, "0.95"),
                ],
            },
        },
    }
    """
    # # create and save the correction set
    # cset = cs.CorrectionSet(
    #     schema_version=2,
    #     description="Corrections derived for the hh2bbtautau analysis.",
    #     corrections=[
    #         create_dy_weight_correction(dy_weight_data),
    #     ],
    # )

    # with gzip.open("hbw_corrections.json.gz", "wt") as f:
    #     f.write(cset.model_dump_json(exclude_unset=True))
