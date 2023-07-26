# coding: utf-8

"""
Configuration of the HH -> bbWW analysis.
"""

import os

import law
import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))


def create_hbw_analysis(
    name,
    id,
    **kwargs,
) -> od.Analysis:

    #
    # the main analysis object
    #

    analysis_inst = od.Analysis(
        name=name,
        id=id,
        **kwargs,
    )

    # analysis-global versions
    analysis_inst.set_aux("versions", {
    })

    # files of sandboxes that might be required by remote tasks
    # (used in cf.HTCondorWorkflow)
    analysis_inst.x.bash_sandboxes = [
        "$CF_BASE/sandboxes/cf.sh",
        "$CF_BASE/sandboxes/venv_columnar.sh",
        # "$CF_BASE/sandboxes/venv_ml_tf.sh",
        "$HBW_BASE/sandboxes/venv_ml_plotting.sh",
    ]

    # cmssw sandboxes that should be bundled for remote jobs in case they are needed
    analysis_inst.set_aux("cmssw_sandboxes", [
        # "$CF_BASE/sandboxes/cmssw_default.sh",
    ])

    # clear the list when cmssw bundling is disabled
    if not law.util.flag_to_bool(os.getenv("HBW_BUNDLE_CMSSW", "1")):
        del analysis_inst.x.cmssw_sandboxes[:]

    # config groups for conveniently looping over certain configs
    # (used in wrapper_factory)
    analysis_inst.set_aux("config_groups", {})

    #
    # import campaigns and load configs
    #

    from hbw.config.config_run2 import add_config
    import cmsdb.campaigns.run2_2017_nano_v9

    campaign_run2_2017_nano_v9 = cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9

    # default config
    config_2017 = add_config(
        analysis_inst,
        campaign_run2_2017_nano_v9.copy(),
        config_name="config_2017",
        config_id=2,
    )

    # config with limited number of files
    config_2017_limited = add_config(
        analysis_inst,
        campaign_run2_2017_nano_v9.copy(),
        config_name="config_2017_limited",
        config_id=12,
        limit_dataset_files=2,
    )

    # new configs but with shorter names
    c17 = config_2017.copy(name="c17", id=102)  # noqa
    l17 = config_2017_limited.copy(name="l17", id=112)  # noqa

    return analysis_inst


# create all relevant analysis instances
analysis_hbw = create_hbw_analysis("analysis_hbw", 1, tags={"is_sl", "is_dl", "is_resonant", "is_nonresonant"})
hbw_sl = create_hbw_analysis("hbw_sl", 2, tags={"is_sl", "is_nonresonant"})
# TODO: commented out for now because otherwise configs are built multiple times
#       should I move each analysis into a separate file to prevent this?
# hbw_dl = create_hbw_analysis("hbw_dl", 3, tags={"is_dl", "is_nonresonant"})
# hbw_res_sl = create_hbw_analysis("hbw_res_sl", 4, tags={"is_sl", "is_resonant"})
