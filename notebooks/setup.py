# coding: utf-8

def setup():
    import sys
    import os

    # add paths to sys to allow importing hbw, columnflow, cmsdb, law, and order
    # this_dir = os.getcwd()
    hbw_dir = os.path.abspath(os.path.join("../"))
    cf_dir = os.path.abspath(os.path.join("../modules/columnflow"))
    db_dir = os.path.abspath(os.path.join("../modules/cmsdb"))
    law_dir = os.path.abspath(os.path.join("../modules/columnflow/modules/law"))
    od_dir = os.path.abspath(os.path.join("../modules/columnflow/modules/order"))
    # LAW_CONFIG_FILE = f"{hbw_dir}/law.cfg"

    for _dir in (hbw_dir, cf_dir, db_dir, law_dir, od_dir):
        if _dir not in sys.path:
            sys.path.append(_dir)

    import law
    # setup the law config such that hbw can be imported
    law.config.add_section("analysis")
    law.config.set("analysis", "default_analysis", "hbw.analysis.hbw_sl.hbw_sl")
    law.config.set("analysis", "default_config", "c17")
    law.config.set("analysis", "default_dataset", "hh_ggf_hbb_hvvqqlnu_kl1_kt1_powheg")
    law.config.set("analysis", "default_columnar_sandbox", "bash::$CF_BASE/sandboxes/venv_columnar.sh")
