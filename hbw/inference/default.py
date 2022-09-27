# coding: utf-8

"""
hbw inference model.
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def default(self):

    #
    # categories
    #

    self.add_category(
        "cat1",
        category="1e",
        variable="m_bb",
        mc_stats=True,
        data_datasets=["data_e_b"],
    )
    self.add_category(
        "cat2",
        category="1mu",
        variable="m_bb",
        mc_stats=True,
        data_datasets=["data_mu_b"],
    )

    #
    # processes
    #

    processes = [
        "hh_ggf_kt_1_kl_0_bbww_sl",
        "hh_ggf_kt_1_kl_1_bbww_sl",
        "hh_ggf_kt_1_kl_2p45_bbww_sl",
        "hh_ggf_kt_1_kl_5_bbww_sl",
        "tt",
        "st",
        # "dy_lep",
        # "w_lnu",
    ]

    inference_procnames = {
        "hh_ggf_kt_1_kl_0_bbww_sl": "ggHH_kl_0_kt_1_hbbhww",
        "hh_ggf_kt_1_kl_1_bbww_sl": "ggHH_kl_1_kt_1_hbbhww",
        "hh_ggf_kt_1_kl_2p45_bbww_sl": "ggHH_kl_2p45_kt_1_hbbhww",
        "hh_ggf_kt_1_kl_5_bbww_sl": "ggHH_kl_5_kt_1_hbbhww",
        "st": "ST",
        "tt": "TT",
    }

    for proc in processes:
        sub_process_insts = [p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)]
        datasets = [
            dataset_inst.name for dataset_inst in self.config_inst.datasets
            if any(map(dataset_inst.has_process, sub_process_insts))
        ]

        self.add_process(
            inference_procnames.get(proc, proc),
            process=proc,
            signal=("hh_ggf" in proc),
            mc_datasets=datasets,
        )

    #
    # parameters
    #

    # groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # xsec rate uncertainties
    for proc in processes:
        process_inst = self.config_inst.get_process(proc)
        # this might be a bit too brute-force since some parameters are correlated and others are not
        for unc_name in process_inst.xsecs[13].uncertainties:
            self.add_parameter(
                f"rate_{proc}_{unc_name}",
                process=inference_procnames.get(proc, proc),
                type=ParameterType.rate_gauss,
                effect=process_inst.xsecs[13].get(names=unc_name, direction=("down", "up"), factor=True),
            )
            self.add_parameter_to_group(f"rate_{proc}_{unc_name}", "theory")

    # minbias xs
    self.add_parameter(
        "CMS_pileup",
        type=ParameterType.shape,
        shift_source="minbias_xs",
    )
    self.add_parameter_to_group("CMS_pileup", "experiment")

    # Leftovers from test inference model, kept as an example for further implementations of uncertainties
    """
    # and again minbias xs, but encoded as symmetrized rate
    self.add_parameter(
        "CMS_pileup2",
        type=ParameterType.rate_uniform,
        transformations=[ParameterTransformation.effect_from_shape, ParameterTransformation.symmetrize],
        shift_source="minbias_xs",
    )
    self.add_parameter_to_group("CMS_pileup2", "experiment")

    # a custom asymmetric uncertainty that is converted from rate to shape
    self.add_parameter(
        "QCDscale_ST",
        process="ST",
        type=ParameterType.shape,
        transformations=[ParameterTransformation.effect_from_rate],
        effect=(0.5, 1.1),
    )

    # test
    self.add_parameter(
        "QCDscale_ttbar",
        category="cat1",
        process="TT",
        type=ParameterType.rate_uniform,
    )
    self.add_parameter(
        "QCDscale_ttbar_norm",
        category="cat1",
        process="TT",
        type=ParameterType.rate_unconstrained,
    )
    """
    #
    # post-processing
    #

    self.cleanup()
