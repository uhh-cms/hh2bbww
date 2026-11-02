#
# set some common variables for testing
#

export test_version="test2"
export test_config="l24v15"

export test_dataset="tt_dl_powheg"
export test_datasets="hh_ggf_hbb_hvv2l2nu_kl1_kt1_powheg,tt_dl_powheg,data_mu_c"

# cutflow
export test_cutflow_categories="incl,2mu"
export test_cutflow_variables="cf_n_jet"
export test_steps="cleanup,Dilepton,SR,Trigger,nJet1,nBjet1"

# plotting
export test_categories="incl,2mu"
export test_variables="jet1_pt"
export test_2d_variables="jet1_eta-jet1_phi"

# shift plotting
export test_shift_sources="pdf,murf_envelope,btag_bc,btag_light,e_sf,mu_id_sf,minbias_xs"
