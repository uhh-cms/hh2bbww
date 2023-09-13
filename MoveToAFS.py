!/usr/bin/python

from datetime import datetime
import sys, time, subprocess
#from ROOT import *
from multiprocessing import Pool, Value
import random
import shutil
import glob
import os
import shutil
import json

debug = False

# list with the control plots
#'plot__proc_16_fabce60b2b__cat_incl__var_electron_eta.png', (before ml_inputs)

plots = [
'plot__proc_22_96f5201175__cat_incl__var_electron_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_electron_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_electron_pt.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_btagHbb.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_deepTagMD_HbbvsQCD.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_mass.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_msoftdrop.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_pt.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_tau1.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_tau2.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet1_tau21.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_btagHbb.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_deepTagMD_HbbvsQCD.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_mass.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_msoftdrop.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_pt.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_tau1.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_tau2.png',
'plot__proc_22_96f5201175__cat_incl__var_fatjet2_tau21.png',
#'plot__proc_22_96f5201175__cat_incl__var_jet1_btagDeepB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet1_btagDeepFlavB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet1_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_jet1_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_jet1_pt.png',
#'plot__proc_22_96f5201175__cat_incl__var_jet2_btagDeepB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet2_btagDeepFlavB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet2_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_jet2_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_jet2_pt.png',
#'plot__proc_22_96f5201175__cat_incl__var_jet3_btagDeepB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet3_btagDeepFlavB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet3_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_jet3_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_jet3_pt.png',
#'plot__proc_22_96f5201175__cat_incl__var_jet4_btagDeepB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet4_btagDeepFlavB.png',
'plot__proc_22_96f5201175__cat_incl__var_jet4_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_jet4_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_jet4_pt.png',
'plot__proc_22_96f5201175__cat_incl__var_muon_eta.png',
'plot__proc_22_96f5201175__cat_incl__var_muon_phi.png',
'plot__proc_22_96f5201175__cat_incl__var_muon_pt.png',
'plot__proc_22_96f5201175__cat_incl__var_n_electron.png',
'plot__proc_22_96f5201175__cat_incl__var_n_fatjet.png',
'plot__proc_22_96f5201175__cat_incl__var_n_jet.png',
'plot__proc_22_96f5201175__cat_incl__var_n_muon.png',
]

# String list for path_in (where the plots are created)

strings = ['config_2017/', 'nominal/', 'calib__skip_jecunc/', 'sel__default/', 'prod__features/', 'datasets_38_3fc8d30155/', 'v1/']
#config = 'config_2017'
#shifts = 'nominal'
#calibration = 'calib__skip_jecunc'
#selector = 'sel__default'
#producer = 'prod__features'
#dataset = 'datasets_38_3fc8d30155'
#version = 'v1'

# ------------------------------------------------------------------------------

def MoveToAFS():

    path_in  = '/nfs/dust/cms/user/moelsjan/WorkingArea/DiHiggs/hh2bbww/data/hbw_store/analysis_hbw/cf.PlotVariables1D/'
    path_out = '/afs/desy.de/user/m/moelsjan/public/controlplots/control_17.05./'

    for string in strings:
        path_in += string

    #print "Start moving ... "
    cmd  = 'cp'+' '
    cmd += path_in
    for plot in plots:

        cmd1 = cmd
        cmd1 += plot + ' ' + path_out

        os.system(cmd1)

        #for key2 in plots[plot1]:
        #    cmd2  = cmd1
        #    cmd2 += key2+' '
        #    cmd2 += path_out
        #    cmd2 += plots[plot1][key2]

            #cmd_list = [cmd2]
            #if 'FSR/SYS' in key1:
            #    cmd_list = [cmd2.replace('PLACEHOLDER', h) for h in FSRsys]
            #if 'JetCorrections' in key1:
            #    cmd_list = [cmd2.replace(path_in, path_in2)]

            #if debug:
            #    print cmd_list

            #for c in cmd_list:
                #print_tmp = c.replace(tmp_nfs, '')
                #print_cmd = print_tmp.replace(tmp_afs, '')
                #print print_cmd
                #os.system(c)


# ------------------------------------------------------------------------------
#

if __name__ == "__main__":
    MoveToAFS()