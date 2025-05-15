# coding: utf-8
"""

"""

from hbw.ml.dense_classifier import DenseClassifier
from hbw.util import build_param_product


weights = lambda bkg_weight: {
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1": 1,
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1": 1,
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": 1,
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
    "tt": bkg_weight,
    "st": bkg_weight,
    "v_lep": bkg_weight,
    "w_lnu": bkg_weight,
    "dy": bkg_weight,
}

example_grid_search = {  # 4*2*2*1*3*3*1 = 144 trainings
    "layers": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)],
    "learningrate": [0.00500, 0.00050],
    "negative_weights": ["abs", "handle"],
    "epochs": [300],
    "batchsize": [1024, 2048, 4096],
    "dropout": [0.1, 0.3, 0.5],
    "sub_process_class_factors": [weights(1)],  # weighting should not change AUCs, so optimize it separately
}

param_product = build_param_product(example_grid_search, lambda i: f"dense_gridsearch_{i}")

# to use these derived models, include this file in the law.cfg (ml_modules)
for model_name, params in param_product.items():
    dense_model = DenseClassifier.derive(model_name, cls_dict=params)

# store model names as tuple to be exportable for scripts
grid_search_models = tuple(param_product.keys())
