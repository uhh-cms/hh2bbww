# coding: utf-8
"""

"""

from hbw.ml.dense_classifier import DenseClassifier
from hbw.util import build_param_product


weights = lambda bkg_weight: {
    "ggHH_kl_1_kt_1_sl_hbbhww": 1,
    "ggHH_kl_1_kt_1_sl_hbbhww": 1,
    "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww": 1,
    "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww": 1,
    "tt": bkg_weight,
    "st": bkg_weight,
    "v_lep": bkg_weight,
    "w_lnu": bkg_weight,
    "dy_lep": bkg_weight,
}

example_grid_search = {  # 4*2*2*1*3*3*1 = 144 trainings
    "layers": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)],
    "learningrate": [0.00500, 0.00050],
    "negative_weights": ["abs", "handle"],
    "epochs": [300],
    "batchsize": [1024, 2048, 4096],
    "dropout": [0.1, 0.3, 0.5],
    "ml_process_weights": [weights(1)],  # weighting should not change AUCs, so optimize it separately
}

param_product = build_param_product(example_grid_search, lambda i: f"dense_gridsearch_{i}")

# to use these derived models, include this file in the law.cfg (ml_modules)
for model_name, params in param_product.items():
    dense_model = DenseClassifier.derive(model_name, cls_dict=params)

# store model names as tuple to be exportable for scripts
grid_search_models = tuple(param_product.keys())
