# coding: utf-8
"""
Producer to evaluate external transformer model.
"""


from __future__ import annotations

import os
import sys

import law
from columnflow.types import Any
from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import set_ak_column
from hbw.production.prepare_objects import prepare_objects
from hbw.util import timeit_multiple, log_memory


np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


@producer(
    uses={
        prepare_objects,
        "GenPart.{pt,eta,phi,mass,status,statusFlags,pdgId}",
        "Generator.{id1,id2,x1,x2,weight}",
    },
    produces={"nlo_dy_reweight_weight"},  # , "nlo_dy_reweight_weight_up", "nlo_dy_reweight_weight_down"},
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_onnx.sh"),
)
@timeit_multiple
def nlo_reweighting(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to evaluate external transformer model.
    """
    if "dy" not in self.dataset_inst.name:
        raise ValueError(f"nlo_reweighting should only run for DY samples, not for {self.dataset_inst.name}.")
    
    log_memory("Start")
    # sampleNumDict = {
    #     "dy_ee_m50toinf_amcatnlo": 181,
    #     "dy_mumu_m50toinf_amcatnlo": 182,
    #     "dy_tautau_m50toinf_amcatnlo": 183,
    #     # "dy_ee_m10to50_amcatnlo",  TODO: check how to handle 10to50 DY
    #     # "dy_mumu_m10to50_amcatnlo",
    #     # "dy_tautau_m10to50_amcatnlo",
    #     "dy_ee_m50toinf_0j_amcatnlo": 201,
    #     "dy_mumu_m50toinf_0j_amcatnlo": 204,
    #     "dy_tautau_m50toinf_0j_amcatnlo": 207,
    #     "dy_ee_m50toinf_1j_amcatnlo": 202,
    #     "dy_mumu_m50toinf_1j_amcatnlo": 205,
    #     "dy_tautau_m50toinf_1j_amcatnlo": 208,
    #     "dy_ee_m50toinf_2j_amcatnlo": 203,
    #     "dy_mumu_m50toinf_2j_amcatnlo": 206,
    #     "dy_tautau_m50toinf_2j_amcatnlo": 209,
    # }

    # make custom python bindings available TODO: maybe this can also be handled by the sandbox?
    sys.path.append("/data/dust/user/letzerba/public/hh2bbww/models/negativeReweight.cpython-39-x86_64-linux-gnu.so")
    import negativeReweight as nloReweight

    logger.info(f"Running reweight over {len(events)} events for sample.")

    Gen_pt = events.GenPart.pt
    nGenPart = ak.count(events.GenPart.pt, axis=-1)
    Gen_pt = ak.flatten(Gen_pt)

    Gen_eta = ak.flatten(events.GenPart.eta)
    Gen_phi = ak.flatten(events.GenPart.phi)
    Gen_mass = ak.flatten(events.GenPart.mass)
    Gen_status = ak.flatten(events.GenPart.status)
    Gen_status_flag = ak.flatten(events.GenPart.statusFlags)
    Gen_pdg = ak.flatten(events.GenPart.pdgId)

    Generator_id1 = events.Generator.id1
    Generator_id2 = events.Generator.id2

    Generator_x1 = events.Generator.x1
    Generator_x2 = events.Generator.x2

    baseReweightSign = events.Generator.weight
    baseReweightSign = np.where(baseReweightSign != 0, baseReweightSign / abs(baseReweightSign), 1)

    # if self.dataset_inst.name not in sampleNumDict:
    #     raise NotImplementedError(f"Sample {self.dataset_inst.name} not found in sampleNumDict for nlo reweighting.")

    # sampleNum = sampleNumDict[self.dataset_inst.name]
    # sampleNum = sampleNum * np.ones_like(Generator_id1)
    sampleName = "DYto"  # TODO: needs to be generalized if we want to reweight also other samples than DY

    def transform(data, sub, mul):
        data = (data - sub) * mul
        data = np.clip(data, -5, 5)
        return data

    # Convert awkward arrays to numpy and ensure correct dtypes
    Gen_pt_np = ak.to_numpy(Gen_pt).astype(np.float32)
    Gen_eta_np = ak.to_numpy(Gen_eta).astype(np.float32)
    Gen_phi_np = ak.to_numpy(Gen_phi).astype(np.float32)
    Gen_mass_np = ak.to_numpy(Gen_mass).astype(np.float32)
    Gen_status_np = ak.to_numpy(Gen_status).astype(np.int32)
    Gen_status_flag_np = ak.to_numpy(Gen_status_flag).astype(np.int32)
    Gen_pdg_np = ak.to_numpy(Gen_pdg).astype(np.int32)
    nGenPart_np = ak.to_numpy(nGenPart).astype(np.int32)

    Generator_x1_np = ak.to_numpy(Generator_x1).astype(np.float32)
    Generator_x2_np = ak.to_numpy(Generator_x2).astype(np.float32)
    Generator_id1_np = ak.to_numpy(Generator_id1).astype(np.int32)
    Generator_id2_np = ak.to_numpy(Generator_id2).astype(np.int32)

    extracted = nloReweight.extract_particles(
        Gen_pt_np,
        Gen_eta_np,
        Gen_phi_np,
        Gen_mass_np,
        Gen_status_np,
        Gen_status_flag_np,
        Gen_pdg_np,
        nGenPart_np,
        Generator_x1_np,
        Generator_x2_np,
        Generator_id1_np,
        Generator_id2_np,
    )

    n_total = extracted.size
    n_events = n_total // 70
    extracted = extracted.reshape(n_events, 7, 10)
    allPartPx = extracted[:, :, 0]
    allPartPy = extracted[:, :, 1]
    allPartPz = extracted[:, :, 2]
    allPartE = extracted[:, :, 3]
    allPartPdgAbs = np.abs(extracted[:, :, 4])
    allPartMask = extracted[:, :, 5]
    allPartPhi = extracted[:, :, 6]
    allPartEta = extracted[:, :, 7]
    allPartPt = extracted[:, :, 8]
    allPartFlag = extracted[:, :, 9]

    # helper for dR
    def dPhi(phi0, phi1):
        dphi = np.abs(phi0 - phi1)
        return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    def dEta(eta0, eta1):
        return np.abs(eta0 - eta1)
    def dR(eta0, phi0, eta1, phi1):
        return np.sqrt(dPhi(phi0, phi1)**2 + dEta(eta0, eta1)**2)

    # nJets: number of parton-shower jets = sum(mask) - 3 (subtract beams + V)
    nJets = np.sum(allPartMask, axis=1).astype(np.float32) - 3.0
    nJets = np.expand_dims(np.where(nJets < 0.0, 0.0, nJets), 1)
    VType = np.ones_like(nJets) * 2.0

    jet0_eta = allPartEta[:, 0]
    jet1_eta = allPartEta[:, 1]
    jet0_phi = allPartPhi[:, 0]
    jet1_phi = allPartPhi[:, 1]
    psDR = np.expand_dims(dR(jet0_eta, jet0_phi, jet1_eta, jet1_phi).astype(np.float32), 1)

    allParticlesMass = np.log(np.sqrt(np.where(
        allPartFlag == 1,
        allPartE * allPartE - allPartPt * allPartPt - allPartPz * allPartPz,
        0.01,
    )))

    isW = np.where((allPartFlag == 1) & ("Wto" in sampleName), 1.0, 0.0)
    isZ = np.where((allPartFlag == 1) & ("DYto" in sampleName or "Zto" in sampleName), 1.0, 0.0)
    allParticlesDeltaMass = np.where(
        isW,
        allParticlesMass - np.log(80),
        np.where(isZ, allParticlesMass - np.log(91), 0),
    )

    pf_features = {
        "allParticles_pt_log": np.expand_dims(np.log(extracted[:, :, 8] + 10), 1),
        "allParticles_energy_log": np.expand_dims(np.log(extracted[:, :, 3] + 10), 1),
        "allParticles_delta_mass": np.expand_dims(allParticlesDeltaMass, 1),
        "allParticles_mass": np.expand_dims(allParticlesMass, 1),
        "isW": np.expand_dims(isW, 1),
        "isZ": np.expand_dims(isZ, 1),
        "isQuark": np.expand_dims(np.where((allPartPdgAbs > 0) & (allPartPdgAbs < 7), 1.0, 0.0), 1),
        "isGluon": np.expand_dims(np.where(allPartPdgAbs == 21, 1.0, 0.0), 1),
        "isFirstGeneration": np.expand_dims(np.where((allPartPdgAbs < 20) & (((allPartPdgAbs % 10) == 1) | ((allPartPdgAbs % 10) == 2)), 1.0, 0.0), 1),  # noqa: E501
        "isSecondGeneration": np.expand_dims(np.where((allPartPdgAbs < 20) & (((allPartPdgAbs % 10) == 3) | ((allPartPdgAbs % 10) == 4)), 1.0, 0.0), 1),  # noqa: E501
        "isThirdGeneration": np.expand_dims(np.where((allPartPdgAbs < 20) & (((allPartPdgAbs % 10) == 5) | ((allPartPdgAbs % 10) == 6)), 1.0, 0.0), 1),  # noqa: E501
        "isUpType": np.expand_dims(np.where((allPartPdgAbs > 0) & (allPartPdgAbs < 7) & ((allPartPdgAbs % 2) == 0), 1.0, 0.0), 1),  # noqa: E501
        "isDownType": np.expand_dims(np.where((allPartPdgAbs > 0) & (allPartPdgAbs < 7) & ((allPartPdgAbs % 2) == 1), 1.0, 0.0), 1),  # noqa: E501
        "isBeam": np.expand_dims(np.where(allPartFlag == 0, 1.0, 0.0), 1),
        "isPS": np.expand_dims(np.where(allPartFlag == 2, 1.0, 0.0), 1),
    }
    pf_vectors = {
        "allParticles_px": np.expand_dims(allPartPx, 1),
        "allParticles_py": np.expand_dims(allPartPy, 1),
        "allParticles_pz": np.expand_dims(allPartPz, 1),
        "allParticles_energy": np.expand_dims(allPartE, 1),
    }
    pf_mask = np.expand_dims(allPartMask, 1).astype(np.float32)

    pf_features = np.concatenate([
        transform(pf_features["allParticles_pt_log"], 1.7, 0.7),
        transform(pf_features["allParticles_energy_log"], 2.0, 0.7),
        transform(pf_features["allParticles_mass"], 4, 1),
        transform(pf_features["allParticles_delta_mass"], 0, 1),
        transform(pf_features["isW"], 0, 1),
        transform(pf_features["isZ"], 0, 1),
        transform(pf_features["isQuark"], 0, 1),
        transform(pf_features["isGluon"], 0, 1),
        transform(pf_features["isFirstGeneration"], 0, 1),
        transform(pf_features["isSecondGeneration"], 0, 1),
        transform(pf_features["isThirdGeneration"], 0, 1),
        transform(pf_features["isUpType"], 0, 1),
        transform(pf_features["isDownType"], 0, 1),
        transform(pf_features["isPS"], 0, 1),
    ], axis=1).astype(np.float32)

    pf_vectors = np.concatenate([
        transform(pf_vectors["allParticles_px"], 0, 0.0001),
        transform(pf_vectors["allParticles_py"], 0, 0.0001),
        transform(pf_vectors["allParticles_pz"], 0, 0.0001),
        transform(pf_vectors["allParticles_energy"], 0, 0.0001),
    ], axis=1).astype(np.float32)

    numEvents = pf_features.shape[0]
    results = []
    batchSize = 64  # TODO: could be decoded from model name
    startIndex = 0
    log_memory("Start of inference loop")
    while (startIndex < numEvents):
        endIndex = min(startIndex + batchSize, numEvents)
        padSize = batchSize - (endIndex - startIndex)

        if (padSize > 0):
            pf_features_pad = np.concatenate(
                [pf_features[startIndex:endIndex, :, :], np.zeros([padSize, 14, 7])], axis=0, dtype=np.float32,
            )
            pf_vectors_pad = np.concatenate(
                [pf_vectors[startIndex:endIndex, :, :], np.zeros([padSize, 4, 7])], axis=0, dtype=np.float32,
            )
            pf_mask_pad = np.concatenate(
                [pf_mask[startIndex:endIndex, :, :], np.zeros([padSize, 1, 7])], axis=0, dtype=np.float32,
            )

            nJets_pad = np.concatenate([nJets[startIndex:endIndex], np.zeros([padSize, 1])], axis=0, dtype=np.float32)
            psDR_pad = np.concatenate([psDR[startIndex:endIndex], np.zeros([padSize, 1])], axis=0, dtype=np.float32)
            VType_pad = np.concatenate([VType[startIndex:endIndex], np.zeros([padSize, 1])], axis=0, dtype=np.float32)

            result = self.session.run(
                None, {
                    "pf_features": pf_features_pad,
                    "pf_vectors": pf_vectors_pad,
                    "pf_mask": pf_mask_pad,
                    "nJets": nJets_pad,
                    "psDR": psDR_pad,
                    "VType": VType_pad,
                })
            resultTrimmed = [np.expand_dims(res[:(endIndex - startIndex)], axis=1) for res in result]

            pPos = np.concatenate(resultTrimmed, axis=1)
        else:
            result = self.session.run(
                None, {
                    "pf_features": pf_features[startIndex:endIndex, :, :],
                    "pf_vectors": pf_vectors[startIndex:endIndex, :, :],
                    "pf_mask": pf_mask[startIndex:endIndex, :, :],
                    "nJets": nJets[startIndex:endIndex],
                    "psDR": psDR[startIndex:endIndex],
                    "VType": VType[startIndex:endIndex],
                })
            resultTrimmed = [np.expand_dims(res[:], axis=1) for res in result]
            pPos = np.concatenate(resultTrimmed, axis=1)

        results.append(pPos)

        startIndex += batchSize

    if (numEvents == 0):
        preds = np.zeros((0, 20, 14))
    else:
        preds = np.concatenate(results, axis=0)

    preds = np.where(np.isnan(preds), 0, preds)

    # Format: preds is expected to be (n_events, n_models, n_systematics)
    # We only consider the nominal systematic (index 0).
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 2):  # TODO: I think this doesn't aply for us?
        # Legacy behavior: (events*2,) or (events,2) -> keep old semantics
        if preds.ndim == 1:
            preds = preds.reshape((len(preds) // 2, 2))
        means = preds[:, 0]
        uncerts = preds[:, 1]
        # transform probabilities in [0,1] to [-1,1] like before
        nominalReweight = baseReweightSign * (means * 2.0 - 1.0)
        reweightUp = baseReweightSign * ((means + uncerts) * 2.0 - 1.0)
        reweightDown = baseReweightSign * ((means - uncerts) * 2.0 - 1.0)
    else:
        # Modern behavior: (events, n_models, n_systematics)(<--NOTE: this happenens for us) or (events, n_models)
        models = preds * 2.0 - 1.0
        if preds.ndim == 3:
            models_t = models[:, :, 0]  # take nominal systematic (index 0)
        elif preds.ndim == 2:
            models_t = models  # already (events, n_models)
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        # means and uncerts are the mean across the n_models (20)
        means = np.mean(models_t, axis=1).astype(float)
        uncerts = np.mean(models_t, axis=1).astype(float)

        # nominalReweight is the sign of the original generator weight
        nominalReweight = baseReweightSign * means

        # For up/down we keep the individual model predictions (10 each)
        nmodels = 20  # models_t.shape[1]
        half = nmodels // 2

        reweightUp = (baseReweightSign[:, None] * models_t[:, :half])
        reweightDown = (baseReweightSign[:, None] * models_t[:, half:half + half])
        reweightUp = np.concatenate([reweightUp, np.ones_like(reweightUp[:, :1])], axis=1)  # add nominal as last column
        reweightDown = np.concatenate([reweightDown, np.ones_like(reweightDown[:, :1])], axis=1)  # add nominal as last column  # noqa: E501

    # set reweight column
    events = set_ak_column(events, "nlo_dy_reweight_weight", nominalReweight)
    # events = set_ak_column(events, "nlo_dy_reweight_weight_up", reweightUp)
    # events = set_ak_column(events, "nlo_dy_reweight_weight_down", reweightDown)
    log_memory("End of inference loop and column setting")

    return events


@nlo_reweighting.requires
def nlo_reweighting_requires(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@nlo_reweighting.setup
def nlo_reweighting_setup(self: Producer, reqs, **kwargs) -> None:
    """
    Initialize the transformer model and set up the session for inference.
    """
    log_memory("Before loading model")
    # load the model
    import onnxruntime as rt

    # ensure CPU-only
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # setup rt session
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CPUExecutionProvider"]

    model = reqs["external_files"].files.nlo_reweight_model.fn

    self.session = rt.InferenceSession(model, sess_options, providers=providers)

    log_memory("After loading model")

    # Get input details
    for inp in self.session.get_inputs():
        print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")


@nlo_reweighting.init
def nlo_reweighting_init(self: Producer, **kwargs) -> None:

    return None


@nlo_reweighting.teardown
def nlo_reweighting_teardown(self: Producer, **kwargs) -> None:
    """
    Stops the model session.
    """
    delattr(self, "session")
    log_memory("After deleting model session")
