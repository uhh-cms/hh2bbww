# coding: utf-8

"""
NN trigger methods for HHtobbWW.
At the moment, just some tests
"""

from collections import defaultdict
from typing import Tuple
import yaml

from columnflow.util import maybe_import, dev_sandbox

from columnflow.columnar_util import set_ak_column, DotDict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer, producer

np = maybe_import("numpy")
ak = maybe_import("awkward")
keras = maybe_import("keras")
pickle = maybe_import("pickle")

# helper function for awkward -> numpy
# this only does a single object!
def getPadNParr(events, obj, n_pad, fields, cuts = None, name = None):

    objects = events[obj]

    if not name: name = obj

    # cuts are defined as a dictionary containing the relevant keys:
    # cuttype, field and value
    if cuts:
        for cut in cuts:
            if cut["cuttype"] == "equals": objects = objects[objects[cut["field"]] == cut["value"]]
            else: raise Exception("Cuttype {} is not implemented.".format(cut["cuttype"]))

    pad_arrs = []
    var_names = []

    # padding with nones
    pad_arr = ak.pad_none(objects, n_pad, clip=True)

    # combining to numpy
    for i in range(n_pad):

        for var in fields:
            pad_with = 0. # we could add a "pad-with" to each field later
            pad_arrs += [ak.to_numpy( ak.fill_none(pad_arr[var][:,i], pad_with) )]
            var_names.append( "{}_{}_{}".format(name, i, var) )

    return np.stack(pad_arrs), var_names


# this producer will run the DNN inference and will produce the L1NNscore output value
@producer(
    uses={
        "L1Jet.pt", "L1Jet.eta", "L1Jet.phi",
        "L1Mu.pt", "L1Mu.eta", "L1Mu.phi",
        "L1EG.pt", "L1EG.eta", "L1EG.phi", "L1EG.hwIso",
        "L1EtSum.pt", "L1EtSum.phi", "L1EtSum.etSumType",
    },
    produces={"L1NNscore"},
    exposed=True,
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_keras.sh"),
    # sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh"),
)
def NN_trigger_inference(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    # first, importing the network and all info
    # this will be three files:
    # - the model itself
    # - the standard scaler object
    # - some custom objects (optional)
    # all this info will be read based on our usual folder structure

    # objects = self.info["model_settings"]["objects"]
    objects = self.info["objects"]
    dataList = []
    for obj in objects:
        dat, names = getPadNParr(events, obj["key"], obj["n_obj"], obj["fields"], 
                                 obj["cuts"] if "cuts" in obj else None, obj["name"])
        dataList.append(dat)

    inputs = np.concatenate(dataList, axis=0).T

    # scaling the input vector
    scaled_inputs = self.scaler.transform(inputs)

    # running the inference
    predictions = self.model.predict(scaled_inputs).flatten()

    # returning the inference results
    events = set_ak_column(events, "L1NNscore", predictions)
    return events


# the following code assures that the required external files are loaded
@NN_trigger_inference.requires
def NN_trigger_inference_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs: return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


# set up the producer
@NN_trigger_inference.setup
def NN_trigger_inference_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: dict) -> None:
    bundle = reqs["external_files"]

    with open(bundle.files.L1NN_infos.path, 'r') as infile: self.info =yaml.safe_load(infile)

    # loading the model
    self.model = keras.models.load_model(bundle.files.L1NN_network.path)

    # loading the scaler
    with open(bundle.files.L1NN_scaler.path, 'rb') as inp: self.scaler = pickle.load(inp)


# this selector uses the L1NNscore column to make a L1 trigger cut
@selector(
    uses={NN_trigger_inference},
    exposed=True,
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_keras.sh"),
    produces={NN_trigger_inference},
)
def NN_trigger_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # this selector will perform a selection on the L1NNscore
    results = SelectionResult()
    # add L1NN score
    events = self[NN_trigger_inference](events, **kwargs)
    # Here we need a function that calculates the optimal threshold for a pure rate of 1 kHz
    # for now we can just use 0.987 or something like that
    threshold = 0.8351974487304688
    greater = True

    # defining some individual masks
    if (greater): L1NNscoremask = events.L1NNscore >= threshold
    else: L1NNscoremask = events.L1NNscore <= threshold

    results.steps["L1NNcut"] = L1NNscoremask
    # steps["L1NNcut"] = L1NNscoremask
    # results = SelectionResult(
    #     steps,
    # )

    return events, results
