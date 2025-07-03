# coding: utf-8
"""
USAGE:
python my_get_das_info.py -c smart -d <das_string>
e.g. /JetHT/Run2018C-UL2018_MiniAODv2_JMENanoAODv9-v1/NANOAOD

The '-c smart' option tries to determine the dataset and process name based on the name of the dataset.
The automated process is highly susceptible to changes in the dataset naming conventions and might
easily produce wrong results.

The -d option can be a single das string (wildcards are possible via '*') or a list of das strings.
If a das string ends with ".txt", the file is read and the das strings are extracted from the file.
The file should contain one das string per line.

"""

from __future__ import annotations

import subprocess
import json
from argparse import ArgumentParser


def get_generator_name(name: str) -> str:
    """
    Function that returns the generator name of a dataset
    """
    if "powheg" in name:
        return "_powheg"
    elif "madgraph" in name:
        return "_madgraph"
    elif "amcatnlo" in name:
        return "_amcatnlo"
    elif "pythia" in name:
        return "_pythia"
    else:
        return ""


def get_broken_files_str(data: dict, n_spaces: int = 20) -> str:
    """
    Function that returns a string represenatation of broken files
    """

    broken_files_list = [
        f'"{d}",  # broken  # noqa: E501' for d in data["broken_files"]
    ] + [
        f'"{d}",  # empty  # noqa: E501' for d in data["empty_files"] if d not in data["broken_files"]
    ]

    if not broken_files_list:
        return ""
    else:
        return (
            f"\n{' ' * n_spaces}" +
            f"\n{' ' * n_spaces}".join(broken_files_list) +
            f"\n{' ' * (n_spaces - 4)}"
        )


def convert_data(data: dict, placeholder="PLACEHOLDER") -> str:
    """
    Function that converts dataset info into one order Dataset per query
    """
    if not data["name"].endswith("AOD"):
        return f"""{data['name']} is not a data dataset!"""

    try:
        era = data["name"].split("/")[2].split("-")[0].split("Run")[1][-1]
    except Exception:
        era = "ERA"
    try:
        name = "data_" + data["name"].split("/")[1].lower()
    except Exception:
        name = f"data_{placeholder}"

    return f"""cpn.add_dataset(
    name="{name}_{era}",
    id={data['dataset_id']},
    processes=[procs.{name}],
    keys=[
        "{data['name']}",  # noqa
    ],
    n_files={data['nfiles']},
    n_events={data['nevents']},
    is_data=True,
    aux={{
        "era": "{era}",
    }}
)
"""


def convert_default(data: dict, placeholder="PLACEHOLDER") -> str:
    """
    Function that converts dataset info into one order Dataset per query
    """
    generator = get_generator_name(data["name"])
    shift = "extension" if "ext" in data["name"] else "nominal"
    return f"""cpn.add_dataset(
    name="{placeholder}{generator}",
    id={data['dataset_id']},
    processes=[procs.{placeholder}],
    info=dict(
        {shift}=DatasetInfo(
            keys=[
                "{data['name']}",  # noqa: E501
            ],
            aux={{
                "broken_files": [{get_broken_files_str(data)}],
            }},
            n_files={data['nfiles_good']},  # {data["nfiles"]}-{data["nfiles_bad"]}
            n_events={data['nevents']},
        ),
    ),
)
"""


def convert_variation(data: dict, placeholder="PLACEHOLDER") -> str:
    """
    Function that converts dataset info into one order Dataset per query. Stores the dataset info
    in a dict with the dataset type as key.
    """
    generator = get_generator_name(data["name"])
    return f"""cpn.add_dataset(
    name="{placeholder}{generator}",
    id={data['dataset_id']},
    processes=[procs.{placeholder}],
    info=dict(
        nominal=DatasetInfo(
            keys=[
                "{data['name']}",  # noqa
            ],
            n_files={data['nfiles']},
            n_events={data['nevents']},
        ),
    ),
)
"""


identifier_map = {
    "_CP5TuneDown_": "tune_down",
    "_CP5TuneUp_": "tune_up",
    "_TuneCP5Down_": "tune_down",
    "_TuneCP5Up_": "tune_up",
    "_TuneCP5CR1_": "cr_1",
    "_TuneCP5CR2_": "cr_2",
    "_Hdamp-158_": "hdamp_down",
    "_Hdamp-418_": "hdamp_up",
    "_MT-171p5_": "mtop_down",
    "_MT-173p5_": "mtop_up",
    # "_withDipoleRecoil_": "with_dipole_recoil",
    "_dipoleRecoilOn_": "dipole_recoil_on",
    # dataset types that I have no use for but want to keep anyways
    # "_MT-166p5_": "comment",
    # "_MT-169p5_": "comment",
    # "_MT-175p5_": "comment",
    # "_MT-178p5_": "comment",
    # "_DS_TuneCP5_": "comment",
    # "_TuneCP5_ERDOn_": "comment",
    # "_TuneCH3_": "comment",
    # dataset types that I want to skip completely
    "_MT-166p5_": "ignore",
    "_MT-169p5_": "ignore",
    "_MT-175p5_": "ignore",
    "_MT-178p5_": "ignore",
    "_DS_TuneCP5_": "ignore",
    "_TuneCP5_ERDOn_": "ignore",
    "_TuneCH3_": "ignore",
    # "example_key": "ignore",
    # nominal entry as the last one such that other dataset types get priority
    # "ext1": "extension",
    "_TuneCP5_": "nominal",
    "_CP5_": "nominal",
}


def higgs_decay(part: str):
    part = part.lower()
    outp = "h"
    if "non" in part:
        outp += "non"
        part = part.replace("non", "")

    if "hto2" in part:
        decay_modes = part.split("hto2")[1].lower().split("to")
        decay_modes[0] = decay_modes[0].replace("mu", "m")[0] * 2
    elif "hto" in part:
        decay_modes = part.split("hto")[1].lower().split("to")
        decay_modes[0] = decay_modes[0].replace("mu", "m")[0:3]
    else:
        print(f"Could not determine decay mode for {part}")
        decay_modes = ["XX"]
    outp += decay_modes[0]
    if len(decay_modes) > 1:
        outp += decay_modes[1]
    return outp


def vv_decay(part: str):
    parts = [p.lower() for p in part.split("-")[0].split("to")]
    if len(parts) == 1:
        parts[0] = {
            "lnu2q": "lnu2q",
            "2l2nu": "2l2nu",
            "4q": "4q",
            "4l": "4l",
            "3lnu": "3lnu",
            "2l2q": "2l2q",
            "2l2nu": "2l2nu",
            "4nu": "4nu",
        }.get(parts[1], parts[1])
    # if (len(parts) > 1) & (parts[0] == "ww"):
    #     parts[0] = {
    #         # WW
    #         "lnu2q": "sl",
    #         "2l2nu": "dl",
    #         "4q": "fh",
    #     }.get(parts[1], parts[1])
    # elif len(parts) > 1:
    #     parts[1] = {
    #         # WW
    #         "lnu2q": "lnuqq",
    #         "2l2nu": "lnulnu",
    #         "4q": "qqqq",
    #         # ZZ
    #         "4l": "llll",
    #         "3lnu": "lllnu",
    #         "2l2q": "llqq",
    #       }.get(parts[1], parts[1])

    return "_".join(parts).replace("/", "")


def hh_decay(part: str):
    decay = part.split("HHto")[1]
    parts = [p.lower() for p in part.split("-")[0].split("to")]

    parts = decay.split("to")
    hh_decay = parts[0].replace("2", "").lower().replace("w", "v")
    hh_decay = "".join(f"_h{h_decay.lower() * 2}" for h_decay in hh_decay)

    sub_parts = {
        "2l2nu": "2l2nu",
        "lnu2j": "qqlnu",
        "lnu2q": "qqlnu",
    }.get(parts[-1].lower(), "")

    return hh_decay + sub_parts


name_identifier = {
    # Data
    "/EGamma": lambda part: "data_egamma_{era}",
    "/MuonEG": lambda part: "data_muoneg_{era}",
    "/Muon": lambda part: "data_mu_{era}",
    # TTbar
    "/TTtoLNu2Q": lambda part: "tt_sl",
    "/TTto2L2Nu": lambda part: "tt_dl",
    "/TTto4Q": lambda part: "tt_fh",
    # Single t
    "/TBbarQ-t-channel": lambda part: "st_tchannel_t",  # NOTE: undersore before t-channel in DAS
    "/TbarBQ-t-channel": lambda part: "st_tchannel_tbar",  # NOTE: undersore before t-channel in DAS
    "/TWminustoLNu2Q": lambda part: "st_twchannel_t_sl",
    "/TbarWplustoLNu2Q": lambda part: "st_twchannel_tbar_sl",
    "/TWminusto2L2Nu": lambda part: "st_twchannel_t_dl",
    "/TbarWplusto2L2Nu": lambda part: "st_twchannel_tbar_dl",
    "/TWminusto4Q": lambda part: "st_twchannel_t_fh",
    "/TbarWplusto4Q": lambda part: "st_twchannel_tbar_fh",
    "/TBbartoLplusNuBbar-s-channel": lambda part: "st_schannel_t_lep" + "_4f" if "4FS" in part else "",
    "/TbarBtoLminusNuB-s-channel": lambda part: "st_schannel_tbar_lep" + "_4f" if "4FS" in part else "",
    # DY
    "/DYto2L-": lambda part: "dy",
    # W + jets
    "/WtoLNu": lambda part: "w_lnu",
    "/ZZZ": lambda part: "zzz",
    "/WZZ": lambda part: "wzz",
    "/WWZ": lambda part: "wwz",
    "/WWW": lambda part: "www",
    "/WW": vv_decay,
    "/WZ": vv_decay,
    "/ZZ": vv_decay,
    # Di-Higgs
    "/VBFHH": lambda part: "hh_vbf" + hh_decay(part),
    "/GluGlutoHH": lambda part: "hh_ggf" + hh_decay(part),
    # Higgs
    "/GluGluH": lambda part: "h_ggf" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "h_ggf",
    "/VBFH": lambda part: "h_vbf" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "h_vbf",
    "/ggZH": lambda part: "zh_gg" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "zh_gg",
    "/ZH": lambda part: "zh" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "zh",
    "/WminusH": lambda part: "wmh" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "wmh",
    "/WplusH": lambda part: "wph" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "wph",
    "/TTZH": lambda part: "ttzh",
    "/TTWH": lambda part: "ttwh",
    "/TTH": lambda part: "tth" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "tth",
    "/BBH": lambda part: "bbh" + f"_{higgs_decay(part)}" if "hto" in part.lower() else "bbh",
    # ttV(V)
    "/TTZZ": lambda part: "ttzz",
    "/TTWZ": lambda part: "ttwz",
    "/TTWW": lambda part: "ttww",
    "/TTZ": lambda part: "ttz",
    "/TTW": lambda part: "ttw",
    "/TTLL": lambda part: "ttz_zll",
    "/TTNuNu": lambda part: "ttz_znunu",
    "/QCD": lambda part: "qcd",
    "/THQ": lambda part: "thq",
    "/THW": lambda part: "thw",
    "/TTG": lambda part: "ttg",
    "/TGQB": lambda part: "tgqb",
    "/TTTT": lambda part: "tttt",
}

add_inf = lambda part: ("toinf" if "to" not in part else "")


def convert_parameter(part: str):
    """
    Function that converts a parameter string into a valid python variable name
    """
    # rename the parameters to the correct names
    part = part.replace("CV", "kv").replace("C2V", "k2v").replace("C3", "kl")
    # remove unnecessary parts of the parameter identifier
    part = part.replace("-", "").replace("p00", "")
    while part.endswith("0") and "p" in part:
        part = part[:-1]
    return part


additional_identifier = {
    "MuEnriched": lambda part: "mu",
    "bcToE": lambda part: "bctoe",
    "DoubleEMEnriched": lambda part: "doubleem",
    "EMEnriched": lambda part: "em",
    "4FS": lambda part: "4f",  # NOTE: this part is supposed to only be added in the dataset, but not in the process
    # "5FS": lambda part: "5f",
    "To3LNu": lambda part: "wlnu_zll",
    "To2L2Nu": lambda part: "dl",
    "ToLNu2Q": lambda part: "sl",
    "HtoZG": lambda part: "hzg",
    "WtoLNu": lambda part: "wlnu",
    "Wto2Q": lambda part: "wqq",
    "Zto2Q": lambda part: "zqq",
    "Zto2L": lambda part: "zll",
    "Zto2Nu": lambda part: "znunu",
    "ZtoLL": lambda part: "zll",
    "ZtoNuNu": lambda part: "znunu",
    "Hto2": higgs_decay,
    "To4Q": lambda part: "fh",
    "MLNu-": lambda part: part.replace("MLNu-", "mlnu") + add_inf(part),
    "MLL-": lambda part: part.replace("MLL-", "m") + add_inf(part),
    "NJet-": lambda part: part.replace("NJet-", "") + "j",
    "0J": lambda part: "0j",
    "1J": lambda part: "1j",
    "2J": lambda part: "2j",
    "3J": lambda part: "3j",
    "4J": lambda part: "4j",
    "PTLL-": lambda part: part.replace("PTLL-", "pt") + add_inf(part),
    "PTLNu-": lambda part: part.replace("PTLNu-", "ptlnu") + add_inf(part),
    "PTG-": lambda part: part.replace("PTG-", "pt") + add_inf(part),
    "PT-": lambda part: part.replace("PT-", "pt").lower() + add_inf(part),
    "HT-": lambda part: part.replace("HT-", "ht") + add_inf(part),
    "CV-": convert_parameter,
    "C2V-": convert_parameter,
    "C3-": convert_parameter,
    "kl-": convert_parameter,
    "kt-": convert_parameter,
    # "c2-": lambda part: part.replace("c2-", "c2").replace("p00", ""),
}


def swap_hdecays_and_k_params(input_string: str):
    """
    Ugly helper function that puts the k parameter substrings to the correct position in the string
    """
    parts = input_string.split("_")
    if parts[0] != "hh":
        return input_string

    for part in parts[::-1]:
        if part.startswith("k"):
            parts.remove(part)
            parts.insert(2, part)

    return "_".join(parts)


def fix_das_name(name):
    """
    Sometimes there are differences in naming conventions. They are fixed here (e.g. change a "_" to a "-")
    """
    for identifier in ["tchannel"]:
        name = name.replace(f"_{identifier}", f"-{identifier}")
    for identifier in ("CV", "C2V", "C3"):
        name = name.replace(f"_{identifier}_", f"_{identifier}-")

    return name


def convert_smart(data: dict) -> str:
    """
    Function that converts dataset info into one order Dataset per query. The function tries to
    determine the dataset and process name based on the name of the dataset.
    NOTE: this function is highly susceptible to changes in the dataset naming conventions and might
    easily produce wrong results.
    """
    name = fix_das_name(data["name"])
    name_components = name.split("_")
    proc_name = "UKNOWN"
    for identifier, dataset_type in name_identifier.items():
        if identifier in name_components[0]:
            proc_name = dataset_type(name_components[0])
            break

    for identifier, func in additional_identifier.items():
        for part in name_components[1:]:
            if identifier in part:
                proc_name += "_" + func(part)
                # add only one part per name_component
                break

    # proc_name = swap_hdecays_and_k_params(proc_name)

    # return convert_nanogen(data, placeholder=proc_name)
    return convert_default(data, placeholder=proc_name)


def convert_top(data: dict, placeholder="PLACEHOLDER") -> str:
    """
    Function that converts dataset info into either an order Datset for nominal datasets
    or to a DatasetInfo for variations of datasets such as tune or mtop.

    Exemplary usage:
    python get_das_info.py -c top -d "/TTtoLNu2Q*/Run3Summer22EENanoAODv12-130X_*/NANOAODSIM"
    """
    generator = get_generator_name(data["name"])
    dataset_type = None

    for identifier in identifier_map:
        if identifier in data["name"]:
            dataset_type = identifier_map[identifier]
            break

    if not dataset_type:
        return f"""
        #####
        #####ERROR! Did not manage to determine type of dataset {data['name']}
        #####
        """

    if dataset_type == "nominal":
        return f"""cpn.add_dataset(
    name="{placeholder}{generator}",
    id={data['dataset_id']},
    processes=[procs.{placeholder}],
    info=dict(
        nominal=DatasetInfo(
            keys=[
                "{data['name']}",  # noqa: E501
            ],
            aux={{
                "broken_files": [{get_broken_files_str(data)}],
            }},
            n_files={data['nfiles_good']},  # {data["nfiles"]}-{data["nfiles_bad"]}
            n_events={data['nevents']},
        ),
    ),
)"""
    elif dataset_type == "comment":
        # comment out this dataset
        return f"""        # {identifier}=DatasetInfo(
        #     keys=[
        #         "{data['name']}",  # noqa: E501
        #     ],
        #     aux={{
        #         "broken_files": [{get_broken_files_str(data)}],
        #     }},
        #     n_files={data['nfiles_good']},  # {data["nfiles"]}-{data["nfiles_bad"]}
        #     n_events={data['nevents']},
        # ),"""
    elif dataset_type == "ignore":
        return ""
    else:
        # some known variation of the dataset
        return f"""        {dataset_type}=DatasetInfo(
            keys=[
                "{data['name']}",  # noqa: E501
            ],
            aux={{
                "broken_files": [{get_broken_files_str(data)}],
            }},
            n_files={data['nfiles_good']},  # {data["nfiles"]}-{data["nfiles_bad"]}
            n_events={data['nevents']},
        ),"""


def nanogen_proc_name(data: dict, placeholder="PLACEHOLDER") -> str:
    name = fix_das_name(data["name"])
    name_components = name.split("_")
    proc_name = "UKNOWN"
    for identifier, dataset_type in name_identifier.items():
        if identifier in name_components[0]:
            proc_name = dataset_type(name_components[0])
            break

    for identifier, func in additional_identifier.items():
        for part in name_components[1:]:
            if identifier in part:
                proc_name += "_" + func(part)
                # add only one part per name_component
                break
    generator = get_generator_name(data["name"])
    proc_name += generator
    if "ext1" in data["name"]:
        # if the dataset is an extension, we add the extension to the name
        proc_name += "_ext1"
    return proc_name


def convert_nanogen(data: dict, placeholder="PLACEHOLDER") -> str:
    proc_name = nanogen_proc_name(data, placeholder)
    return f"""{proc_name}:
  key: {data['name']}
"""


def convert_keys(data: dict) -> str:
    """
    Function that only returns the dataset key.
    """
    return f"""{data['name']}"""


def convert_minimal(data: dict) -> str:
    """
    Function that only returns the dataset key + number of events.
    """
    # add underscores per 1000 events
    nevents_str = f"{data['nevents']:,}".replace(",", "_")
    nfiles_str = f"{data['nfiles']:,}".replace(",", "_")
    return f"""{data['name']}\nFiles: {nfiles_str}\nEvents: {nevents_str}\n"""


convert_functions = {
    "default": convert_default,
    "variation": convert_variation,
    "keys": convert_keys,
    "top": convert_top,
    "smart": convert_smart,
    "nanogen": convert_nanogen,
    "nanogen_proc_name": nanogen_proc_name,
    "minimal": convert_minimal,
    "data": convert_data,
}


def load_das_info(dataset: str, add_file_info: bool = False) -> dict:
    from law.util import interruptable_popen

    # call dasgoclient command
    cmd = f"dasgoclient -query='{'file ' if add_file_info else ''}dataset={dataset}' -json"
    code, out, _ = interruptable_popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        executable="/bin/bash",
    )
    if code != 0:
        raise Exception(f"dasgoclient query failed:\n{out}")
    infos = json.loads(out)

    return infos


def get_das_info(dataset: str) -> dict:
    info_of_interest = {"name": dataset}

    file_infos = load_das_info(dataset, add_file_info=True)

    info_of_interest["dataset_id"] = file_infos[0]["file"][0]["dataset_id"]

    empty_files_filter = lambda info: info["file"][0]["nevents"] == 0
    broken_files_filter = lambda info: info["file"][0]["is_file_valid"] == 0

    good_files = list(filter(lambda x: not broken_files_filter(x) and not empty_files_filter(x), file_infos))

    dataset_id = {info["file"][0]["dataset_id"] for info in good_files}
    if len(dataset_id) == 1:
        info_of_interest["dataset_id"] = dataset_id.pop()
    else:
        raise ValueError(f"Multiple dataset IDs ({dataset_id}) found for dataset {dataset}")

    info_of_interest["nfiles"] = len(file_infos)
    info_of_interest["nfiles_good"] = len(good_files)
    info_of_interest["nevents"] = sum(info["file"][0]["nevents"] for info in good_files)

    empty_files = [
        info["file"][0]["name"]
        for info in filter(empty_files_filter, file_infos)
    ]
    broken_files = [
        info["file"][0]["name"]
        for info in filter(broken_files_filter, file_infos)
    ]
    info_of_interest["empty_files"] = empty_files
    info_of_interest["broken_files"] = broken_files

    info_of_interest["nfiles_bad"] = len(set(empty_files + broken_files))

    return info_of_interest


def get_das_info_old(
    dataset: str,
) -> dict:
    from law.util import interruptable_popen

    # call dasgoclient command
    cmd = f"dasgoclient -query='dataset={dataset}' -json"
    code, out, _ = interruptable_popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        executable="/bin/bash",
    )
    if code != 0:
        raise Exception(f"dasgoclient query failed:\n{out}")
    infos = json.loads(out)
    info_of_interest = {"name": dataset}
    for info in infos:
        dataset_info = info["dataset"][0]
        # Get json format of single das_string gives multiple dictornaries with different info
        # Avoid to print multiple infos twice and ask specificly for the kew of interest
        if "dataset_info" in info["das"]["services"][0]:
            info_of_interest["dataset_id"] = dataset_info.get("dataset_id", "")
        elif "filesummaries" in info["das"]["services"][0]:
            info_of_interest["nfiles"] = dataset_info.get("nfiles", "")
            info_of_interest["nevents"] = dataset_info.get("nevents", "")

    return info_of_interest


def print_das_info(
    das_strings: list[str],
    keys_of_interest: tuple | None = None,
    convert_function_str: str | None = None,
):
    from law.util import interruptable_popen

    # get the requested convert function
    convert_function = convert_functions[convert_function_str]

    datasets = []

    for das_string in das_strings.copy():
        if das_string.endswith(".txt"):
            with open(das_string, "r") as f:
                # extract the das strings from the file and extend the list of datasets

                das_strings.extend([
                    das_str
                    for das_str in f.read().splitlines()
                    if not das_str.startswith("#")
                ])

            # remove the file from the list of datasets
            das_strings.remove(das_string)

    for das_string in das_strings:
        # set default keys of interest
        # NOTE: this attribute is currently not used
        keys_of_interest = keys_of_interest or (
            "name", "dataset_id", "nfiles", "nevents",
        )

        wildcard = "*" in das_string
        if not wildcard:
            # keep consisting structure
            datasets.append(das_string)
        else:
            # using a wildcard leads to a different structer in json format
            cmd = f"dasgoclient -query='dataset={das_string}' -json"
            code, out, _ = interruptable_popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                executable="/bin/bash",
            )
            if code != 0:
                raise Exception(f"dasgoclient query failed:\n{out}")
            infos = json.loads(out)
            for info in infos:
                dataset_name = info.get("dataset", [])[0].get("name", "")
                datasets.append(dataset_name)

    for dataset in datasets:
        info_of_interest = get_das_info(dataset)
        desired_output = convert_function(info_of_interest)
        if desired_output is not None:
            print(desired_output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", nargs="+", help="das names or txt file with das names")
    parser.add_argument(
        "-c",
        "--convert",
        dest="convert",
        help="function that converts info into code",
        default="smart",
        choices=list(convert_functions),
    )
    args = parser.parse_args()
    print_das_info(args.dataset, convert_function_str=args.convert)
