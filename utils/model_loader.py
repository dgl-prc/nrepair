import os
import onnx
from utils.constant import PROJECT_PATH, DataType
from utils import constant
from repair.step2_repair.onnx_modify import pyonnx2masked_onnx, nnet2masked_onnx_acasxu
from utils.constant import ModelType
from utils.help_func import load_pickle
from utils.exceptions import UnsupportedDataType
import json
# from socrates.json_parser import parse_model
from models.fnn.fnn import *
from image_deeppoly.read_net_file import read_tensorflow_net


def load_model(model_path, mtype):
    if mtype == ModelType.ONNX:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    elif mtype == ModelType.SOCRATES:
        model = load_pickle(model_path)
    else:
        raise UnsupportedDataType(mtype)
    return model


def load_ori_model(dataset, mid, mtype):
    if mtype == ModelType.ONNX:
        onnx_path = getattr(constant, f"{dataset.upper()}_MODEL_PATH").format(mid)
        model_path = os.path.join(PROJECT_PATH, onnx_path)
        model = load_model(model_path, mtype)
    else:
        # js_model_path = f"/home/dgl/project/nn_repair/coeff_based_repair/converted_models/{dataset}_{mid}"
        # with open(os.path.join(js_model_path, "spec.json"), 'r') as f:
        #     spec = json.load(f)
        # model = parse_model(spec['model'])
        pass
    return model


def load_ori_onnx(dataset, mid):
    onnx_path = getattr(constant, f"{dataset.upper()}_MODEL_PATH").format(mid)
    onnx_model = onnx.load(os.path.join(PROJECT_PATH, onnx_path))
    onnx.checker.check_model(onnx_model)
    return onnx_model


def load_text_pyt(dataset, mid):
    is_trained_with_pytorch = True
    assert dataset != DataType.ACASXU
    num_pixels = 784 if dataset == DataType.MNIST else 3072
    pyt_path = os.path.join(PROJECT_PATH, getattr(constant, f"{dataset.upper()}_PYMODEL_PATH").format(mid))
    model, is_conv, means, stds = read_tensorflow_net(pyt_path, num_pixels, is_trained_with_pytorch, False)
    return model


def init_pyt_model(datatype, num_layer, hidden_size):
    if datatype == DataType.MNIST:
        model = MnistNet(num_layer, hidden_size)
    elif datatype == DataType.CIFAR10:
        model = Cifar10Net(num_layer, hidden_size)
    else:
        model = ACASXu()
    return model


def load_acasxu_repaired(spec, mid, epsilon, timeid,repair_type):
    """ prepare the path of each repaired model in each interval.
    Args:
        spec:
        mid:
        epsilon:
        timeid:

    Returns:

    """
    # "/home/dgl/project/nn_repair/data/acasxu/repaired_onnx/MM/n_2_1/property_2/interval-11/epsilon_inf/20210724234614"
    paths = []
    # folder = os.path.join(PROJECT_PATH, f"data/acasxu/repaired_onnx/n_{mid}/property_{spec}/")
    folder = os.path.join(PROJECT_PATH, f"data/acasxu/repaired_onnx/{repair_type}/n_{mid}/property_{spec}/")
    for interval_name in os.listdir(folder):
        repaired_onnx_path = os.path.join(folder, f"{interval_name}/epsilon_{epsilon}/{timeid}/repaired.onnx")
        interval_id = int(interval_name.split("-")[1])
        if os.path.exists(repaired_onnx_path):
            paths.append((interval_id - 1, repaired_onnx_path))
    return paths


# def load_p2_repaired():
#     mids = [f"{aid}_{tid}" for aid in range(2, 6) for tid in range(1, 10)]
#     mids.remove("3_3")
#     mids.remove("4_2")
#     folder = os.path.join(PROJECT_PATH, "data/acasxu/repaired_onnx/property_2/n_{}")
#     onnx_paths = {}
#     for mid in mids:
#         onnx_folder = folder.format(mid)
#         timeid = os.listdir(onnx_folder)[0]
#         actives = os.listdir(os.path.join(onnx_folder, timeid))
#         paths = []
#         for active in actives:
#             files_names = os.listdir(os.path.join(onnx_folder, timeid, active))
#             paths.extend([os.path.join(onnx_folder, timeid, active, file_name) for file_name in files_names])
#         onnx_paths[mid] = paths
#     return onnx_paths
#
#
# def load_p8_repaired():
#     mid = "2_9"
#     paths = []
#     folder = os.path.join(PROJECT_PATH, "data/acasxu/repaired_onnx/property_8/n_2_9/20200819233200")
#     for active in os.listdir(folder):
#         files_names = os.listdir(os.path.join(folder, active))
#         paths.extend([os.path.join(folder, active, file_name) for file_name in files_names])
#     return {mid: paths}
#
#
# def load_p7_repaired(epsilon, timeid):
#     mid = "1_9"
#     paths = []
#     folder = os.path.join(PROJECT_PATH, f"data/acasxu/repaired_onnx/n_{mid}/property_7/")
#     for interval_name in os.listdir(folder):
#         repaired_onnx_path = os.path.join(folder, f"{interval_name}/epsilon_{epsilon}/{timeid}/repaired.onnx")
#         interval_id = int(interval_name.split("-")[1])
#         paths.append((interval_id - 1, repaired_onnx_path))
#     return {mid: paths}


def load_mask_onnx(dataset, net_id):
    if dataset == DataType.ACASXU:
        net_name = f"ACASXU_run2a_{net_id}_batch_2000.nnet"
        nnet_file = os.path.join(PROJECT_PATH, f"data/acasxu/nnet/{net_name}")
        msk_onnx_model = nnet2masked_onnx_acasxu(nnet_file)
    elif dataset in [DataType.MNIST, DataType.CIFAR10]:
        ori_onnx = load_ori_onnx(dataset, net_id)
        msk_onnx_model = pyonnx2masked_onnx(ori_onnx)
    return msk_onnx_model
