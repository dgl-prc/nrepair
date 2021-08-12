import sys
import os
import socket

cpu_affinity = os.sched_getaffinity(0)
server_name = socket.gethostname()
sys.path.append("../../")

if server_name == "Yserver":
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/python_interface/')
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/deepg/code/')
else:
    sys.path.insert(0, '../../env_setup/ELINA/python_interface/')
    sys.path.insert(0, '../../env_setup/ELINA/deepg/code/')

from repair.step1_ctex.acasxu_verify.split_input_spaces import *
from repair.step1_ctex.deeppoly_api import search_ctex_recursively
from utils.data_loader import *
from utils.help_func import save_pickle
from utils.time_util import *
import random


def get_fault_net_ids(spec_num):
    if spec_num == 1:
        net_ids = [(3, 9), (4, 7), (4, 9), (5, 8)]
    elif spec_num == 2:
        net_ids = [(i, j) for i in range(2, 6) for j in range(1, 10)]
        # These two can be verified.
        net_ids.remove((3, 3))
        net_ids.remove((4, 2))
    elif spec_num == 3:
        net_ids = [(1, 1)]
    elif spec_num == 7:
        net_ids = [(1, 9)]
    elif spec_num == 8:
        net_ids = [(2, 9)]
    return net_ids


def __parse_kwargs(spec_num, mid, kwargs):
    save_failed_space = True
    if "eran" not in kwargs:
        model = load_ori_onnx(DataType.ACASXU, mid)
        eran = ERAN(model, is_onnx=True)
    else:
        eran = kwargs["eran"]

    if "subspaces" not in kwargs:
        # by default, load the failed subspaces
        subspaces = load_failed_spaces(spec_num, mid)
        save_failed_space = False
    else:
        subspaces = kwargs["subspaces"]

    if "output_constraints" not in kwargs:
        output_constraints = load_acasxu_outputs_constraints(spec_num)
    else:
        output_constraints = kwargs["output_constraints"]

    return subspaces, output_constraints, eran, save_failed_space


def generate_ctex_acasxu(spec_num, aid, tid, max_depth, **kwargs):
    mid = f"{aid}_{tid}"
    subspaces, constraints, eran, save_failed_space = __parse_kwargs(spec_num, mid, kwargs)
    total_spaces = len(subspaces)
    failed_subspaces = []
    ctexs = []
    for i, subspace in enumerate(subspaces):
        specLB, specUB = subspace
        is_hold, is_ctex_found = search_ctex_recursively(specLB, specUB, eran, constraints, ctexs, specLB, specUB,
                                                         max_depth=max_depth, depth=1)
        if not is_hold:
            failed_subspaces.append(subspace)
        else:
            assert not is_ctex_found

        ctex_info = f"Counterexample found:{is_ctex_found}" if not is_hold else ""
        print(f"{current_timestamp()} Progress(n_{mid}) {i + 1}/{total_spaces}. Space Verified: {is_hold}. {ctex_info}")
        if len(ctexs) > 0 and len(ctexs) % 2 == 0:
            ###############
            # check point
            ###############
            tmp_path = f"./check_points/spec_{spec_num}/n{aid}_{tid}.pkl"
            save_pickle(tmp_path, {"check_point": i, "subspaces": subspaces, "ctexs": ctexs,
                                   "failed_subspaces": failed_subspaces})

    #############
    # save data
    #############
    print("---------------------------------------------------------------")
    print(f"Done! n_{mid}, Total: {total_spaces}, Failed: {len(failed_subspaces)}, Counterexamples found: {len(ctexs)}")

    if len(failed_subspaces) > 0 and save_failed_space:
        save_path = os.path.join(PROJECT_PATH, FAILED_SPACES.format(spec_num, mid))
        save_pickle(save_path, failed_subspaces)
        print(f"failed subspaces saved in {save_path}")

    if len(ctexs) > 0:
        save_path = os.path.join(PROJECT_PATH, VERIFIED_CTEX_ACASXU.format(spec_num, mid))
        save_pickle(save_path, ctexs)
        print(f"counterexamples saved in {save_path}")

def search_ctex_p2(max_depth):
    spec_num = 2
    failed_model_id = [(aid, tid) for aid in range(2, 6) for tid in range(1, 10)]
    ##########################
    # remove verified models
    ##########################
    failed_model_id.remove((3, 3))
    failed_model_id.remove((4, 2))
    for aid, tid in failed_model_id:
        print(f"============{current_timestamp()}============={aid}=={tid}===========================")
        generate_ctex_acasxu(spec_num, aid, tid, max_depth)
    print(f"===={current_timestamp()}======DONE!===============")


if __name__ == '__main__':
    mode = sys.argv[1]  # option [single|batch]
    _spec_num = int(sys.argv[2])
    __max_depth = int(sys.argv[3])
    config.timeout_milp = int(sys.argv[4])  # seconds
    if mode == "single":
        print(mode)
        _aid = sys.argv[5]
        _tid = sys.argv[6]
        generate_ctex_acasxu(_spec_num, _aid, _tid, __max_depth)
    else 
        assert mode == "batch":
        # batch
        if _spec_num == 2:
            search_ctex_p2(__max_depth)
