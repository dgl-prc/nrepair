import sys
import os
import socket

cpu_affinity = os.sched_getaffinity(0)
server_name = socket.gethostname()
sys.path.append("../")

if server_name == "Yserver":
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/python_interface/')
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/deepg/code/')
else:
    sys.path.insert(0, '../../env_setup/ELINA/python_interface/')
    sys.path.insert(0, '../../env_setup/ELINA/deepg/code/')

import numpy as np
from utils.constant import *
from deeppoly.eran import ERAN
from deeppoly.config import config
from utils.data_loader import load_acasxu_inputs_constraints, load_acasxu_outputs_constraints
from utils.model_loader import load_ori_onnx
import copy


def split_input_spaces(spec_num, aid, tid, raw_input_constraints=None):
    raw_input_constraints = copy.deepcopy(raw_input_constraints)
    mid = f"{aid}_{tid}"
    constraints = load_acasxu_outputs_constraints(spec_num)
    model = load_ori_onnx(DataType.ACASXU, mid)
    eran = ERAN(model, is_onnx=True)
    if raw_input_constraints is None:
        specLB, specUB = load_acasxu_inputs_constraints(spec_num)
    else:
        specLB, specUB = raw_input_constraints
    print(f'============================ THE WHOLE INPUT SPACE (net_{aid}_{tid}) ===========================')
    print(f'specLB:{specLB}')
    print(f'specUB:{specUB}')
    is_whold_hold, nn, nlb, nub = eran.analyze_box(specLB, specUB, VERIFY_DOMAIN, config.timeout_lp,
                                                   config.timeout_milp,
                                                   config.use_default_heuristic, constraints)
    # expensive min/max gradient calculation
    nn.set_last_weights(constraints)
    grads_lower, grads_upper = nn.back_propagate_gradiant(nlb, nub)
    smears = [max(-grad_l, grad_u) * (u - l) for grad_l, grad_u, l, u in
              zip(grads_lower, grads_upper, specLB, specUB)]
    split_multiple = 20 / np.sum(smears)
    num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
    step_size = []
    for i in range(5):
        if num_splits[i] == 0:
            num_splits[i] = 1
        step_size.append((specUB[i] - specLB[i]) / num_splits[i])
    start_val = np.copy(specLB)
    end_val = np.copy(specUB)
    total_subspace = 0
    subspaces = []
    for i in range(num_splits[0]):
        specLB[0] = start_val[0] + i * step_size[0]
        specUB[0] = np.fmin(end_val[0], start_val[0] + (i + 1) * step_size[0])
        for j in range(num_splits[1]):
            specLB[1] = start_val[1] + j * step_size[1]
            specUB[1] = np.fmin(end_val[1], start_val[1] + (j + 1) * step_size[1])
            for k in range(num_splits[2]):
                specLB[2] = start_val[2] + k * step_size[2]
                specUB[2] = np.fmin(end_val[2], start_val[2] + (k + 1) * step_size[2])
                for l in range(num_splits[3]):
                    specLB[3] = start_val[3] + l * step_size[3]
                    specUB[3] = np.fmin(end_val[3], start_val[3] + (l + 1) * step_size[3])
                    for m in range(num_splits[4]):
                        specLB[4] = start_val[4] + m * step_size[4]
                        specUB[4] = np.fmin(end_val[4], start_val[4] + (m + 1) * step_size[4])
                        total_subspace += 1
                        subspaces.append((specLB.copy(), specUB.copy()))
    return eran, subspaces, constraints
