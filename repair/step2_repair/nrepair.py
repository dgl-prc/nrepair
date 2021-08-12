import sys

sys.path.append("../../")
import socket
import sys
import os
import time
from utils.model_hook import get_instance_hooker
from deeppoly.verify_utils import normalize

cpu_affinity = os.sched_getaffinity(0)
server_name = socket.gethostname()
if server_name == "Yserver":
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/python_interface/')
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/deepg/code/')
else:
    sys.path.insert(0, '../../env_setup/ELINA/python_interface/')
    sys.path.insert(0, '../../env_setup/ELINA/deepg/code/')

import onnxruntime.backend as rt
from deeppoly.eran import ERAN
# from vanilla_deeppoly_old.eran import ERAN as NewERAN
from image_deeppoly.eran import ERAN as NewERAN
from repair.step1_ctex.image_verifiy_new_poly import verifiy_single_img
from utils.time_util import folder_timestamp, current_timestamp
from utils.data_loader import load_constraints_ctex
from repair.step2_repair.sort_neurons import *
from repair.step2_repair.onnx_modify import *
from repair.step1_ctex.deeppoly_api import verify_recursive
from deeppoly.config import config
from repair.step1_ctex.acasxu_verify.s3_check_artifacts import check_pred
# from repair.step1_ctex.image_verifiy_without_conv import verifiy_single_img
from utils.constant import *
from repair.step2_repair.onnx2pyt import *
from repair.step2_repair.common_utils import print_overall_result, process_repair_result
from utils.model_loader import *
from models.fnn.masked_fnn import MaskedNet


def test_ctexs(repaired_nn, test_cases, spec_num, y_label=-1):
    """
    Args:
        repaired_nn: onnx model.
        test_cases: list. Each element in the list is an counterexample
        spec_num: int. the idx of property.
    Returns:
    """
    runnable = rt.prepare(repaired_nn, 'CPU')
    is_passed = True
    loss_f = get_loss_fun(spec_num)
    for ctex in test_cases:
        if y_label == -1:
            output = runnable.run(np.array(ctex, dtype=np.float32))
            is_passed = check_pred(output, spec_num)
            loss = loss_f(torch.tensor(output))
        else:
            output = runnable.run(np.array([ctex], dtype=np.float32))
            is_passed = np.argmax(output) == y_label
            loss = loss_f(torch.tensor(output[0]), y_label)
    return is_passed, loss


def is_verified(repaired_nn, dataset, input_boxes, output_constraints, y_label=-1):
    if dataset == DataType.ACASXU:
        eran = ERAN(repaired_nn, is_onnx=True)
        is_success = verify_recursive(input_boxes[0], input_boxes[1], eran, VERIFY_DOMAIN,
                                      output_constraints, max_depth=10,
                                      depth=1)
    else:
        eran = NewERAN(repaired_nn, is_onnx=True)
        # is_success = verifiy_single_img(input_boxes[0], input_boxes[1], eran, y_label)
        is_success,_ = verifiy_single_img(input_boxes[0], input_boxes[1], eran, y_label)
    return is_success


# def is_verified_v2():


def check_model_equality(masked_onnx, model_pyt, ctex, ori_onnx=None):
    ################
    # test ori_onnx
    ################
    onnx_path = os.path.join(PROJECT_PATH, "data/acasxu/onnx/ACASXU_run2a_1_9_batch_2000.onnx")
    ori_onnx = onnx.load(onnx_path)
    runnable = rt.prepare(ori_onnx, 'CPU')
    output = runnable.run(np.array(ctex, dtype=np.float32))
    print("ori onnx:", output)

    #####################
    # test masked_onnx
    #####################
    runnable = rt.prepare(masked_onnx, 'CPU')
    output1 = runnable.run(np.array(ctex, dtype=np.float32))
    print("masked onnx: ", output1)
    #####################
    # test pyt_model
    #####################
    input_ctex = torch.tensor(ctex).reshape(1, 5)
    output2 = model_pyt(input_ctex)
    print("masked pyt model: ", output2)


def repair_nn(masked_onnx, ctex, lnum, nnum, test_cases, input_boxes, output_constraints,
              eta, max_modified, max_iter, spec_num, y_label=-1, dataset=DataType.ACASXU, timeout=3600):
    """stop criteria: when the unique neurones modified exceed the specified maximum number
    For a unique neuron, the number of modification times on it should not exceed the specified maximum number
    Args:
        masked_onnx:
        ctex:
        lnum:
        nnum:
        test_cases:
        input_boxes:
        output_constraints:
        eta:
        max_modified:
        max_iter:
        spec_num:
        y_label:
        dataset:
    Returns:
        repaired_nn: onnx model. The repaired model.
        is_success: bool. If the verification is successful on the B-level interval after step2_repair
        cnt: int . The number of modified neurones.
    """
    model_pyt = MaskedNet(lnum, nnum, dataset)
    repaired_nn = masked_onnx
    iter_cnt = 0
    uniq_pos = {}
    pos_bias = {}
    is_success = RepairResultType.FAILED
    eran_time = 0.0
    tstime = time.time()
    # print(onnx.helper.printable_graph(repaired_nn.graph))
    while True:
        model_pyt = onnx2pyt(repaired_nn, model_pyt, dataset)
        hooker = get_instance_hooker(model_pyt, lnum)
        grad_layer, layer_out = get_grads(model_pyt, hooker, ctex, spec_num, y_label, dataset=dataset)
        neuron = target_neuron(grad_layer, layer_out, uniq_pos, pos_bias, max_iter)
        # if neuron["out"] is None:
        #     is_success = RepairResultType.FAILED_ZERO_OUTPUT
        #     break
        repaired_nn, bias_weight = modify_onnx_sgd(repaired_nn, neuron, eta)
        pos_bias[neuron["pos"]] = bias_weight
        is_statisfied, loss = test_ctexs(repaired_nn, test_cases, spec_num, y_label)
        # verifiy it on the B-level space
        if is_statisfied:
            stime = time.time()
            is_success = is_verified(repaired_nn, dataset, input_boxes, output_constraints, y_label)
            is_success = int(is_success)
            eran_time += time.time() - stime
        iter_cnt += 1
        if is_success == RepairResultType.SUCCESS or len(uniq_pos) > max_modified:
            break
        if iter_cnt % 10 == 0:
            print(
                f"{current_timestamp()} iter cnt :{iter_cnt}, unique neurons: {len(uniq_pos)},loss:{loss},nid:{neuron['pos']},nout:{neuron['out']}, bias_weight:{bias_weight}")
        if time.time() - tstime > timeout:
            is_success = RepairResultType.TIMEOUT
            break
    ttime = time.time() - tstime
    cost = (ttime, eran_time)
    return repaired_nn, is_success, iter_cnt, len(uniq_pos), cost


def tutorial_images(dataset, lnum, nnum, pbt, eta, modify_ratio, max_iter, is_normal):
    args = (dataset, lnum, nnum, pbt, eta, modify_ratio, max_iter, is_normal)
    print(f"args:{args}")
    MEANS = getattr(constant, f"{dataset.upper()}_MEANS")
    STDS = getattr(constant, f"{dataset.upper()}_STDS")
    spec_num = "img"
    # model coordinate
    mid = f"{lnum}_{nnum}"
    # total neurons
    max_modified = int((lnum * nnum) * modify_ratio)
    print(f"-------------------->n_{mid}<----------------")
    img_ids, input_contrains, output_constraints, ctexes, adv_lables = load_constraints_ctex(dataset=dataset, mid=mid,
                                                                                             pbt=pbt)
    masked_onnx = load_mask_onnx(dataset, mid)
    save_timeid = folder_timestamp()
    print(f"------------------REPAIR TIME ID : {save_timeid}-------------------")
    repaired_case = 0
    zero_exceptions = 0
    avg_mn = []  # average modified neurones
    avg_iter = []
    costs = []
    for img_id, ctex, interval, y_label in zip(img_ids, ctexes, input_contrains, output_constraints):
        print(f'---------------Image-{img_id + 1}---------------------------')
        is_ori_success = is_verified(masked_onnx, dataset, interval, output_constraints, y_label)
        print("=========================================================================")
        print(f"Check if original onnx model verified. Expected:False, Result:{is_ori_success}")
        print("=========================================================================")
        if is_normal:
            ctex = normalize(ctex, MEANS, STDS, dataset, False)
        # only take one counterexample into account
        repaired_nn, is_success, iter_cnt, uni_cnt, cost = repair_nn(masked_onnx, ctex, lnum, nnum, [ctex], interval,
                                                                     None,
                                                                     eta, max_modified, max_iter,
                                                                     spec_num=spec_num,
                                                                     y_label=y_label, dataset=dataset)
        repaired_case, zero_exceptions = process_repair_result(ctex_id=img_id, save_timeid=save_timeid,
                                                               repaired_nn=repaired_nn,
                                                               is_success=is_success,
                                                               dataset=dataset, img_id=img_id, eta=eta, mid=mid,
                                                               iter_cnt=iter_cnt, uni_cnt=uni_cnt,
                                                               repair_approach="Nrepair",
                                                               repaired_case=repaired_case,
                                                               zero_exceptions=zero_exceptions,
                                                               avg_mn=avg_mn, avg_iter=avg_iter, costs=costs, cost=cost)

    print_overall_result(args, save_timeid, mid, costs, repaired_case, zero_exceptions, ctexes, avg_mn, avg_iter)


def tutorial_acasxu(spec_num, mid, eta, modify_ratio, max_iter):
    dataset = DataType.ACASXU
    lnum = 6
    nnum = 50
    args = [dataset, spec_num, mid, eta, modify_ratio, max_iter]
    print(f"args:{args}")
    # total neurons
    max_modified = int(lnum * nnum * modify_ratio)
    print(f"-------------------->n_{mid}<----------------")
    input_contraints, output_constraints, ctexes = load_constraints_ctex(dataset=dataset, mid=mid, spec_num=spec_num)
    ###############same with image########
    masked_onnx = load_mask_onnx(dataset, mid)
    save_timeid = folder_timestamp()
    print(f"------------------REPAIR TIME ID : {save_timeid}-------------------")
    repaired_case = 0
    zero_exceptions = 0
    avg_mn = []  # average modified neurones
    avg_iter = []
    costs = []
    ###############end same###############
    interval_id = 0
    for ctex, interval in zip(ctexes, input_contraints):
        interval_id += 1
        print(f'{current_timestamp()}---------------Interval-{interval_id}---------------------------')
        is_ori_success = is_verified(masked_onnx, dataset, interval, output_constraints)
        print("=========================================================================")
        assert is_ori_success is False
        print(f"{current_timestamp()} Check if original onnx model verified. Expected:False, Result:{is_ori_success}")
        print("=========================================================================")
        # only take one counterexample into account
        repaired_nn, is_success, iter_cnt, uni_cnt, cost = repair_nn(masked_onnx, ctex, lnum, nnum, [ctex], interval,
                                                                     output_constraints,
                                                                     eta, max_modified, max_iter,
                                                                     spec_num=spec_num,
                                                                     dataset=dataset)
        repaired_case, zero_exceptions = process_repair_result(ctex_id=interval_id, save_timeid=save_timeid,
                                                               repaired_nn=repaired_nn,
                                                               is_success=is_success,
                                                               dataset=dataset, interval_id=interval_id,
                                                               spec_num=spec_num,
                                                               eta=eta, zero_exceptions=zero_exceptions,
                                                               mid=mid, iter_cnt=iter_cnt, uni_cnt=uni_cnt,
                                                               repaired_case=repaired_case,
                                                               repair_approach="Nrepair",
                                                               avg_mn=avg_mn, avg_iter=avg_iter, costs=costs, cost=cost)
    print_overall_result(args, save_timeid, mid, costs, repaired_case, zero_exceptions, ctexes, avg_mn, avg_iter)


def batch_repair_acasxu(eta, max_iter):
    spec_num = 2
    modify_ratio = 0.05
    config.timeout_milp = 1
    failed_model_id = [(aid, tid) for aid in range(2, 6) for tid in range(1, 10)]
    ##########################
    # remove verified models
    ##########################
    failed_model_id.remove((3, 3))
    failed_model_id.remove((4, 2))
    for aid, tid in failed_model_id:
        mid = f"{aid}_{tid}"
        tutorial_acasxu(spec_num, mid, eta, modify_ratio, max_iter)


if __name__ == '__main__':
    _data_set = sys.argv[1]
    _eta = float(sys.argv[2])
    _maxiter = float(sys.argv[3])
    if _data_set == 'acasxu':
        _spec_num = int(sys.argv[4])
        if _spec_num == 2:
            batch_repair_acasxu(_eta, _maxiter)
        elif _spec_num == 7:
            tutorial_acasxu(7, "1_9", _eta, 0.05, _maxiter)
        elif _spec_num == 8:
            tutorial_acasxu(8, "2_9", _eta, 0.05, _maxiter)
    elif _data_set in ['mnist', "cifar10"]:
        _lnum = int(sys.argv[4])
        _nnum = int(sys.argv[5])
        _pbt = float(sys.argv[6])
        _mr = float(sys.argv[7])
        _is_normal = bool(int(sys.argv[8]))
        tutorial_images(_data_set, _lnum, _nnum, _pbt, _eta, _mr, _maxiter, _is_normal)
