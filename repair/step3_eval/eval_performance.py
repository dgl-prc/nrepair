import copy
import sys
from math import log as ln

import onnxruntime.backend as backend
import scipy.stats as stats
import torch
from deeppoly.read_net_file import *
from repair.step1_ctex.acasxu_verify.s3_check_artifacts import check_pred
from utils import constant
from utils.constant import ACASXU_MEANS, ACASXU_STDS, ModelType, DataType, MNIST_MEANS, MNIST_STDS, CIFAR10_MEANS, CIFAR10_STDS
from utils.data_loader import load_acasxu_inputs_constraints, load_img_tests, normalize
from utils.model_loader import load_ori_model

def _get_truncnorm_data(lower, upper, mu, sigma, size):
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(size)

def __is_in_subspace(input_x, lbs, ubs):
    for xi, lb, ub in zip(input_x, lbs, ubs):
        if not (lb <= xi <= ub):
            return False
    return True


def __check_input_space(normalized_x, subspaces):
    """filter inputs which are in specified subspace.
    Args:
        x:np.array, size=(5,).
        subspaces: list(tuple). Each tuple in the list is an input subspace corresponding to the repaired model.
    Returns:
    """
    space_search = []
    for subspace in subspaces:
        normalized_lbs, normalized_ubs = subspace
        is_qualified = __is_in_subspace(normalized_x, normalized_lbs, normalized_ubs)
        space_search.append(is_qualified)
    if any(space_search):
        idx = space_search.index(True)
    else:
        idx = -1
    return idx


def __check_acasxu_input(x, model, spec_num):
    """check if the given input satisfies the specified property under target model.
    Args:
        x: np.array, size=(5,). The input sample
        model: onnx model. The model to step2_repair
        spec_num: int. The serial number of property.
    Returns:
        is_qualified: bool. The result that if the input satisfies the specified property
        pred:int. The predicted label of input given by the onnx model.
    """
    specLB, specUB = load_acasxu_inputs_constraints(spec_num)
    normalized_input = copy.deepcopy(x)
    ori_output = backend.run(model, normalized_input)[0]
    # lowest score corresponding to the best action
    pred = np.argmin(ori_output)
    if __is_in_subspace(normalized_input, specLB, specUB):
        ####################################################################
        # check if the output satisfies the output constraints
        ####################################################################
        is_satisfy = check_pred(ori_output, spec_num)
        if is_satisfy:
            return True, pred
        else:
            return False, -1
    else:
        return True, pred



def synthesize_acasxu_data(ori_model, spec_num, input_spaces, size=1000, rnd_seed=20200801):
    """ randomly synthesize some input data for the acas xu model.
    Note that all the synthesized sample should satisfy the corresponding property under the original model, for that
    the predictions of the repaired model and the original model should be different on these samples which incur
    violation of the property on the original model.
    Each dimension is [feet, radians,radians,v1, v2], where only the radians can take minus value.
    The original range of each dimension is listed as follows:
        - feet, [0,62000]
        - radians, [-3.1415926, 3.1415926]
        - radians, [-3.1415926, 3.1415926]
        - v1, [0,1200]
        - v2, [0,1200]

    Args:
        ori_model: onnx model. The original model to verify and step2_repair.
        spec_num: int. The serial number of property.
        repaired_subspaces: list(tuple). Each tuple in the list is an input subspace corresponding to the repaired model.
        size: int. The number of samples.
        rnd_seed: int. Random seed.

    Returns:
        data: list(tuple). The synthesized data which satisfy the specified property under ori_model. Each element is
                           a tuple: (x, pred), where x is the sample and "pred" is the label given by the original model.
    """

    def normal_sampling(lb, ub):
        # set the random seed
        np.random.seed(rnd_seed)
        init_size = size * 5
        feet_data = _get_truncnorm_data(lb[0], ub[0], ACASXU_MEANS[0], ACASXU_STDS[0], init_size)
        p1_data1 = _get_truncnorm_data(lb[1], ub[1], ACASXU_MEANS[1], ACASXU_STDS[1], init_size)
        p1_data2 = _get_truncnorm_data(lb[2], ub[2], ACASXU_MEANS[2], ACASXU_STDS[2], init_size)
        v_data1 = _get_truncnorm_data(lb[3], ub[3], ACASXU_MEANS[3], ACASXU_STDS[3], init_size)
        v_data2 = _get_truncnorm_data(lb[4], ub[4], ACASXU_MEANS[4], ACASXU_STDS[4], init_size)
        sythe_data = []
        for p, theta, phi, v1, v2 in zip(feet_data, p1_data1, p1_data2, v_data1, v_data2):
            x = np.array([p, theta, phi, v1, v2], dtype=np.float32)
            is_qualified, pred = __check_acasxu_input(x, ori_model, spec_num)
            if is_qualified:
                sythe_data.append((x, pred))
            if len(sythe_data) >= size:
                break
        return sythe_data

    test_data = []
    for space in input_spaces:
        spec_lbs, spec_ubs = space
        test_data.extend(normal_sampling(spec_lbs, spec_ubs))
    return test_data


def acasxu_over_all(test_data, ori_model, repaired_models, repaired_regions):
    """
    In our setting, we do not intend to use the repaired model to replace the original model for any case.
    The right way to use the repaired models is that we only use them to predict the samples from their
    corresponding interval. That is, the new predicted model is {original model, repaired model1 in interval 1, repaired model2 in interval 2, ...}
    Args:
        test_data:
        ori_model:
        repaired_models:
        repaired_regions:

    Returns:

    """
    not_fdlt_cnt = 0
    subregion_inputs = 0
    data_size = len(test_data)
    for sample in test_data:
        x, ori_pred = sample
        normalized_x = copy.deepcopy(x)
        idx = __check_input_space(normalized_x, repaired_regions)
        # when the input is from some erroneous interval, we then use the corresponding model to predict it.
        if idx != -1:
            subregion_inputs += 1
            ori_output = backend.run(ori_model, normalized_x)[0]
            ori_new_pred = np.argmin(ori_output)
            model = repaired_models[idx]
            assert ori_pred == ori_new_pred
            repaired_output = backend.run(model, normalized_x)[0]
            repaired_pred = np.argmin(repaired_output)
            if ori_pred != repaired_pred:
                not_fdlt_cnt += 1
    fdlt = (data_size - not_fdlt_cnt) / data_size
    return fdlt, subregion_inputs




###################################
# Image
###################################

def prepare_img_testing_data(dataset, mid, mtype):
    model = load_ori_model(dataset, mid, mtype)
    ori_correct_idx = {}
    tests = load_img_tests(dataset)
    testing_data = {}
    MEANS, STDS = getattr(constant, f"{dataset.upper()}_MEANS"), getattr(constant, f"{dataset.upper()}_STDS")
    for i, test in enumerate(tests):
        if i == 0:
            continue
        image = np.float32(test[1:len(test)]) / np.float32(255)
        image = normalize(image, MEANS, STDS, dataset,is_conv=False)
        image = image.reshape(1, -1)
        label = int(test[0])
        testing_data[i] = (image, label)
        ori_pred = model_predict(model, image, mtype)
        if ori_pred == label:
            ori_correct_idx[i] = ori_pred
        if i % 500 == 0:
            sys.stdout.write(f"\rProgress-eval {i * 100 / 10000}%")
            sys.stdout.flush()
    print(f"\rNum of testing data: {len(testing_data)}. Correctly predicted: {len(ori_correct_idx)}")
    return testing_data, ori_correct_idx

def calculate_accr(test_data, model, ori_correct_idx, mtype):
    accr_cnt = 0
    net_acc_cnt = 0
    pcnt = 0
    for i in test_data:
        image, label = test_data[i]
        repaired_pred = model_predict(model, image, mtype)
        if repaired_pred == label:
            accr_cnt += 1
            if i in ori_correct_idx:
                net_acc_cnt += 1
        pcnt += 1
    acc_r = accr_cnt / len(ori_correct_idx.keys())
    net_acc = net_acc_cnt / len(ori_correct_idx.keys())
    print(f"\raccR:{acc_r:.4} netAcc:{net_acc:.4}")
    return acc_r, net_acc


def model_predict(model, x, mtype):
    """
    Args:
        model:
        x: ndarray. shape(1,inputdim)
        mtype:

    Returns:

    """
    if mtype == ModelType.ONNX:
        output = backend.run(model, x)
    elif mtype == ModelType.SOCRATES:
        output = model.apply(x)
    else:
        assert mtype == ModelType.PYTORCH
        output = model(torch.tensor(x))
        output = output.detach().numpy()
    pred = np.argmax(output)
    return pred


def eval_image_by_synthesize_data(repaired_model, lb, ub, y_true, data_type, test_size=1000, rnd_seed=20200801):
    dims = len(lb)
    np.random.seed(rnd_seed)
    if data_type == DataType.MNIST:
        rnd_data = [_get_truncnorm_data(lb[i], ub[i], MNIST_MEANS[0], MNIST_STDS[0], test_size) for i in range(dims)]
    else:
        assert data_type == DataType.CIFAR10
        channel_dim = dims // 3
        rnd_data = [_get_truncnorm_data(lb[i], ub[i], CIFAR10_MEANS[0], CIFAR10_STDS[0], test_size) for i in
                    range(channel_dim)]
        rnd_data.extend(
            [_get_truncnorm_data(lb[i], ub[i], CIFAR10_MEANS[1], CIFAR10_STDS[1], test_size) for i in
             range(channel_dim, 2 * channel_dim)])
        rnd_data.extend(
            [_get_truncnorm_data(lb[i], ub[i], CIFAR10_MEANS[2], CIFAR10_STDS[2], test_size) for i in
             range(2 * channel_dim, dims)])
    # test
    for idx in range(test_size):
        x = np.float32([rnd_data[dim][idx] for dim in range(dims)])
        x = x.reshape(1, -1)
        pred_rp = model_predict(repaired_model, x, ModelType.ONNX)
        if pred_rp != y_true:
            print(f"False verified at {idx}-image")
            return False
    return True
