import sys

sys.path.append("../../")
import torch
import numpy as np
import onnxruntime.backend as backend
import torch

from deeppoly.verify_utils import normalize
from repair.step2_repair.loss_func import get_loss_fun
from utils import constant
from utils.constant import DataType
from utils.data_loader import load_constraints_ctex
from utils.model_loader import load_ori_onnx
import onnxruntime.backend as rt


def check_image(dataset, mid, pbt, is_normlize):
    print(f"=================={dataset},{mid},{pbt},{is_normlize}=================")
    MEANS = getattr(constant, f"{_dataset.upper()}_MEANS")
    STDS = getattr(constant, f"{_dataset.upper()}_STDS")
    img_ids, input_contrains, output_constraints, ctexes, adv_labels = load_constraints_ctex(dataset=dataset, mid=mid,
                                                                                             pbt=pbt)
    ori_onnx = load_ori_onnx(dataset, mid)
    runnable = rt.prepare(ori_onnx, 'CPU')
    valid_cnt = 0
    for img_id, ctex, interval, y_label, adv_label in zip(img_ids, ctexes, input_contrains, output_constraints,
                                                          adv_labels):
        if is_normlize:
            ctex = normalize(ctex, MEANS, STDS, dataset, False)
        output = runnable.run(np.array([ctex], dtype=np.float32))
        ctex_label = np.argmax(output)
        if ctex_label == adv_label or ctex_label != y_label:
            valid_cnt += 1
        else:
            print(f"Image-id-{img_id} Failed. CTEX-LABEL:{ctex_label}, Y-LABEL:{y_label}, ADV-LABEL:{adv_label}")
    print(f"Total valid:{valid_cnt}/{len(adv_labels)}")


def convert_image_onnx(image, data_type, is_conv):
    if isinstance(image, list):
        image = np.array(image, dtype=np.float32)
    # MEANS, STDS = getattr(constant, f"{data_type}_MEANS".upper()), getattr(constant, f"{data_type}_STDS".upper())
    # image = normalize(image, MEANS, STDS, data_type, is_conv)
    image = image.reshape(32, 32, 3)
    image = np.transpose(image, axes=(2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image
#
#
# def test_image_ctexs(repaired_nn, test_cases, spec_num, y_label=-1):
#     """
#     Args:
#         repaired_nn: onnx model.
#         test_cases: list. Each element in the list is an counterexample
#         spec_num: int. the idx of property.
#     Returns:
#     """
#     runnable = backend.prepare(repaired_nn, 'CPU')
#     is_passed = True
#     loss_f = get_loss_fun(spec_num)
#     for ctex in test_cases:
#         output = runnable.run(ctex)
#         # if len(output) > 0:
#         is_passed = np.argmax(output) == y_label
#         loss = loss_f(torch.tensor(output[0]), y_label)
#     return is_passed, loss


if __name__ == '__main__':

    _dataset = sys.argv[1]
    _is_normlize = bool(int(sys.argv[2]))
    _pbt = 0.03 if _dataset == DataType.MNIST else 0.0012
    for _mid in ["3_100", "5_100", "7_100"]:
        check_image(_dataset, _mid, _pbt, _is_normlize)
# dataset = DataType.CIFAR10
# net_name = "convSmallRELU__Point"
# pbt = 0.012
# onnx_model = load_ori_onnx(dataset, net_name)
# runnable = backend.prepare(onnx_model, 'CPU')
# cnt = 0
# adv_cnt = 0
# img_ids, input_contrains, output_constraints, ctexes, adv_labels = load_constraints_ctex(dataset=dataset,
#                                                                                          mid=net_name, pbt=pbt)
# for img_id, ctex, interval, y_label, adv_label in zip(img_ids, ctexes, input_contrains, output_constraints,
#                                                       adv_labels):
#     ctex_onnx = convert_image_onnx(ctex, dataset, is_conv=True)
#     output = runnable.run(ctex_onnx)
#     pred = np.argmax(output)
#     # no_adv, _ = test_image_ctexs(onnx_model, [ctex_onnx], spec_num=-1, y_label=y_label)
#     print(f"pred:{pred}, adv_label:{adv_label}")
#     if pred == y_label:
#         cnt += 1
#         print(f"img_id:{img_id}, y_label:{y_label}")
#     if pred == adv_label:
#         adv_cnt += 1
# print(f"benign cnt:{cnt}")
# print(f"adv cnt:{adv_cnt}")
# # denormalize(adv_image[0], MEANS, STDS, dataset)
#
