import sys

sys.path.append("../")
import re
import itertools
from utils.help_func import load_pickle
from deeppoly.constraint_utils import get_constraints_from_file
from deeppoly.verify_utils import normalize
from utils.constant import *
from utils import constant
import csv


def parse_input_box(text):
    """This function extracted from the eran project"""
    intervals_list = []
    for line in text.split('\n'):
        if line != "":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb, ub] = interval.split(",")
                intervals.append((float(lb), float(ub)))
            intervals_list.append(intervals)
    # return every combination
    boxes = itertools.product(*intervals_list)
    return list(boxes)


def load_acasxu_outputs_constraints(spec_num):
    output_constraints_path = os.path.join(PROJECT_PATH, ACASXU_OUTPUT_CONSTRAINTS.format(spec_num))
    output_constraints = get_constraints_from_file(output_constraints_path)
    return output_constraints


def load_acasxu_inputs_constraints(spec_num):
    property_file = os.path.join(PROJECT_PATH, ACASXU_INPUT_CONSTRAINTS.format(spec_num))
    with open(property_file, 'r') as f:
        tests = f.read()
    box = parse_input_box(tests)[0]
    spec_lb = [interval[0] for interval in box]
    spec_ub = [interval[1] for interval in box]
    spec_lb = normalize(spec_lb, ACASXU_MEANS, ACASXU_STDS, DataType.ACASXU, False)
    spec_ub = normalize(spec_ub, ACASXU_MEANS, ACASXU_STDS, DataType.ACASXU, False)
    return spec_lb, spec_ub


def load_failed_spaces(spec_num, mid):
    data_path = os.path.join(PROJECT_PATH, FAILED_SPACES.format(spec_num, mid))
    failed_spaces = load_pickle(data_path)
    return failed_spaces


def __parse_verify_rst_acas(ctex_data):
    input_contrains_list = []
    ctexes = []
    for ctext_ele in ctex_data:
        ctexes.append(ctext_ele["cexp"])
        specLB = ctext_ele["specLB"]
        specUB = ctext_ele["specUB"]
        input_contrains_list.append((specLB, specUB))
    return input_contrains_list, ctexes


def __parse_verify_rst_img(ctex_data):
    input_contrains_list = []
    output_constraints = []
    ctexes = []
    img_ids = []
    adv_labels = []
    for ctext_ele in ctex_data:
        ctexes.append(ctext_ele["cexp"])
        specLB = ctext_ele["specLB"]
        specUB = ctext_ele["specUB"]
        input_contrains_list.append((specLB, specUB))
        output_constraints.append(ctext_ele["y_true"])
        adv_labels.append(ctext_ele["y_adv"])
        img_ids.append(ctext_ele["img_id"])
    return img_ids, input_contrains_list, output_constraints, ctexes, adv_labels


def load_constraints_ctex(**kwargs):
    dataset = kwargs["dataset"]
    mid = kwargs["mid"]
    if dataset == DataType.ACASXU:
        spec_num = kwargs["spec_num"]
        ctex_path = os.path.join(PROJECT_PATH, VERIFIED_CTEX_ACASXU.format(spec_num, mid))
        ctex_data = load_pickle(ctex_path)
        input_contraints, ctexes = __parse_verify_rst_acas(ctex_data)
        output_constraints = load_acasxu_outputs_constraints(spec_num)
        return input_contraints, output_constraints, ctexes
    elif dataset in [DataType.MNIST, DataType.CIFAR10]:
        pbt = kwargs["pbt"]
        VERIFIED_CTEX = getattr(constant, f"VERIFIED_CTEX_{dataset.upper()}")
        ctex_path = os.path.join(PROJECT_PATH, VERIFIED_CTEX.format(f"{pbt:.4}", mid))
        ctex_data = load_pickle(ctex_path)
        return __parse_verify_rst_img(ctex_data)


def load_img_tests(dataset):
    data_path = os.path.join(PROJECT_PATH, getattr(constant, f"{dataset.upper()}_TEST_DATA"))
    csvfile = open(data_path, 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests
