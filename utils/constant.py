import socket
import os

server_name = socket.gethostname()
if server_name == "Yserver":
    PROJECT_PATH = "/home/dgl/project/eran_model/"
else:
    PROJECT_PATH = f"/home/{os.getlogin()}/project/nn_repair/"


class DataType:
    ACASXU = "acasxu"
    MNIST = "mnist"
    CIFAR10 = "cifar10"



class RepairType:
    MM = "MM"
    nRepair = "nRepair"

class RepairResultType:
    SUCCESS = 1
    FAILED = 0
    FAILED_ZERO_OUTPUT = -2
    TIMEOUT = -3


class ModelType:
    ONNX = "onnx"
    PYTORCH = "pytorch"
    SOCRATES = "socrates"


class Network:
    FNN = "fnn"
    LENET = "lenet"


class LayerType:
    FC = "fc"
    CONV = "conv"



VERIFY_DOMAIN = "deeppoly"

MNIST_INPUTSIZE = 784
MNIST_TRAIN = "/home/dgl/project/eran_model/data/mnist/mnist_train.csv"
MNIST_MODEL_PATH = "data/mnist/onnx/mnist_{}.onnx"
MNIST_PYMODEL_PATH = "data/mnist/onnx/mnist_{}.pyt"
# MNIST_MEANS = [0]
# MNIST_STDS = [1]
MNIST_MEANS = [0.1307]
MNIST_STDS = [0.3081]
MNIST_TEST_DATA = "data/mnist/mnist_test.csv"
MNIST_REPAIRED_ONNX_PATH = "data/mnist/repaired_onnx/summary_{}.txt"

CIFAR10_INPUTSIZE = 3072
CIFAR10_TRAIN = "/home/dgl/project/eran_model/data/cifar10/cifar10_train.csv"
CIFAR10_MODEL_PATH = "data/cifar10/onnx/cifar10_{}.onnx"
CIFAR10_PYMODEL_PATH = "data/cifar10/onnx/cifar10_{}.pyt"
CIFAR10_MEANS = [0.4914, 0.4822, 0.4465]
CIFAR10_STDS = [0.2023, 0.1994, 0.2010]
CIFAR10_TEST_DATA = "data/cifar10/cifar10_test.csv"
# CIFAR10_TEST_DATA = "data/cifar10/cifar10_test_full.csv"
CIFAR10_REPAIRED_ONNX_PATH = "data/cifar10/repaired_onnx/summary_{}.txt"

ACASXU_MODEL_PATH = "data/acasxu/onnx/ACASXU_run2a_{}_batch_2000.onnx"
ACASXU_LB = [0., -3.1415926, -3.1415926, 0., 0.]
ACASXU_UB = [62000, 3.1415926, 3.1415926, 1200, 1200]
ACASXU_MEANS = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
ACASXU_STDS = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
ACASXU_INPUT_CONSTRAINTS = "data/acasxu/property/p{}_inputs_constraints.txt"
ACASXU_OUTPUT_CONSTRAINTS = "data/acasxu/property/p{}_outputs_constraints.txt"
ACASXU_INPUTSIZE = 5
###############################################
# arg: {property id}, {mid}
###############################################
FAILED_SPACES = "data/acasxu/failed_spaces/spec_{}/n_{}.pkl"

###############################################
# arg: property id"
###############################################
VERIFIED_CTEX_ACASXU = "data/acasxu/ctex/spec_{}/n_{}.pkl"

###############################################
# arg: image id
###############################################
VERIFIED_CTEX_MNIST = "data/mnist/ctex/pbt_{}/n_{}.pkl"

###############################################
# arg: image id
###############################################
VERIFIED_CTEX_CIFAR10 = "data/cifar10/ctex/pbt_{}/n_{}.pkl"

#################
# repaired model
#################
# arg: {dataset} {baseline name} {mid} {property id} {interval id} {epsilon} {timeid}
ACASXU_REPAIRED_ONNX_PATH = "data/{}/repaired_onnx/{}/n_{}/property_{}/interval-{}/epsilon_{}/{}"
# arg: {dataset} {mid}       {property(img) id}    {epsilon} {timeid}
IMG_REPAIRED_ONNX_PATH = "data/{}/repaired_onnx/n_{}/img-{}/epsilon_{}/{}"
SUMMARY_REPAIRED_ONNX_PATH = "data/{}/repaired_onnx/summary_{}_{}.txt"

#################################################################
# rq2 repaired models
# arg: {dataset} {property id}, {mid}, {iid}
#################################################################
RQ2_REPAIRED_ONNX = "data/{}/rq2_data/property_{}/n_{}/avalue_{}/"

RQ2_IMG_ONNX = "data/{}/rq2_data/n_{}/avalue_{}/"

RQ3_REPAIRED_ONNX = "data/{}/rq3_data/n_{}/active_{}_mration_{}/{}"

# this intervals have been searched 1-index.
# P7_EXCLUDED = [i for i in range(12, 24)] + [1, 2, 3,
#                                             633,
#                                             76,
#                                             631,
#                                             654,
#                                             306,
#                                             304,
#                                             528,
#                                             842,
#                                             221,
#                                             784,
#                                             389,
#                                             771,
#                                             112,
#                                             550,
#                                             183,
#                                             872,
#                                             736,
#                                             542,
#                                             463,
#                                             186,
#                                             38,
#                                             248,
#                                             426,
#                                             396,
#                                             36,
#                                             466,
#                                             948,
#                                             460,
#                                             95,
#                                             756,
#                                             788,
#                                             727,
#                                             102,
#                                             326,
#                                             541,
#                                             459,
#                                             707,
#                                             175,
#                                             781,
#                                             573,
#                                             335]
# P7_EXCLUDED_YSERVER = [447, 556, 744]
# P7_EXCLUDED_ZJUSEC = [
#     383,
#     793,
#     505,
#     206,
#     297,
#     403,
#     188,
#     127,
#     161,
#     11,
#     853,
#     706,
#     225,
#     936
# ]
