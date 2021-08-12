import numpy as np
import torch
from onnx import helper, numpy_helper, TensorProto
from models.fnn.fnn import ACASXu
import copy
import onnx
from utils.constant import DataType, LayerType


def read_nnet(nnetFile, withNorm=False):
    '''Read a .nnet file and return list of weight matrices and bias vectors
    This code is copied from https://github.com/sisl/NNet/blob/master/utils/readNNet.py
    Inputs:
        nnetFile: (string) .nnet file to read
        withNorm: (bool) If true, return normalization parameters

    Returns:
        weights: List of weight matrices for fully connected network
        biases: List of bias vectors for fully connected network
    '''

    # Open NNet file
    f = open(nnetFile, 'r')

    # Skip header lines
    line = f.readline()
    while line[:2] == "//":
        line = f.readline()

    # Extract information about network architecture
    record = line.split(',')
    numLayers = int(record[0])
    inputSize = int(record[1])

    line = f.readline()
    record = line.split(',')
    layerSizes = np.zeros(numLayers + 1, 'int')
    for i in range(numLayers + 1):
        layerSizes[i] = int(record[i])

    # Skip extra obsolete parameter line
    f.readline()

    # Read the normalization information
    line = f.readline()
    inputMins = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    inputMaxes = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    means = [float(x) for x in line.strip().split(",") if x]

    line = f.readline()
    ranges = [float(x) for x in line.strip().split(",") if x]

    # Read weights and biases
    weights = []
    biases = []
    for layernum in range(numLayers):

        previousLayerSize = layerSizes[layernum]
        currentLayerSize = layerSizes[layernum + 1]
        weights.append([])
        biases.append([])
        weights[layernum] = np.zeros((currentLayerSize, previousLayerSize))
        for i in range(currentLayerSize):
            line = f.readline()
            aux = [float(x) for x in line.strip().split(",")[:-1]]
            for j in range(previousLayerSize):
                weights[layernum][i, j] = aux[j]
        # biases
        biases[layernum] = np.zeros(currentLayerSize)
        for i in range(currentLayerSize):
            line = f.readline()
            x = float(line.strip().split(",")[0])
            biases[layernum][i] = x

    f.close()

    if withNorm:
        return weights, biases, inputMins, inputMaxes, means, ranges
    return weights, biases


def nnet2masked_onnx_acasxu(nnet_file):
    """
    Add a mask layer after each hidden layer.
    Each mask layer consists of an identity matrix (weights) and an zero_array (bias)
    Args:
        nnet_file:
    Returns:
    """
    weights, biases = read_nnet(nnet_file)
    inputSize = weights[0].shape[1]
    outputSize = weights[-1].shape[0]
    outputVar = "y_out"  # name of input
    inputVar = "X"  # name of output
    numLayers = len(weights)

    # Initialize graph
    inputs = [helper.make_tensor_value_info(inputVar, TensorProto.FLOAT, [inputSize])]
    outputs = [helper.make_tensor_value_info(outputVar, TensorProto.FLOAT, [outputSize])]
    operations = []
    initializers = []
    for i in range(numLayers):
        # Use outputVar for the last layer
        outputName = "H%d" % i
        if i == numLayers - 1:
            outputName = outputVar
        # Weight matrix multiplication
        operations.append(helper.make_node("MatMul", ["W%d" % i, inputVar], ["M%d" % i]))
        initializers.append(numpy_helper.from_array(weights[i].astype(np.float32), name="W%d" % i))
        # Bias add
        operations.append(helper.make_node("Add", ["M%d" % i, "B%d" % i], [outputName]))
        initializers.append(numpy_helper.from_array(biases[i].astype(np.float32), name="B%d" % i))

        # Use Relu activation for all layers except the last layer
        if i < numLayers - 1:
            operations.append(helper.make_node("Relu", ["H%d" % i], ["R%d" % i]))
            # add mask layer
            mask_layer_weight = np.identity(50)
            mask_layer_bias = np.zeros(50)
            # Weight matrix multiplication
            initializers.append(numpy_helper.from_array(mask_layer_weight.astype(np.float32), name="MSKW%d" % i))
            operations.append(helper.make_node("MatMul", ["MSKW%d" % i, "R%d" % i], ["KW%d" % i]))
            # Bias add
            initializers.append(numpy_helper.from_array(mask_layer_bias.astype(np.float32), name="MSKB%d" % i))
            operations.append(helper.make_node("Add", ["KW%d" % i, "MSKB%d" % i], ["KB%d" % i]))

            inputVar = "KB%d" % i

    # Create the graph and model in onnx
    graph_proto = helper.make_graph(operations, "masked_onnx_Model", inputs, outputs, initializers)
    model_def = helper.make_model(graph_proto)
    # onnx.save(model_def, save_path)
    return model_def


def pyonnx2masked_onnx(ori_onnx):
    """ Convert a pytorch-derived onnx model to a mask model.
    Only support Gemm and Relu operations.
    Args:
        ori_onnx: onnx model which is exported from pytorch
    Returns:

    """
    alpha = 1.0
    beta = 1.0
    transB = 1
    initializers = [init for init in ori_onnx.graph.initializer]
    new_initializers = []
    operations = []
    total_p_layers = len(initializers)
    num_nodes = len(ori_onnx.graph.node)
    last_output_name = -1
    mask_layer_id = 0
    for i, init in enumerate(initializers):
        new_initializers.append(copy.deepcopy(init))
        if i < num_nodes:
            ori_node = ori_onnx.graph.node[i]
        else:
            continue
        # modify the input of the original node
        if ori_node.op_type == "Gemm" and i > 1:
            wight = ori_node.input[1]
            bias = ori_node.input[2]
            out_name = ori_node.output[0]
            new_node = helper.make_node("Gemm", [last_output_name, wight, bias], [out_name], alpha=alpha, beta=beta,
                                        transB=transB)
            operations.append(new_node)
        else:
            operations.append(ori_node)

        # get the original output name of each layer (weight layer + Relu)
        if ori_node.op_type == "Relu":
            # the variable name of the output of previous layer.
            last_output_name = ori_node.output[0]

        # add mask layers
        if i % 2 != 0 and i < total_p_layers - 1:
            # get hidden size
            hsize = init.dims[0]
            mask_layer_weight = np.identity(hsize)
            mask_layer_bias = np.zeros(hsize)
            MSKW_NAME = f"MSKW{mask_layer_id}"
            MSKB_NAME = f"MSKB{mask_layer_id}"
            MSKOUT = f"MSKOUT{mask_layer_id}"
            mask_layer_id += 1
            # Weight matrix multiplication
            initializers.append(
                numpy_helper.from_array(mask_layer_weight.astype(np.float32), name=MSKW_NAME))

            initializers.append(
                numpy_helper.from_array(mask_layer_bias.astype(np.float32), name=MSKB_NAME))

            operations.append(
                helper.make_node("Gemm", [last_output_name, MSKW_NAME, MSKB_NAME], [MSKOUT], alpha=alpha,
                                 beta=beta,
                                 transB=transB))
            last_output_name = MSKOUT

    graph_proto = helper.make_graph(operations, "maksed_onnx_Model", ori_onnx.graph.input,
                                    ori_onnx.graph.output, initializers)
    model_def = helper.make_model(graph_proto)

    return model_def


def nnet2pyt(nnet_path):
    """ Convert nnet to pytorch model
    Args:
        nnet_path:
    Returns:
    """
    model_pyt = ACASXu()
    weights, biases = read_nnet(nnet_path)
    param_weights = []
    param_biases = []
    for name, param in model_pyt.named_parameters():
        if name.endswith("bias"):
            param_biases.append(param)
        else:
            param_weights.append(param)
    for idx in range(len(weights)):
        param_weights[idx].data = torch.tensor(weights[idx], dtype=torch.float32)
        param_biases[idx].data = torch.tensor(biases[idx], dtype=torch.float32)
    return model_pyt

    # # test
    # predit1 = model_pyt(torch.tensor([[1000.0, 0.0, -1.5, 100.0, 100.0]]))
    # predit2 = model_pyt(torch.tensor([[1000.0, 0.0, 1.5, 100.0, 100.0]]))
    # print(predit1.tolist()[0])
    # print(predit2.tolist()[0])


def get_mask_layer_name(lid):
    """
    Args:
        lid: the index of layer. 0-index. This lid considering masked layers, so we need to convert it to the format
        of model's definition.
    Returns:
    """
    lid //= 2
    weight_name = f"MSKW{lid}"
    bias_name = f"MSKB{lid}"
    return weight_name, bias_name


def modify_onnx(masked_onnx, lid, nid, sign, n_output, epsilon):
    """
    For the deactivate action, we only set the value of related position in the mask_layer as 0
    For the activate action,   we first set the value of related position in the mask_layer as 0,
                               and then set the bias as a certain positive value.
    The above can be summarized as : neuron_value = (max_active_value-sign*max_active_value)/2
    Args:
        masked_onnx:
        lid:
        nid:
        sign: int. -1 or 1.
        n_output: float. neuron output
        epsilon: step size,(0,1)
    Returns:

    """
    # print(onnx.helper.printable_graph(masked_onnx.graph))
    # find the name of target layer
    hsize = masked_onnx.graph.initializer[0].dims[0]
    weight_name, bias_ame = get_mask_layer_name(lid)
    initializers = [init for init in masked_onnx.graph.initializer]
    new_weights = np.identity(hsize, dtype=np.float32)
    new_bias = np.zeros(hsize, dtype=np.float32)

    # assume that the weight layer is closely followed by the bias layer.
    for i, init in enumerate(initializers):
        if init.name == weight_name:
            # no matter activate or deactivate, we should first reset the related value to 0
            new_weights[nid][nid] = 0
            initializers[i] = numpy_helper.from_array(new_weights.astype(np.float32), name=init.name)
        if init.name == bias_ame:
            # we perform activate/deactivate by set the related bias as a certain value.
            new_bias[nid] = n_output * (1 - sign * epsilon)
            initializers[i] = numpy_helper.from_array(new_bias.astype(np.float32), name=init.name)

    graph_proto = helper.make_graph(masked_onnx.graph.node, "maksed_onnx_Model", masked_onnx.graph.input,
                                    masked_onnx.graph.output, initializers)
    model_def = helper.make_model(graph_proto)
    return model_def


def modify_onnx_sgd(masked_onnx, neuron, eta):
    """
    Args:
        masked_onnx:
        lid:
        nid:
        sign: int. -1 or 1.
        n_output: float. neuron output
        eta: step size,(0,1)
    Returns:

    """
    lid, nid = neuron["pos"]
    n_output = neuron["out"]
    grad = neuron["grad"]
    # find the name of target layer
    hsize = masked_onnx.graph.initializer[0].dims[0]
    weight_name, bias_ame = get_mask_layer_name(lid)
    initializers = [init for init in masked_onnx.graph.initializer]
    new_weights = np.identity(hsize, dtype=np.float32)
    # assume that the weight layer is closely followed by the bias layer.
    updated_weight = 0.0
    for i, init in enumerate(initializers):
        if init.name == weight_name:
            # no matter activate or deactivate, we should first reset the related value to 0
            new_weights[nid][nid] = 0
            initializers[i] = numpy_helper.from_array(new_weights.astype(np.float32), name=init.name)
        if init.name == bias_ame:
            # we perform activate/deactivate by set the related bias as a certain value.
            bias_weight = numpy_helper.to_array(init)
            bias_weight = list(bias_weight)
            if bias_weight[nid] == 0.0:
                new_weight = n_output - eta * grad
            else:
                new_weight = bias_weight[nid] - eta * grad
            # relu
            updated_weight = max(0.0, new_weight)
            bias_weight[nid] = updated_weight
            initializers[i] = numpy_helper.from_array(np.array(bias_weight, dtype=np.float32), name=init.name)
            break

    graph_proto = helper.make_graph(masked_onnx.graph.node, "maksed_onnx_Model", masked_onnx.graph.input,
                                    masked_onnx.graph.output, initializers)
    model_def = helper.make_model(graph_proto)
    return model_def, updated_weight


def get_target_param_name(lid, onnx_layers, modify_conv=False):
    """
    Note that if
    Args:
        lid:
        onnx_layers:
        modify_conv:

    Returns:

    """
    # [("1", LayerType.CONV),
    #  ("3", LayerType.CONV),
    #  ("6", LayerType.FC),
    #  ("8", LayerType.FC)]
    p = 0
    for layer_name in onnx_layers:
        if not modify_conv:
            if onnx_layers[layer_name] == LayerType.FC:
                p += 1
            else:
                continue
        else:
            p += 1
        if p - 1 == lid:  # convert to 0-index:
            return f"{layer_name}.weight", f"{layer_name}.bias"


def modify_onnx_no_mask(onnx_model, onnx_layers, neuron, eta, modify_conv=False):
    """ This function is similar to modify_onnx_sgd, but the modification is performed on the original onnx model.
    Currently, we limit the modification on fully connected layer.

    Args:
        onnx_model:
        lid:
        nid:
        sign: int. -1 or 1.
        n_output: float. neuron output
        eta: step size,(0,1)
    Returns:

    """
    lid, nid = neuron["pos"]
    n_output = neuron["out"]
    grad = neuron["grad"]
    # find the name of target layer
    weight_name, bias_ame = get_target_param_name(lid, onnx_layers, modify_conv)
    initializers = [init for init in onnx_model.graph.initializer]
    updated_weight = 0.0
    modify_cnt = 0
    for i, init in enumerate(initializers):
        if init.name == weight_name:
            # no matter activate or deactivate, we should first reset the neuron's weight to 0
            old_weights = copy.copy(numpy_helper.to_array(init))
            old_weights[nid] = np.zeros(shape=old_weights[nid].shape, dtype=np.float32)
            initializers[i] = numpy_helper.from_array(old_weights.astype(np.float32), name=init.name)
            modify_cnt += 1
        if init.name == bias_ame:
            # we perform activate/deactivate by set the related bias as a certain value.
            bias_weight = numpy_helper.to_array(init)
            bias_weight = list(bias_weight)
            if bias_weight[nid] == 0.0:
                new_weight = n_output - eta * grad
            else:
                new_weight = bias_weight[nid] - eta * grad
            # relu
            updated_weight = max(0.0, new_weight)
            bias_weight[nid] = updated_weight
            initializers[i] = numpy_helper.from_array(np.array(bias_weight, dtype=np.float32), name=init.name)
            modify_cnt += 1
        if modify_cnt == 2:
            break
    # graph_proto = helper.make_graph(ori_node_def, "repaired_onnx_Model", onnx_model.graph.input,
    #                                 ori_output_def, initializers)
    graph_proto = helper.make_graph(onnx_model.graph.node, "repaired_onnx_Model", onnx_model.graph.input,
                                    onnx_model.graph.output, initializers)
    model_def = helper.make_model(graph_proto)
    onnx.checker.check_model(onnx_model)
    return model_def, updated_weight
