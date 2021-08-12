import sys

sys.path.append("../")
import torch
import copy
import numpy as np
from repair.step2_repair.loss_func import get_loss_fun
from utils.constant import DataType, ModelType, LayerType
from onnx import numpy_helper


def grad_loss_last_layer(loss_f, layer_outpout, weight, bias, y_label):
    """ calculate the gradient of cross entropy loss wrt the output of last hidden layer.
    Args:
        loss_f: function. the loss function.
        layer_outpout: the output of last hidden layer. this is used as the input of last layer.
        weight: the weight from the last hidden layer to the output layer
        bias:
        target_class: int. the index of target class
    Returns:

    """
    if not isinstance(layer_outpout, torch.Tensor):
        layer_outpout = torch.tensor(layer_outpout).reshape(1, -1)
        layer_outpout.requires_grad = True
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight)
    if not isinstance(bias, torch.Tensor):
        bias = torch.tensor(bias)
    y = torch.addmm(bias, layer_outpout, weight.t())
    # loss_f = torch.nn.CrossEntropyLoss()
    if y_label == -1:
        loss = loss_f(y)
    else:
        loss = loss_f(y, y_label)
    loss.backward()
    return layer_outpout.grad


def grad_each_layer(weights, layer_output, last_layer_grad, model_type):
    """ Calculate the gradient of **loss** with respect to each layer.
    Note this function can also calculate the gradient of **ouptut** with respect to each layer by
    set 'grad_v' as 'weights[-1][target_class]'
    Only support two sorts of network: FNN without activation function and FNN with Relu.
    Args:
        weights: the model's weights of all layers
        target_class: the desired class
        layer_output: the output of each layer
    Returns:
        grad_v: the gradient on input
        grad_layer: dict. the gradient on each layer and the layer id is 0-index
    """
    total_layers = len(weights)
    # note the idx of weight/output layer is same the definition of the model.
    layer_idx = [i for i in range(total_layers)][::-1]
    # layer_idx = layer_idx[::-1]
    grad_v = last_layer_grad
    # note that the idx of grad layer is from the input layer to the last masked layer.
    grad_layer = []
    grad_layer.append((layer_idx[0], grad_v.tolist()[0]))
    for layer in layer_idx[1:]:
        weight = weights[layer]  # from the previous layer to current layer.
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        if model_type == ModelType.PYTORCH and layer % 2 == 0:  # masked layer <---  normal layer, current is normal layer.
            # Output of normal layer = Relu([Output of masked layer]*[Weight of normal layer])
            # In this case, we need to deal with the relu operation
            output = layer_output[layer]  # the output of current layer.
            if not isinstance(output, torch.Tensor):
                output = torch.tensor(output)
            mask = torch.tensor(output > 0, dtype=int)
            mask = mask.reshape(-1, 1)
            weight = mask * weight
        grad_v = torch.matmul(grad_v, weight)
        grad_layer.append((layer, grad_v.tolist()[0]))  # remove the batch-dim
    # sort layer by the ascending order of layer_id.
    # the map of layer id: the grad of grad_layer_0 is the grad of input, and the grad of grad_layer_1 is in the layer 0
    grad_layer = sorted(grad_layer, key=lambda x: x[0])
    return grad_v, grad_layer


def extract_weights_bias(model, model_type, onnx_layers=None):
    """ extract the weight and bias of each layer.
    Args:
        model:
        model_type: [pytorch|onnx]
    Returns:
        weights: list(tensor).
        bias: list(tensor).
    """
    weights = []
    biases = []
    if model_type == "pytorch":
        for name, param in model.named_parameters():
            if name.endswith("bias"):
                biases.append(param.data)
            else:
                weights.append(param.data)
    elif model_type == "onnx":
        for init in model.graph.initializer:
            name = init.name
            layer_id = name.split(".")[0]
            if onnx_layers is not None:
                if onnx_layers[layer_id] == LayerType.FC:
                    param = copy.copy(numpy_helper.to_array(init))
                    if name.endswith("bias") or name.startswith("B"):
                        biases.append(param)
                    else:
                        weights.append(param)
            else:
                param = copy.copy(numpy_helper.to_array(init))
                if name.endswith("bias") or name.startswith("B"):
                    biases.append(param)
                else:
                    weights.append(param)
    else:
        raise Exception(f"Unsupported model type:{model_type}")
    return weights, biases


def get_grads(model_pyt, hooker, counterexample, spec_num, y_label=-1, dataset=DataType.ACASXU,
              model_type=ModelType.PYTORCH):
    if dataset == DataType.ACASXU:
        input_ctex = torch.tensor(counterexample).reshape(1, 5)
    else:
        input_ctex = torch.tensor([counterexample], dtype=torch.float32)
    input_ctex.requires_grad = True
    loss_f = get_loss_fun(spec_num)
    _ = model_pyt(input_ctex)  # to get the output of each layer.
    params, bias = extract_weights_bias(model_pyt, model_type=model_type)
    last_grad = grad_loss_last_layer(loss_f, hooker.layer_out[-2], params[-1], bias[-1], y_label)
    grad_v, grad_layer = grad_each_layer(params, hooker.layer_out, last_grad, model_type)
    layer_out = copy.deepcopy(hooker.layer_out)
    # clear the records in hooker
    hooker.clear_cache()
    return grad_layer, layer_out


def sort_neurons(grad_layer):
    """ sort neuron according to gradient
    Args:
        grad_layer: list. (layer_id,grads). Elements in the list are organized by the ascending order of layer_id
    Returns:
        sorted_idx: np.array, shape=(N,). The sorted indices of all neurons. 0-index
                    The neurons are sorted according the magnitude of its gradient by the descending order.
                    These indices are flattened,and N is the total number of neurons.
        signs:np.array, shape=(N,). The sign of the gradient of each neuron.
    """
    signs = []
    normalized_grads = []
    # sort layer by the ascending order of layer_id.
    grad_layer = sorted(grad_layer, key=lambda x: x[0])
    layer_size = []
    for layer_id, gradients in grad_layer:
        sign = np.sign(gradients)
        # normalized. Note that the sign of gradient dose not
        # imply the magnitude but direction, so the normalization
        # should be performed on the abstract values.
        normal_grad = np.abs(gradients)
        signs.extend(sign)
        normalized_grads.extend(normal_grad)
        layer_size.append((layer_id, len(gradients)))
    sorted_idx = np.argsort(normalized_grads)[::-1]
    return sorted_idx, np.array(signs), layer_size


def sort_neurons_mask(grad_layer):
    """ sort neuron according to gradient
    Args:
        grad_layer: list. (layer_id,grads). Elements in the list are organized by the ascending order of layer_id
    Returns:
        sorted_idx: np.array, shape=(N,). The sorted indices of all neurons. 0-index
                    The neurons are sorted according the magnitude of its gradient by the descending order.
                    These indices are flattened,and N is the total number of neurons.
        signs:np.array, shape=(N,). The sign of the gradient of each neuron.
    """
    signs = []
    normalized_grads = []
    # sort layer by the ascending order of layer_id.
    grad_layer = sorted(grad_layer, key=lambda x: x[0])
    layer_size = []
    for layer_id, gradients in grad_layer:
        sign = np.sign(gradients)
        # normalized. Note that the sign of gradient dose not
        # imply the magnitude but direction, so the normalization
        # should be performed on the abstract values.
        normal_grad = np.abs(gradients)
        signs.extend(sign)
        normalized_grads.extend(normal_grad)
        layer_size.append((layer_id, len(gradients)))
    sorted_idx = np.argsort(normalized_grads)[::-1]
    return sorted_idx, np.array(signs), layer_size


def count_uniq_neurons(neuron, uniq_pos):
    if neuron["pos"] in uniq_pos:
        uniq_pos[neuron["pos"]] += 1
    else:
        uniq_pos[neuron["pos"]] = 1
    return uniq_pos[neuron["pos"]]


def target_neuron(grad_layer, layer_out, uniq_pos, pos_bias, max_iter):
    """ select the neuron which has the biggest gradient.
    Args:
        grad_layer: list(tuple). the grad of each layer. The first is the grads on the input,
                    and the last element is the grads on the last masked layer.
        layer_out:  list(tuple). the output of each layer. The first is the output on the first layer,
                    and the last element is the output on the output layer.
    Returns:
        neuron: dict. (grad, (layer_id, nid)). The gradient, and the position of the selected neuron.
                Note that the layerid is 0-index considering the masked layer.
    """

    grads = []
    # sort layer by the ascending order of layer_id.
    grad_layer = sorted(grad_layer, key=lambda x: x[0])
    for layer_id, gradients in grad_layer:
        # skip the grads on the input
        if layer_id == 0:
            continue
        # skip the grads on the mask layer.Note that for the grad_layer,
        # the odd layer is the original linear layer.
        if layer_id % 2 == 0:
            continue
        for nid, grad in enumerate(gradients):
            # align the idx of grad_layer with the variable "layer_output".
            grads.append((grad, (layer_id - 1, nid)))
    grads = sorted(grads, key=lambda x: abs(x[0]), reverse=True)
    for max_grad in grads:
        lid = max_grad[1][0]
        nid = max_grad[1][1]
        n_out = layer_out[lid][nid]
        neuron = {}
        neuron["out"] = n_out
        neuron["pos"] = (lid, nid)
        neuron["grad"] = max_grad[0]
        itered = count_uniq_neurons(neuron, uniq_pos)
        if itered > max_iter:
            continue
        if itered > 1 and pos_bias[neuron["pos"]] == 0.0 and neuron["grad"] > 0:
            continue
        return neuron


def target_neuron_onnx(grad_layer, layer_out, uniq_pos, pos_bias, max_iter):
    """ select the neuron which has the biggest gradient.
    Note that the grad of grad_layer[0] is the grad of neurons on the layer before the first FC layer, and
    the grad of grad_layer[1] is the grad of neurons of the first FC layer
    """

    grads = []
    # sort layer by the ascending order of layer_id.
    grad_layer = sorted(grad_layer, key=lambda x: x[0])
    for layer_id, gradients in grad_layer:
        # skip the grads on the input
        if layer_id == 0:
            continue
        for nid, grad in enumerate(gradients):
            # align the idx of grad_layer with the variable "layer_output".
            grads.append((grad, (layer_id - 1, nid)))
    grads = sorted(grads, key=lambda x: abs(x[0]), reverse=True)
    for max_grad in grads:
        lid = max_grad[1][0]
        nid = max_grad[1][1]
        n_out = layer_out[lid][nid]
        neuron = {}
        neuron["out"] = n_out
        neuron["pos"] = (lid, nid)
        neuron["grad"] = max_grad[0]
        itered = count_uniq_neurons(neuron, uniq_pos)
        if itered > max_iter:
            continue
        if itered > 1 and pos_bias[neuron["pos"]] == 0.0 and neuron["grad"] > 0:
            continue
        return neuron
