from onnx import numpy_helper
import torch
from utils.constant import DataType

def onnx_name_covert_acasxu(params_name):
    """Convert the parameter name of onnx to pytorch.
    Only support the ACAS Xu model.
    <onnx_name, pytorch_name>
    <W0, linear1.weight> <B0, linear1.bias>
    Args:
        params_name: str. the parameter name of onnx
    Returns:
        the parameter name of pytorch
    """
    if params_name.startswith("MSK"):
        param_type = params_name[3]
        layer_no = int(params_name[4:])
        if param_type == "W":
            pyt_name = f"model.MSK{layer_no}.weight"  # layer number: 1-index
        elif param_type == "B":
            pyt_name = f"model.MSK{layer_no}.bias"
        else:
            raise Exception(f"Unsupported type:{param_type}")
    else:
        assert params_name.startswith("W") or params_name.startswith("B")
        param_type = params_name[0]
        layer_no = int(params_name[1:])
        if layer_no == 6:
            if params_name.startswith("W"):
                pyt_name = f"model.output.weight"  # layer number: 1-index
            elif param_type == "B":
                pyt_name = f"model.output.bias"
            else:
                raise Exception(f"Unsupported type:{param_type}")
        else:
            if params_name.startswith("W"):
                pyt_name = f"model.h{layer_no}.weight"  # layer number: 1-index
            elif param_type == "B":
                pyt_name = f"model.h{layer_no}.bias"
            else:
                raise Exception(f"Unsupported type:{param_type}")
    return pyt_name


def onnx_name_covert_mask(params_name):
    if params_name.startswith("model"):
        return params_name
    else:
        assert params_name.startswith("MSK")
        param_type = params_name[3]
        layer_no = int(params_name[4:])
        if param_type == "W":
            pyt_name = f"model.MSK{layer_no}.weight"  # layer number: 1-index
        elif param_type == "B":
            pyt_name = f"model.MSK{layer_no}.bias"
        else:
            raise Exception(f"Unsupported type:{param_type}")
        return pyt_name


def onnx2pyt(onnx_model, model_pyt, data_type):
    """
    Args:
        onnx_path: str. the path of onnx model
        model_pyt: an instance of pytorch model
        name_convert: function. map the name of each module of onnx to pytorch
    Returns:
        model_pyt
    """
    name_convert = onnx_name_covert_acasxu if data_type == DataType.ACASXU else onnx_name_covert_mask
    graph = onnx_model.graph
    initalizers = dict()
    for init in graph.initializer:
        initalizers[name_convert(init.name)] = numpy_helper.to_array(init)
    for name, p in model_pyt.named_parameters():
        p.data = (torch.from_numpy(initalizers[name])).data
    return model_pyt
