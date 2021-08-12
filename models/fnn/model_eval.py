import sys

sys.path.append("../../")
from models.fnn.train import *


def get_input_shape(network_type, datatype):
    if network_type == Network.FNN:
        input_size = getattr(constant, f"{datatype.upper()}_INPUTSIZE")
        dshape = (input_size,)
    elif network_type == Network.LENET:
        dshape = (1, 28, 28) if datatype == DataType.MNIST else (3, 32, 32)
    else:
        raise Exception(f"Unknow network: {network_type}")
    return dshape


def eval(datatype, num_layer, hidden_size, batch_size, device):
    mid = f"{num_layer}_{hidden_size}"
    print(f"---------------{mid}------------------------")
    input_shape = get_input_shape(network_type=Network.FNN, datatype=datatype)
    train_dataloader, test_dataloader = load_data_loder(datatype, batch_size, input_shape)
    if datatype == DataType.MNIST:
        model = MnistNet(num_layer, hidden_size)
    elif datatype == DataType.CIFAR10:
        model = Cifar10Net(num_layer, hidden_size)
    else:
        raise Exception("")

    pyt_path = getattr(constant, f"{datatype.upper()}_PYMODEL_PATH").format(mid)
    model.load_state_dict(torch.load(os.path.join(PROJECT_PATH, pyt_path)))
    model = model.to(device)
    model.eval()
    # train_acc = test(model, train_dataloader, device)
    train_acc = 0.
    test_acc = test(model, test_dataloader, device)
    print(f"mid:n_{mid} train_acc:{train_acc:.4} test_acc:{test_acc:.4}")


if __name__ == '__main__':
    _datatype = sys.argv[1]
    _num_layer = int(sys.argv[2])
    device_id = int(sys.argv[3])
    device = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    eval(_datatype, _num_layer, 100, 64, device)
