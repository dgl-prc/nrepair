import sys

sys.path.append("../../")
import numpy as np
import csv
from models.fnn.fnn import MnistNet, Cifar10Net
from models.cnn.lenet import LeNet
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import copy
from utils.constant import *
import utils.constant as constant
from utils.time_util import current_timestamp


class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, shape):
        self.X = X
        self.Y = Y
        self.shape = shape
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEANS, std=MNIST_STDS)])

    def __getitem__(self, index):
        label = torch.tensor(int(self.Y[index]))
        img = torch.from_numpy(np.array(self.X[index], np.float32, copy=False))
        img.div_(255)
        img.sub_(0).div_(1)  # mean=0,std=1
        if self.shape == (1, 28, 28):
            img = np.array(self.X[index], copy=False).reshape(28, 28)
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.Y)


from PIL import Image
import torchvision.transforms as transforms
from utils.constant import CIFAR10_MEANS, CIFAR10_STDS

class CIFAR10Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, shape):
        self.X = X
        self.Y = Y
        self.shape = shape
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_MEANS, std=CIFAR10_STDS)])

    def __getitem__(self, index):
        label = torch.tensor(int(self.Y[index]))
        img = torch.from_numpy(np.array(self.X[index], np.float32, copy=False))
        img.div_(255)
        img.sub_(0.5).div_(1)  # mean=0.5,std=1 for all channels
        if self.shape == (3, 32, 32):
            img = np.array(self.X[index], copy=False).reshape(3, 32, 32)
            img = img.transpose((1, 2, 0))
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.Y)


def load_data(datatype, data_path, shape=None):
    new_data = []
    labels = []
    with open(data_path, 'r')  as f:
        data = csv.reader(f, delimiter=',')
        for i, label_image in enumerate(data):
            if i == 0:
                continue
            else:
                labels.append(label_image[0])
                image = label_image[1:len(label_image)]
                new_data.append(image)
    if datatype == DataType.MNIST:
        dataset = MNISTDataset(new_data, labels, shape)
    else:
        dataset = CIFAR10Dataset(new_data, labels, shape)
    return dataset


def test(model, test_dataloader, device):
    model.eval()
    model = model.to(device)
    acc = 0
    for i, (imgs, labels) in enumerate(test_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        acc += sum(preds == labels)
    return acc.item() / len(test_dataloader.dataset)


def load_data_loder(datatype, batch_size, shape):
    train_data = load_data(datatype, f"{PROJECT_PATH}data/{datatype}/{datatype}_train.csv", shape=shape)
    test_data = load_data(datatype, f"{PROJECT_PATH}data/{datatype}/{datatype}_test.csv", shape=shape)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def train(datatype, network_type, epochs, batch_size, device, fnn_args=None):
    if network_type == Network.FNN:
        num_layer = fnn_args[0]
        hidden_size = fnn_args[1]
        mid = f"{num_layer}_{hidden_size}"
        model = MnistNet(num_layer, hidden_size) if datatype == DataType.MNIST else Cifar10Net(num_layer, hidden_size)
        input_size = getattr(constant, f"{datatype.upper()}_INPUTSIZE")
        dshape = (input_size,)
    elif network_type == Network.LENET:
        mid = f"{network_type}"
        model = LeNet(1) if datatype == DataType.MNIST else LeNet(3)
        dshape = (1, 28, 28) if datatype == DataType.MNIST else (3, 32, 32)
    else:
        raise Exception(f"Unknow network: {network_type}")
    train_dataloader, test_dataloader = load_data_loder(datatype, batch_size, dshape)
    onnx_path = getattr(constant, f"{datatype.upper()}_MODEL_PATH").format(mid)
    pyt_path = getattr(constant, f"{datatype.upper()}_PYMODEL_PATH").format(mid)
    model = model.to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    test_acc = test(model, test_dataloader, device)
    print("epoch:{:3}\tacc:{:.4f}".format(0, test_acc))
    best_model = copy.deepcopy(model)
    best_acc = 0.
    for epoch in range(epochs):
        model.train()
        for i, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criteria(output, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(
                    "EPOCH {:3}\t{:5}/{:5}\tLOSS {:.4f}".format(epoch + 1, (i + 1) * batch_size,
                                                                len(train_dataloader.dataset),
                                                                loss))
        test_acc = test(model, test_dataloader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
        print("{} ====epoch:{:3}\tacc:{:.4f}=====".format(current_timestamp(), epoch + 1, test_acc))
    save_path_onnx = os.path.join(PROJECT_PATH, onnx_path)
    save_path_pyt = os.path.join(PROJECT_PATH, pyt_path)
    # save to onnx model
    dummy_input = torch.zeros(1, *dshape, dtype=torch.float32)
    torch.onnx.export(best_model.cpu(), dummy_input, save_path_onnx)
    # save to pytorch model
    torch.save(best_model.cpu().state_dict(), save_path_pyt)
    print(f"Done! test_acc:{best_acc}")
    print(f"Pytorch Model Saved in: {pyt_path}")
    print(f"ONNX Model Saved in: {onnx_path}")


if __name__ == '__main__':
    _datatype = sys.argv[1]
    _epochs = int(sys.argv[2])
    device_id = int(sys.argv[3])
    _network = sys.argv[4]
    if _network == Network.FNN:
        fnn_args = (int(sys.argv[5]), int(sys.argv[6]))  # _num_layer _h_size
    else:
        fnn_args = None
    _device = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    print(sys.argv)
    train(_datatype, _network, _epochs, batch_size=64, device=_device, fnn_args=fnn_args)
