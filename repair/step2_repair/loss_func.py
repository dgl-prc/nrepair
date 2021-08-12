import torch


def loss_f1(output, labels):
    assert isinstance(output, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    return torch.nn.functional.cross_entropy(output, labels)


def loss_f2(output):
    assert isinstance(output, torch.Tensor)
    #     target_label is 0
    labels = torch.tensor([0])
    # the bigger the score of target label is, the bigger the loss is
    return -1 * torch.nn.functional.cross_entropy(output, labels)


def loss_f7(output):
    target_l1 = torch.tensor([3])
    target_l2 = torch.tensor([4])
    # y3 and y4 are never the minimum score
    loss1 = torch.nn.functional.cross_entropy(output, target_l1)
    loss2 = torch.nn.functional.cross_entropy(output, target_l2)
    return loss1 + loss2


def loss_f8(output):
    target_l1 = torch.tensor([0])
    target_l2 = torch.tensor([1])
    loss1 = -1 * torch.nn.functional.cross_entropy(output, target_l1)
    loss2 = -1 * torch.nn.functional.cross_entropy(output, target_l2)
    return loss1 + loss2


def loss_img(output, y_ture):
    labels = torch.tensor([y_ture])
    return torch.nn.functional.cross_entropy(output, labels)


def get_loss_fun(spec_num):
    if spec_num == 2:
        return loss_f2
    elif spec_num == 7:
        return loss_f7
    elif spec_num == 8:
        return loss_f8
    else:
        return loss_img
