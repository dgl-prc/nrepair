import sys

sys.path.append("../../")
from repair.step2_repair.nn_repair import *


def search_img(data_set, uniq_ratio, maxiter):
    if data_set == DataType.CIFAR10:
        _pbt = 0.0012
    else:
        _pbt = 0.03
    for _lnum in [3, 5, 7]:
        for _epsilon in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            tutorial_images(data_set, _lnum, 100, _pbt, _epsilon, uniq_ratio, maxiter)


def search_acasxu(uniq_ratio, maxiter):
    # for _spec, _mid in [(2,"3_8"),(7,"1_9"),(8,"2_9")]:
    for _spec, _mid in [(8, "2_9")]:
        # eps_lst = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        eps_lst = [0.25, 0.2, 0.15, 0.1]
        # eps_lst = [0.05, 0.1, 0.15, 0.2, 0.25]
        for _epsilon in eps_lst:
            tutorial_acasxu(_spec, _mid, _epsilon, uniq_ratio, maxiter)

if __name__ == '__main__':
    _maxiter = 50
    _uniq_ratio = 0.05
    _data_set = sys.argv[1]
    if _data_set == DataType.ACASXU:
        search_acasxu(_uniq_ratio, _maxiter)
    else:
        search_img(_data_set, _uniq_ratio, _maxiter)
