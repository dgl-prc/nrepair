import sys

sys.path.append("../../")
from repair.step3_eval.eval_performance import *
from utils.data_loader import *
from utils.model_loader import *
from utils.time_util import current_timestamp
import re


def extract_interval_id(x):
    pattern = re.compile("interval_(\d+).onnx")
    id = int(pattern.search(x).group(1))
    return id


def extract_img_id(x):
    pattern = re.compile("img-(\d+)")
    id = int(pattern.search(x).group(1))
    return id

def extracted_image_repair_model_path(is_batch, dataset, mid, epsilon, file_path=None):
    """ Note that both file 'repair/step2_repair/logs/search_epsilon_{dataset}_saved_paths.log'
        and 'data/{}/repaired_onnx/summary_{}_{}.log' are generated via the following command:
         'cat logs/{repair.log}|grep "Repaired Model saved"|awk '{print $7}' >> {paths.log}'
    Args:
        is_batch:
        dataset:
        mid:
        epsilon:
    Returns:
    """
    if file_path is None:
        if is_batch:
            # file_path = os.path.join(PROJECT_PATH, f"repair/step2_repair/logs/search_epsilon_{dataset}_saved_paths.log")
            file_path = os.path.join(PROJECT_PATH,
                                     f"repair/step2_repair/logs/search_epsilon_{dataset}_manner3_saved_paths.log")
        else:
            reapired_onnx_paths = SUMMARY_REPAIRED_ONNX_PATH.format(dataset, mid, epsilon)
            file_path = os.path.join(PROJECT_PATH, reapired_onnx_paths)
    print(f"The models path:{file_path}")
    with open(file_path, "r") as f:
        model_paths = [os.path.join(e.strip(), "repaired.onnx") for e in f.readlines()]
    if is_batch:
        target_paths = [p for p in model_paths if p.__contains__(f"n_{mid}") and p.__contains__(f"/epsilon_{epsilon}/")]
        model_paths = target_paths

    return model_paths


def eval_imgs_gloablly(dataset, mid, epsilon, model_paths):
    mtype = ModelType.ONNX
    print("eval gloablly", dataset, mid, epsilon, mtype)
    avg_accr = []
    avg_netacc = []
    print(f"{current_timestamp()}========Prepare data for model {mid}===={epsilon}=============")
    tests, ori_correct_idx = prepare_img_testing_data(dataset, mid, mtype)
    print(f"{current_timestamp()}========Testing reaired models....=================")
    for i, repaired_path in enumerate(model_paths):
        print(f"{current_timestamp()} -------------> Progress-model {i + 1}/{len(model_paths)} <----------------")
        repaired_model = load_model(repaired_path, mtype)
        acc_r, net_acc = calculate_accr(tests, repaired_model, ori_correct_idx, mtype)
        avg_accr.append(acc_r)
        avg_netacc.append(net_acc)
        print(f"avg_accr:{acc_r},net_acc:{net_acc}")
    print(f"{current_timestamp()}===============End================")
    print(f"Result-n_{mid}: args : {dataset, mid, epsilon}")
    print(f"Result-n_{mid}: Avg: n_{mid} accR:{np.mean(avg_accr):.4} netAcc:{np.mean(avg_netacc):.4}")


def eval_imgs_gloablly_pyt(dataset, mid, epsilon, model_paths, mtype, testing_data=None):
    from repair.step2_repair.onnx2pyt import onnx2pyt
    from models.fnn.masked_fnn import MaskedNet
    print("eval gloablly", dataset, mid, epsilon, mtype)
    avg_accr = []
    avg_netacc = []
    if testing_data is None:
        print(f"{current_timestamp()}========Prepare data for model {mid}===={epsilon}=============")
        tests, ori_correct_idx = prepare_img_testing_data(dataset, mid, mtype)
    else:
        tests, ori_correct_idx = testing_data
    print(f"{current_timestamp()}========Testing reaired models....=================")
    for i, repaired_path in enumerate(model_paths):
        print(f"{current_timestamp()} -------------> Progress-model {i + 1}/{len(model_paths)} <----------------")
        repaired_model = load_model(repaired_path, mtype)
        pyt_model = MaskedNet(int(mid.split("_")[0]), int(mid.split("_")[1]), dataset)
        onnx2pyt(repaired_model, pyt_model, dataset)
        acc_r, net_acc = calculate_accr(tests, pyt_model, ori_correct_idx, ModelType.PYTORCH)
        avg_accr.append(acc_r)
        avg_netacc.append(net_acc)
    print(f"{current_timestamp()}===============End================")
    print(f"Result-n_{mid}: args : {dataset, mid, epsilon}")
    print(f"Result-n_{mid}: Avg: n_{mid} accR:{np.mean(avg_accr):.4} netAcc:{np.mean(avg_netacc):.4}")


def eval_imgs_locally(dataset, mid, epsilon, model_paths, mtype):
    print("eval locally", dataset, mid, epsilon, mtype)
    pbt = 0.03 if dataset == DataType.MNIST else 0.0012
    print(f"{current_timestamp()}========Prepare data for model {mid}===={epsilon}=============")
    img_ids, input_contrains, output_constraints, ctexes = load_constraints_ctex(dataset=dataset, mid=mid, pbt=pbt)
    print(f"{current_timestamp()}========Testing reaired models....=================")
    for i, repaired_path in enumerate(model_paths):
        img_id = int(extract_img_id(repaired_path))
        idx = img_ids.index(img_id)
        input_contrain = input_contrains[idx]
        lb, ub = input_contrain[0], input_contrain[1]
        y_true = output_constraints[idx]
        repaired_model = load_model(repaired_path, mtype)
        is_qualified = eval_image_by_synthesize_data(repaired_model, lb, ub, y_true, dataset, test_size=1000,
                                                     rnd_seed=20210101)
        if is_qualified:
            print("Fidelity 100%")
    print(f"{current_timestamp()}===============End================")


if __name__ == '__main__':
    _dataset = sys.argv[1]
    _repair_type = sys.argv[2]
    if _repair_type == RepairType.MM:
        _is_normal = bool(int(sys.argv[3]))
        _epsilon = "MM_normal" if _is_normal else "MM"
    else:
        _epsilon = sys.argv[3]
    _mid = sys.argv[4]
    assert _dataset in [DataType.MNIST, DataType.CIFAR10]
    model_paths = extracted_image_repair_model_path(is_batch=False, dataset=_dataset, mid=_mid, epsilon=_epsilon)
    # eval_imgs_locally(_dataset, _mid, _epsilon, model_paths)
    eval_imgs_gloablly(_dataset, _mid, _epsilon, model_paths)
