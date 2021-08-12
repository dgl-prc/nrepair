import os
import socket
import sys

sys.path.append("../../")
cpu_affinity = os.sched_getaffinity(0)
server_name = socket.gethostname()

if server_name == "Yserver":
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/python_interface/')
    sys.path.insert(0, '/home/dgl/project/eran/ELINA/deepg/code/')
else:
    sys.path.insert(0, '../../env_setup/ELINA/python_interface/')
    sys.path.insert(0, '../../env_setup/ELINA/deepg/code/')

from image_deeppoly.ai_milp import *
from image_deeppoly.config import config
from image_deeppoly.constraint_utils import *
from image_deeppoly.eran import ERAN
from utils.help_func import *
from utils.time_util import *
from utils.data_loader import *
from utils.model_loader import load_ori_onnx


def get_ori_pred(eran, image, data_type, is_conv):
    specLB = normalize(image, MEANS, STDS, data_type, is_conv)
    specUB = normalize(image, MEANS, STDS, data_type, is_conv)
    label, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, VERIFY_DOMAIN, config.timeout_lp,
                                                 config.timeout_milp, config.use_default_heuristic)

    return label


def verifiy_single_img(spec_lb, spec_ub, eran, label, return_adv=False):
    perturbed_label, nn, nlb, nub, failed_labels, x = eran.analyze_box(spec_lb, spec_ub, VERIFY_DOMAIN,
                                                                       config.timeout_lp,
                                                                       config.timeout_milp,
                                                                       config.use_default_heuristic, label=label,
                                                                       prop=-1, K=0, s=0,
                                                                       timeout_final_lp=config.timeout_final_lp,
                                                                       timeout_final_milp=config.timeout_final_milp,
                                                                       use_milp=True,
                                                                       complete=True,
                                                                       terminate_on_failure=False,
                                                                       partial_milp=0,
                                                                       max_milp_neurons=0,
                                                                       approx_k=0
                                                                       )

    if perturbed_label == label:
        return True, (None, label)
    else:
        assert failed_labels is not None
        failed_labels = list(set(failed_labels))
        constraints = get_constraints_for_dominant_label(label, failed_labels)
        is_verified, adv_image, adv_val = verify_network_with_milp(nn, spec_lb, spec_ub, nlb, nub, constraints)
        cex_label = -1
        if return_adv and adv_image != None:
            cex_label, _, _, _, _, _ = eran.analyze_box(adv_image[0], adv_image[0], VERIFY_DOMAIN,
                                                        config.timeout_lp, config.timeout_milp,
                                                        config.use_default_heuristic)
            if cex_label == label:
                adv_image == None
        return is_verified, (adv_image, cex_label)


def search_img_ctex(dataset, tests, eran, is_conv, pbt, i_start=0, max_ctexs=200):
    """
    Args:
        tests:
        eran:
        pbt: perturbation

    Returns:
    """
    use_milp = True
    complete = True
    print("----------------Args in search_img_ctex---------------")
    print(dataset, tests, eran, pbt, i_start, max_ctexs)
    ctexs = []
    domain = VERIFY_DOMAIN
    verified_images = 0
    folder_id = folder_timestamp()
    acc_cnt = 0
    for i, test in enumerate(tests):
        if i < i_start:
            continue
        image = np.float64(test[1:len(test)]) / np.float64(255)
        label = get_ori_pred(eran, image, dataset, is_conv)
        # only consider the correctly predicted.
        if label != int(test[0]):
            print(f"{current_timestamp()} Wrongly classified skipped! correct {acc_cnt}/{i + 1} Image: {i}(0-index)")
            continue
        acc_cnt += 1
        #########
        # debug
        #########
        # print(f"acc: {acc_cnt}/{i+1}")
        # continue
        specLB = np.clip(image - pbt, 0, 1)
        specUB = np.clip(image + pbt, 0, 1)
        specLB = normalize(specLB, MEANS, STDS, dataset, is_conv)
        specUB = normalize(specUB, MEANS, STDS, dataset, is_conv)
        perturbed_label, nn, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, domain, config.timeout_lp,
                                                                           config.timeout_milp,
                                                                           config.use_default_heuristic, label=label,
                                                                           prop=-1, K=0, s=0,
                                                                           timeout_final_lp=config.timeout_final_lp,
                                                                           timeout_final_milp=config.timeout_final_milp,
                                                                           use_milp=use_milp,
                                                                           complete=complete,
                                                                           terminate_on_failure=not complete,
                                                                           partial_milp=0,
                                                                           max_milp_neurons=0,
                                                                           approx_k=0
                                                                           )

        info = ""
        is_verified = False
        found_ctex = False
        if perturbed_label == label:
            info = f"Verified"
            verified_images += 1
            is_verified = True
        else:
            if failed_labels is not None:
                failed_labels = list(set(failed_labels))
                constraints = get_constraints_for_dominant_label(label, failed_labels)
                is_verified, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                if is_verified:
                    info = f"Verified as Safe using MILP. Label: {label}"
                    verified_images += 1
                    is_verified = True
                else:
                    if adv_image != None:
                        cex_label, _, _, _, _, _ = eran.analyze_box(adv_image[0], adv_image[0], domain,
                                                                    config.timeout_lp, config.timeout_milp,
                                                                    config.use_default_heuristic)
                        if cex_label != label:
                            ctexs.append({"img_id": i, "y_true": label, "y_adv": cex_label, "cexp": adv_image[0],
                                          "specLB": specLB, "specUB": specUB})
                            found_ctex = True
                            info = f"Verified unsafe against label {cex_label}, correct label {label}"
                        else:
                            info = "Failed with MILP, without a adeversarial example"
                    else:
                        info = "Failed with MILP"
            else:
                if x != None:
                    cex_label, _, _, _, _, _ = eran.analyze_box(x, x, 'deepzono', config.timeout_lp,
                                                                config.timeout_milp,
                                                                config.use_default_heuristic,
                                                                approx_k=config.approx_k)
                    print("cex label ", cex_label, "label ", label)
                    if (cex_label != label):
                        ctexs.append({"img_id": i, "y_true": label, "y_adv": cex_label, "cexp": x,
                                      "specLB": specLB, "specUB": specUB})
                        found_ctex = True
                        info = f"Verified unsafe against label {cex_label}, correct label {label}"
                    else:
                        info = "Failed with MILP, without a adeversarial example"
                else:
                    info = "Failed"

        print(
            f"{current_timestamp()} correct {acc_cnt}/{i + 1} Image: {i}(0-index)  Verified: {is_verified} {info}. Counterexample found:{found_ctex}")
        if i % 30 == 0:
            tmp_path = f"./check_points/{dataset}/{folder_id}/n_{mid}.pkl"
            save_pickle(tmp_path, ctexs)
        if len(ctexs) >= max_ctexs:
            break
    print("acc_cnt:{}".format(acc_cnt))
    print("-----------------------------")
    print(f"Total: {len(ctexs)} found!")
    if len(ctexs) > 0:
        VERIFIED_CTEX = getattr(constant, f"VERIFIED_CTEX_{dataset.upper()}")
        save_path = os.path.join(PROJECT_PATH, VERIFIED_CTEX.format(f"{pbt:.4}", mid))
        save_pickle(save_path, ctexs)
        print(f"Saved in {save_path}")
    else:
        print("Nothing Saved!")


if __name__ == '__main__':
    _dataset = sys.argv[1]
    _network = sys.argv[2]
    _pbt = float(sys.argv[3])
    if _network == Network.FNN:
        _lid = sys.argv[4]
        _nid = sys.argv[5]
        mid = f"{_lid}_{_nid}"
        _is_conv = False
    else:
        mid = f"{_network}"
        _is_conv = True
    is_onnx = True
    print(sys.argv)
    model = load_ori_onnx(_dataset, mid)
    _eran = ERAN(model, is_onnx=is_onnx)
    _tests = load_img_tests(_dataset)
    MEANS = config.mean if config.mean is not None else getattr(constant, f"{_dataset}_MEANS".upper())
    STDS = config.std if config.std is not None else getattr(constant, f"{_dataset}_STDS".upper())
    search_img_ctex(_dataset, _tests, _eran, _is_conv, pbt=_pbt)
