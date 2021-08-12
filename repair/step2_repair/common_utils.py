from utils.time_util import current_timestamp
from repair.step2_repair.onnx_modify import *
from utils.constant import *
from utils.model_loader import *


def process_repair_result(**kwargs):
    ctex_id = kwargs["ctex_id"]
    repaired_nn = kwargs["repaired_nn"]
    is_success = kwargs["is_success"]
    dataset = kwargs["dataset"]
    mid = kwargs["mid"]
    eta = kwargs["eta"]
    save_timeid = kwargs["save_timeid"]
    iter_cnt = kwargs["iter_cnt"]
    uni_cnt = kwargs["uni_cnt"]
    repaired_case = kwargs["repaired_case"]
    zero_exceptions = kwargs["zero_exceptions"]
    avg_mn = kwargs["avg_mn"]
    avg_iter = kwargs["avg_iter"]
    costs = kwargs["costs"]
    cost = kwargs["cost"]
    repair_approach = kwargs["repair_approach"]

    if is_success == RepairResultType.SUCCESS:
        if dataset == DataType.ACASXU:
            interval_id = kwargs["interval_id"]
            spec_num = kwargs["spec_num"]
            save_path = os.path.join(PROJECT_PATH,
                                     ACASXU_REPAIRED_ONNX_PATH.format(dataset, repair_approach, mid, spec_num, interval_id,
                                                                      f"{eta:.4}",
                                                                      save_timeid))
        else:
            img_id = kwargs["img_id"]
            save_path = os.path.join(PROJECT_PATH,
                                     IMG_REPAIRED_ONNX_PATH.format(dataset, mid, img_id, f"{eta:.4}",
                                                                   save_timeid))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        onnx.save(repaired_nn,
                  os.path.join(save_path, f"repaired.onnx"))
        print(f"Successfully repaired. Total iteration:{iter_cnt}, Unique neurones modified:{uni_cnt}")
        print(f"{current_timestamp()} Repaired Model saved in {save_path}")
        repaired_case += 1
        avg_mn.append(uni_cnt)
        avg_iter.append(iter_cnt)
        costs.append(cost)
    else:
        if is_success == RepairResultType.FAILED_ZERO_OUTPUT:
            print(f"{current_timestamp()}>>>>>>>>>>>>>mid:{mid} ctext_id:{ctex_id} Repair Failed for zero layer-output<<<<<<<<<<<<<")
            zero_exceptions += 1
        elif is_success == RepairResultType.TIMEOUT:
            print(f"{current_timestamp()}>>>>>>>>>>>>>mid:{mid}  ctext_id:{ctex_id} Repair Failed for TIME OUT!<<<<<<<<<<<<<")
        else:
            print(f"{current_timestamp()}>>>>>>>>>>>>>mid:{mid}   ctext_id:{ctex_id} Repair Failed!<<<<<<<<<<<<<")

    return repaired_case, zero_exceptions


def print_overall_result(args, save_timeid, mid, costs, repaired_case, zero_exceptions, ctexes, avg_mn, avg_iter):
    print("------------------Done!-----------------------------")
    avg_ttime = np.mean([cost[0] for cost in costs])
    avg_eran_time = np.mean([cost[1] for cost in costs])

    print(f"Result-n_{mid}: args : {args}")
    print(f"Result-n_{mid}: save_timeid : {save_timeid}")
    print(f"Result-n_{mid}: {repaired_case}/{len(ctexes)} repaired")
    print(f"Result-n_{mid}: {zero_exceptions} failed for the zero layer output")
    print(f"Result-n_{mid}: Avg modified: {np.mean(avg_mn)}. Avg iter: {np.mean(avg_iter)}.")
    print(f"Result-n_{mid}: Avg total costs: {avg_ttime}. Avg ERAN time: {avg_eran_time}.")
    print("-----------------------------------------------")
