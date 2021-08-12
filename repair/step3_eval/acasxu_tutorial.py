import sys

sys.path.append("../../")

from utils.time_util import current_timestamp
from repair.step3_eval.eval_performance import synthesize_acasxu_data, acasxu_over_all
from utils.data_loader import *
from utils.model_loader import load_acasxu_repaired, load_ori_onnx
import numpy as np
import onnx

dataset = DataType.ACASXU


def eval_acasxu(dataset, mid, spec_num, epsilon, timeid, repair_type):
    onnx_paths = load_acasxu_repaired(spec_num, mid, epsilon, timeid, repair_type)
    input_lb, input_ub = load_acasxu_inputs_constraints(spec_num)
    ori_model = load_ori_onnx(dataset, mid)
    print(f'{current_timestamp()}-----------{mid}---{spec_num}-----{timeid}--eps:{epsilon}---')
    input_contraints, _, _ = load_constraints_ctex(mid=mid, dataset=dataset, spec_num=spec_num)
    repaired_models = []
    repaired_regions = []
    for interval_id, file_path in onnx_paths:
        model = onnx.load(file_path)
        repaired_models.append(model)
        repaired_regions.append(input_contraints[interval_id])
    test_data = synthesize_acasxu_data(ori_model, spec_num, [(input_lb, input_ub)], size=10000)
    fdlt, subregion_inputs = acasxu_over_all(test_data, ori_model, repaired_models, repaired_regions)
    print(f"{current_timestamp()} Result-n_{mid}: args : {dataset, mid, epsilon}")
    print(f"{current_timestamp()} Result-n_{mid}: overall fdlt:{fdlt * 100:.4}% subregion_inputs:{subregion_inputs}")


def get_saved_mids_timeids(repair_type, spec_num):
    if repair_type == RepairType.nRepair:
        if spec_num == 2:
            timeids = [
                "20210116131010",
                "20210116132651",
                "20210116140736",
                "20210116141604",
                "20210116142359",
                "20210116150301",
                "20210116152328",
                "20210116175526",
                "20210116184626",
                "20210116211208",
                "20210116213056",
                "20210116213519",
                "20210116222458",
                "20210116223949",
                "20210116235156",
                "20210117020541",
                "20210117023617",
                "20210117073259",
                "20210117073653",
                "20210117090010",
                "20210117091449",
                "20210117092231",
                "20210117095126",
                "20210117100032",
                "20210117125026",
                "20210117191212",
                "20210117192135",
                "20210117192559",
                "20210117192724",
                "20210117193235",
                "20210117193851",
                "20210117195136",
                "20210117201648",
                "20210117224220"]
        elif spec_num == 7:
            pass
        else:
            assert spec_num == 8
            pass
    else:
        assert repair_type == RepairType.MM
        if spec_num == 2:
            timeids = ["20210724234614", "20210725004625", "20210725072244", "20210725234723", "20210726152957",
                       "20210727100101", "20210727131016", "20210727171632", "20210727221048",  # 2 end
                       "20210726193523", "20210727223929", "20210727223915", "20210602221255", "20210604230205",
                       "20210604230153", "20210603111553", "20210603113403",  # 3-end
                       "20210726211659", "20210726222610", "20210727033827", "20210727064142", "20210727094002",
                       "20210727095813", "20210727124027", "20210727195247",  # 4-end
                       "20210726211528", "20210727013045", "20210727064002", "20210604230258", "20210607000609",
                       "20210607115435", "20210604230332", "20210606224808", "20210607104235"
                       ]
        elif spec_num == 7:
            timeids = []
        else:
            assert spec_num == 8
            timeids = ["20210607150215"]

    if spec_num == 2:
        mids = ["2_1", "2_2", "2_3", "2_4", "2_5", "2_6", "2_7", "2_8", "2_9",
                "3_1", "3_2", "3_4", "3_5", "3_6", "3_7", "3_8", "3_9",
                "4_1", "4_3", "4_4", "4_5", "4_6", "4_7", "4_8", "4_9",
                "5_1", "5_2", "5_3", "5_4", "5_5", "5_6", "5_7", "5_8", "5_9"
                ]
    elif spec_num == 7:
        mids = ["1_9"]
    else:
        mids = ["2_9"]
    return mids, timeids


def eval_acaxu_nrepair(repair_type, spec_num):
    epsilon = 0.35 if repair_type == RepairType.nRepair else np.inf
    mids, timeids = get_saved_mids_timeids(repair_type, spec_num)
    for mid, timeid in zip(mids, timeids):
        eval_acasxu(dataset, mid, spec_num, epsilon, timeid, repair_type)


if __name__ == '__main__':
    _repair_type = sys.argv[1]
    _spec_num = int(sys.argv[2])
    eval_acaxu_nrepair(_repair_type, _spec_num)
