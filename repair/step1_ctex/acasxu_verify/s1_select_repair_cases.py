"""
The "type-1 subspaces" refers to these subspaces which are
failed to verify with the vanilla deeppoly, i.e., without recursively verification or milp.

The "type-2 subspaces" refers to these subspaces which are
failed to verify (with recursively verification and  milp) and no counterexamples are found.
"""
import sys

sys.path.append("../../")
from repair.step1_ctex.deeppoly_api import verify_recursive
from repair.step1_ctex.acasxu_verify.split_input_spaces import *
from utils.data_loader import *
from utils.help_func import save_pickle
from utils.time_util import *


def estimate_repair_cases(spec_num, aid, tid, recursive, max_depth, *kwargs):
    """ roughly count the subspaces which need to be re-verified recursively.
    With these subspaces, we can roughly estimate the number of failed subspaces in which counterexamples are not found.
    Args:
        spec_num:. property id.
        aid:
        tid:
        recursive:
    Returns:
        num_subspaces: int. The total number of subspaces divided from the original input constraints.
        failed_subspaces: list. The subspaces that are failed to verify, i.e., need to be re-verified recursively.
    """
    if len(kwargs) == 3:
        eran, normalized_subspaces, output_constraints = kwargs[0], kwargs[1], kwargs[2],
    else:
        eran, normalized_subspaces, output_constraints = split_input_spaces(spec_num, aid, tid)
    num_subspaces = len(normalized_subspaces)
    failed_subspaces = []
    i = 0
    for subspace in normalized_subspaces:
        specLB, specUB = subspace
        if recursive:
            is_hold = verify_recursive(specLB, specUB, eran, VERIFY_DOMAIN, output_constraints, max_depth=max_depth,
                                       depth=1)
        else:
            is_hold, _, _, _ = eran.analyze_box(specLB, specUB, VERIFY_DOMAIN, config.timeout_lp,
                                                config.timeout_milp, config.use_default_heuristic,
                                                output_constraints)
        if not is_hold:
            failed_subspaces.append(subspace)
        i += 1
        print(f"{current_timestamp()} Progress {i}/{num_subspaces} {is_hold}")
    return num_subspaces, failed_subspaces


def search_p2(max_depth, recursive):
    # nets which have type-2 subspaces
    high_priority_nets = [(3, 8), (4, 1),
                          (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
                          (5, 1), (5, 3), (5, 5), (5, 6), (5, 8), (5, 9)
                          ]

    spec_num = 2
    raw_input_constraints = load_acasxu_inputs_constraints(spec_num)
    mids = [(aid, tid) for aid in range(2, 6, 1) for tid in range(1, 10, 1)]
    low_priority_nets = [ele for ele in mids if not ele in high_priority_nets]
    new_mids = high_priority_nets + low_priority_nets
    new_mids.remove((2, 1))
    for aid, tid in new_mids:
        print(f"============{current_timestamp()}============={aid}=={tid}===========================")
        eran, subspaces, constraints = split_input_spaces(spec_num, aid, tid, copy.deepcopy(raw_input_constraints))
        num_subspaces, failed_subspaces = estimate_repair_cases(spec_num, aid, tid, recursive, max_depth, eran,
                                                                subspaces, constraints)
        print(f"n_{aid}{tid}, total: {num_subspaces}, failed: {len(failed_subspaces)}")
        if len(failed_subspaces) > 0:
            save_path = os.path.join(PROJECT_PATH, FAILED_SPACES.format(spec_num, f"{aid}_{tid}"))
            save_pickle(save_path, failed_subspaces)
    print(f"===={current_timestamp()}======DONE!===============")

if __name__ == '__main__':
    # search_p2(max_depth=10, recursive=True)
    # test_in_batch()
    _spec_num = sys.argv[1]
    _aid = sys.argv[2]
    _tid = sys.argv[3]
    _recursive = bool(int(sys.argv[4]))
    _max_depth = int(sys.argv[5])
    config.timeout_milp = 1
    print(_spec_num, _aid, _tid, _recursive, _max_depth, config.timeout_milp)
    num_subspaces, failed_subspaces = estimate_repair_cases(_spec_num, _aid, _tid, _recursive, _max_depth)
    print(f"total: {num_subspaces}, failed: {len(failed_subspaces)}")
    if len(failed_subspaces) > 0:
        save_path = os.path.join(PROJECT_PATH, FAILED_SPACES.format(_spec_num, f"{_aid}_{_tid}"))
        save_pickle(save_path, failed_subspaces)
