from deeppoly.ai_milp import *
from deeppoly.verify_utils import estimate_grads
from utils.constant import VERIFY_DOMAIN

def __binary_split(specLB, specUB, model):
    grads = estimate_grads(specLB, specUB, model)
    smears = np.multiply(grads + 0.00001, [u - l for u, l in zip(specUB, specLB)])
    index = np.argmax(smears)
    m = (specLB[index] + specUB[index]) / 2
    return index, m


def verify_recursive(specLB, specUB, eran, domain, constraints, max_depth=10, depth=1):
    # print(f"============depth:{depth}=================")
    is_hold_deeppoly, nn, nlb, nub = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp,
                                                      config.use_default_heuristic, constraints)
    if is_hold_deeppoly:
        return is_hold_deeppoly
    elif depth <= max_depth:
        index, m = __binary_split(specLB, specUB, eran.model)
        left_rst = verify_recursive(specLB,
                                    [ub if i != index else m for i, ub in enumerate(specUB)],
                                    eran, domain, constraints,
                                    max_depth, depth + 1)

        if left_rst:
            right_rst = verify_recursive(
                [lb if i != index else m for i, lb in enumerate(specLB)],
                specUB, eran, domain, constraints,
                max_depth, depth + 1)
        return left_rst and right_rst
    else:
        is_hold_milp, adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
        return is_hold_milp


def search_ctex_recursively(specLB, specUB, eran, constraints, ctexs, b_specLB, b_specUB,
                            max_depth=10, depth=1):
    """
    Args:
        specLB:
        specUB:
        eran:
        constraints:
        ctexs: list(dict). {"specLB":specLB,"specUB":specUB,"cexp":cexp}
        b_specLB: the B-level split of the whole input space
        b_specUB: the B-level split of the whole input space
        max_depth:
        depth:

    Returns:

    """
    domain = VERIFY_DOMAIN
    is_ctex_found = False
    is_hold_deeppoly, nn, nlb, nub = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp,
                                                      config.use_default_heuristic, constraints)
    if is_hold_deeppoly:
        return is_hold_deeppoly, is_ctex_found
    elif depth <= max_depth:
        index, m = __binary_split(specLB, specUB, eran.model)
        # search in the left side
        left_rst, is_ctex_found = search_ctex_recursively(specLB,
                                                          [ub if i != index else m for i, ub in enumerate(specUB)],
                                                          eran, constraints,
                                                          ctexs,
                                                          b_specLB, b_specUB,
                                                          max_depth, depth + 1)
        if is_ctex_found:
            return left_rst, is_ctex_found
        else:
            # search in the right side if counterexample is not found in the left.
            right_rst, is_ctex_found = search_ctex_recursively(
                [lb if i != index else m for i, lb in enumerate(specLB)],
                specUB,
                eran, constraints,
                ctexs,
                b_specLB, b_specUB,
                max_depth, depth + 1)
            return left_rst and right_rst, is_ctex_found
    else:
        # reach the max depth of recursion
        is_hold_milp, ctex = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
        if not is_hold_milp:
            if ctex != None:
                # check if the adv_img is valid
                statisfy_constraint, _, nlb, nub = eran.analyze_box(ctex, ctex, domain, config.timeout_lp,
                                                                    config.timeout_milp, config.use_default_heuristic,
                                                                    constraints)
                if not statisfy_constraint:
                    is_ctex_found = True
                    ctexs.append({"specLB": b_specLB, "specUB": b_specUB, "cexp": ctex})
        return is_hold_milp, is_ctex_found
