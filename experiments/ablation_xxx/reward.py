from verl.utils.reward_score.re_search import compute_score as default_compute_score


def compute_score(solution_str: str, ground_truth, tokenizer=None):
    """Example experiment reward hook.

    Copy this file and edit the logic when you want to ablate the reward without
    touching the shared runtime code.
    """
    return default_compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        tokenizer=tokenizer,
    )
