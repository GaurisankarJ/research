from verl.utils.reward_score.r1_searcher_answer import compute_score as default_compute_score


def compute_score(solution_str: str, ground_truth, tokenizer=None):
    """Ablation zero reward hook.

    This reward hook is used to ablate the reward function.
    """
    return default_compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        tokenizer=tokenizer,
    )
