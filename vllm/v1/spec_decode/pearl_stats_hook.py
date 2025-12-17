# PEARL Acceptance Stats Integration Hook
#
# This file provides a hook for integrating PEARL's acceptance stats
# with vLLM's scheduler verification logic.
#
# USAGE:
# In vllm/v1/core/sched/scheduler.py, after calling observe_draft():
#
# ```python
# if spec_config.method == "pearl":
#     from vllm.v1.spec_decode.pearl_stats_hook import update_pearl_stats
#     update_pearl_stats(
#         model_runner=model_runner,
#         num_accepted_tokens=num_accepted_tokens,
#         num_draft_tokens=num_draft_tokens
#     )
# ```

from vllm.logger import init_logger

logger = init_logger(__name__)

_pearl_proposer = None

def register_pearl_proposer(proposer):
    """Register PEARL proposer for stats updates."""
    global _pearl_proposer
    _pearl_proposer = proposer
    logger.info("[PEARL] Proposer registered for stats tracking")

def update_pearl_stats(num_accepted_tokens: int, num_draft_tokens: int):
    """
    Update PEARL acceptance statistics.

    This should be called after draft token verification in the scheduler.

    Args:
        num_accepted_tokens: Number of draft tokens that were accepted
        num_draft_tokens: Total number of draft tokens proposed
    """
    global _pearl_proposer

    if _pearl_proposer is None:
        logger.warning("[PEARL] Proposer not registered for stats tracking")
        return

    try:
        _pearl_proposer.update_acceptance_stats(num_accepted_tokens, num_draft_tokens)

        logger.debug(
            f"[PEARL] Stats updated: {num_accepted_tokens}/{num_draft_tokens} accepted "
            f"({num_accepted_tokens/max(num_draft_tokens,1)*100:.1f}%)"
        )
    except Exception as e:
        logger.error(f"[PEARL] Error updating stats: {e}")
