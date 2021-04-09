# ============================================
#               DoubleQTrainer
# ============================================
class DoubleQTrainer(FixedQTrainer):
    """
    Learning method from [van Hasselt et al. 2015][1].

    [1]: https://arxiv.org/abs/1509.06461v3
    """
    __name__ = "DoubleQTrainer"

    # -----
    # constructor
    # -----
    def __init__(
        self,
        agent: "ba.BaseAgent",
        lossFunctions: List,
        memory: "bm.BaseMemory",
        nets: List,
        optimizers: List,
        params: DictConfig,
        device: str
    ) -> None:
        super().__init__(agent, lossFunctions, memory, nets, optimizers, params, device)

    # -----
    # _get_targets
    # -----
    def _get_targets(
        self,
        nextStates: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Since learning in DQL is a bootstrapped process, the quality
        of the Q-value estimates depends on the state-action pairs
        that have been tried. This raises the following issue: when
        we choose an action for a given state, how do we know that
        that action is actually the best one?

        To deal with this, double DQL separates out the estimate of
        the Q-value for the chosen action in the current state and
        the determination of the best action to take in the next
        state by employing a target network.

        The DDQN version of the Bellman equation is:

            target = R + g * Q_target(ns, argmax(Q(ns, a)))

        Where R is the reward, g is the discount factor, ns is
        nextState, a is action.
        """
        # Use the primary network to get Q-values for the nextStates
        # Don't think I need no_grad since detach() is used in learn()
        # qNextPrimary should have shape (N, nActions)
        qNextPrimary = self.net(nextStates)
        # Select the index corresponding to the largest Q-value for
        # each sample. These are what the primary net believes to
        # be the best actions for the given nextStates
        # bestActions should have shape (N, 1)
        bestActions = torch.argmax(qNextPrimary, 1, keepdim=True)
        # Now get the Q-values as determined by the target net
        # qNext should have shape (N, nActions)
        qNext = self.targetNet(nextStates)
        # Just as in normal dqn, we only want to update the Q-values
        # corresponding to the actions chosen; it's just that here
        # those actions were determined from the primary net
        oneHot = (
            qNext
            * torch.nn.functional.one_hot(
                bestActions.to(torch.int64), qNext.shape[1]
            ).squeeze()
        )
        # qChosen has shape (N, 1)
        qNextMax = torch.sum(oneHot, 1, keepdims=True)
        maskedVals = (1.0 - dones) * qNextMax
        targets = rewards + self.discountRate * maskedVals
        return targets
