import random
from .base import ProposerStrategy

class ConcedingProposer(ProposerStrategy):
    """
    This proposer strategy starts at the top and decrements until its offer is accepted
    It will stay at the same level if its offer is accepted, and will never drop below 
    the minimum offer allowed in the game
    """
    def __init__(self, start_offer, decrement):
        self.current_offer = start_offer
        self.decrement = decrement

    def propose(self, game, self_player):
        return max(game.min_offer, min(self.current_offer, game.max_offer))

    def receive_feedback(self, accepted, offer, game):
        if not accepted:
            self.current_offer -= self.decrement

class RandomProposer(ProposerStrategy):
    """
    This proposer strategy selects a random offer from either a uniform (default)
    or normal distribution. If normal, mean and stddev must be provided
    """
    def __init__(self, distribution="uniform", *, mean=None, stddev=None):
        self.distribution = distribution
        self.mean = mean
        self.stddev = stddev

    def receive_feedback(self, accepted, offer, game):
        pass

    def propose(self, game, self_player):
        if self.distribution == "uniform":
            offer = random.uniform(game.min_offer, game.max_offer)
        elif self.distribution == "normal":
            assert self.mean is not None and self.stddev is not None, "Normal distribution requires mean and stddev"
            offer = random.gauss(self.mean, self.stddev)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Offers must be valid
        return max(game.min_offer, min(offer, game.max_offer))

class TitForTatProposer(ProposerStrategy):
    """
    This proposer strategy involves copying the opponents most recent offer to its player.
    If both players are playing this strategy, we can get social utility maximization
    although this strategy is not generally a best response
    """
    def __init__(self):
        self.last_opponent_offer = None

    def propose(self, game, self_player):
        if self.last_opponent_offer is None:
            return 0
        return self.last_opponent_offer

    def receive_feedback(self, accepted, offer, game):
        pass

    def observe_offer_from_opponent(self, offer):
        if offer != 0:
            self.last_opponent_offer = offer

class LearningProposer(ProposerStrategy):
    """
    This is the first version of a learning proposer strategy. It is too jumpy 
    and is completely dominated by BinarySearch proposer strategy.
    """
    def __init__(self, start_offer, step_size):
        self.current_offer = start_offer
        self.step_size = step_size

    def propose(self, game, self_player):
        return max(game.min_offer, min(self.current_offer, game.max_offer))

    def receive_feedback(self, accepted, offer, game):
        if accepted:
            self.current_offer += self.step_size
        else:
            self.current_offer -= self.step_size

class BinarySearchProposer(ProposerStrategy):
    """
    This proposer strategy relies on finding the best offer for its player that 
    the opponent will accept. When played against a utilitarian responder, it leads
    to the "perfect information" outcome, which may be a PSNE among naive strategies
    although it can be confused by probabilistic or strategic responders. 
    This class likely needs some tweaking for behavior against such opponents.
    """
    def __init__(self, min_offer, max_offer):
        self.low = min_offer
        self.high = max_offer
        self.last_offer = (min_offer + max_offer) / 2

    def propose(self, game, self_player):
        self.last_offer = (self.low + self.high) / 2
        return self.last_offer

    def receive_feedback(self, accepted, offer, game):
        if accepted:
            self.low = max(self.low, offer)
        else:
            self.high = min(self.high, offer)

class RiskAwareProposer(ProposerStrategy):
    """
    This proposer attempts to predict the utility it will get from making each of
    a suite of different proposals, and offers the one it believes will maximize 
    its utility on the current turn. It is the first proposer that attempts to 
    consider the bad thing that might happen (i.e. rejection cost), although it 
    remains short-term in its thinking
    """
    def __init__(self, offer_levels=None):
        self.offer_levels = offer_levels or list(range(0, 101, 10))
        self.stats = {x: {"trials": 0, "accepts": 0} for x in self.offer_levels}
        self.rejection_penalties = []

    def propose(self, game, self_player):
        best_offer = None
        best_expected_utility = float("-inf")

        if self_player == game.player1:
            default_reject_utility = game.player_1_bad + game.proposer_bad
        else:
            default_reject_utility = game.player_2_bad + game.proposer_bad

        # Estimate expected utility for each offer level
        for offer in self.offer_levels:
            stats = self.stats[offer]
            trials = stats["trials"]
            accepts = stats["accepts"]

            # Estimate acceptance probability
            if trials == 0:
                p_accept = 0.5  # Naive
            else:
                p_accept = accepts / trials

            # Estimate rejection penalty (average of past penalties)
            if self.rejection_penalties:
                avg_reject_utility = sum(self.rejection_penalties) / len(self.rejection_penalties)
            else:
                avg_reject_utility = default_reject_utility

            expected_utility = p_accept * offer + (1 - p_accept) * avg_reject_utility

            if expected_utility > best_expected_utility:
                best_expected_utility = expected_utility
                best_offer = offer

        return best_offer

    def receive_feedback(self, accepted, offer, game):
        stats = self.stats[offer]
        stats["trials"] += 1
        if accepted:
            stats["accepts"] += 1
        else:
            penalty = game.get_bad_utility(player=game.proposer, is_proposer=True)
            self.rejection_penalties.append(penalty)

class ForwardLookingProposer:
    """
    This proposer class is a work in progress and has not been tested, 
    I don't even know if it runs. The idea is that it attempts to consider 
    the impact of rejection not just on the current round but also on future rounds.
    This class was created to be particularly responsive to shifting p when rejection occurs
    and is likely to be more risk averse than other models
    """
    def __init__(
        self,
        offer_levels=None,
        horizon=5,
        prior_accept_prob=0.5,
        responder_utility_estimate=-50
    ):
        self.offer_levels = offer_levels or list(range(0, 101, 10))
        self.horizon = horizon
        self.prior_accept_prob = prior_accept_prob
        self.responder_utility_estimate = responder_utility_estimate
        self.accept_stats = {x: {"accepts": 0, "trials": 0} for x in self.offer_levels}

    def propose(self, game, self_player):
        best_offer = None
        best_expected_utility = float("-inf")

        for offer in self.offer_levels:
            P_accept = self.estimate_accept_prob(offer)
            EU = self.simulate_expected_utility(
                offer=offer,
                P_accept=P_accept,
                game=game,
                self_player=self_player
            )

            if EU > best_expected_utility:
                best_expected_utility = EU
                best_offer = offer

        return best_offer

    def estimate_accept_prob(self, offer):
        stats = self.accept_stats[offer]
        if stats["trials"] == 0:
            return self.prior_accept_prob
        return stats["accepts"] / stats["trials"]

    def simulate_expected_utility(self, offer, P_accept, game, self_player):
        proposer_util = offer
        responder_util = -offer
        reject_penalty = game.get_bad_utility(player=self_player, is_proposer=True)

        base_p = game.p_base
        bump = game.p_reject_bump
        bumped_p = min(1.0, base_p + bump)

        # Utility if accepted: proposer keeps role, offer each round
        EU_accept = sum([(1 - base_p)**t * proposer_util for t in range(self.horizon)])

        # Utility if rejected: penalty now, then risk of switch
        EU_reject = reject_penalty
        role_prob = 1.0

        for t in range(1, self.horizon):
            role_prob *= (1 - bumped_p)
            EU_reject += role_prob * proposer_util + (1 - role_prob) * self.responder_utility_estimate

        return P_accept * EU_accept + (1 - P_accept) * EU_reject

    def receive_feedback(self, accepted, offer, game):
        stats = self.accept_stats[offer]
        stats["trials"] += 1
        if accepted:
            stats["accepts"] += 1
        else:
            # Use rejection to refine estimated cost of losing control
            penalty = game.get_bad_utility(player=game.proposer, is_proposer=True)
            self.responder_utility_estimate = 0.9 * self.responder_utility_estimate + 0.1 * penalty
