import random
import math
from .base import ResponderStrategy

class UtilitarianResponder(ResponderStrategy):
    """
    This naive responder class merely selects the best option for the current 
    round in a utility maximizing approach
    """
    def respond(self, offer, game, self_player):
        # Determine if player is player1 or player2
        if self_player == game.player1:
            bad_utility = game.player_1_bad + game.receiver_bad
        else:
            bad_utility = game.player_2_bad + game.receiver_bad

        return -offer > bad_utility

class TitForTatResponder(ResponderStrategy):
    """
    This responder accepts all offers that are better than rejection 
    and not worse than the previous offer made. It outperforms utilitarian responder
    against BinarySearch proposer, as it sees the best offer it has seen as a floor
    """
    def __init__(self):
        self.last_offer = None

    def respond(self, offer, game, self_player):
        current_utility = -offer

        # Compute rejection penalty
        if self_player == game.player1:
            bad_utility = game.player_1_bad + game.receiver_bad
        else:
            bad_utility = game.player_2_bad + game.receiver_bad

        # First round: accept only if better than rejecting
        if self.last_offer is None:
            return current_utility > bad_utility

        # Otherwise: enforce both consistency and improvement
        return current_utility >= self.last_offer and current_utility > bad_utility

    def receive_feedback(self, offer, accepted):
        self.last_offer = -offer

class ProbabilisticResponder(ResponderStrategy):
    """
    This responder takes a probabilistic approach to accepting offers. 
    The input alpha impacts how sensitive it is to utility difference between 
    accept and reject, values closer to zero lead to higher variance in behavior
    It uses a logistic sigmoid to calculate the likelihood of accepting the offer.
    This class outperforms naive classes against BinarySearch proposer
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def respond(self, offer, game, self_player):
        utility_accept = -offer

        if self_player == game.player1:
            utility_reject = game.player_1_bad + game.receiver_bad
        else:
            utility_reject = game.player_2_bad + game.receiver_bad

        utility_diff = utility_accept - utility_reject
        prob_accept = 1 / (1 + math.exp(-self.alpha * utility_diff))

        return random.random() < prob_accept

class StrategicRejectorResponder:
    """
    This responder strategically rejects offers to motivate the proposer to make 
    better deals. This implementation is pretty dumb: if you make the same offer 
    it accepts the first 4 times but the fifth time it will reject. Against certain 
    proposers, if given a min_offer floor it eventually reaches equilibrium of offers of min_offer, 
    which are always accepted.
    """
    def __init__(self, stagnation_tolerance=4, epsilon=1e-2, min_offer=0):
        self.stagnation_tolerance = stagnation_tolerance
        self.epsilon = epsilon
        self.prev_offer = None
        self.streak = 0
        self.min_offer = min_offer

    def respond(self, offer, game, self_player=None):
        responder_utility = -offer
        reject_utility = game.get_bad_utility(player=self_player, is_proposer=False)

        # Step 1: protect against bad offers
        if responder_utility < reject_utility:
            return False

        # Step 2: if offer is already at minimum, don't punish stagnation
        if offer <= self.min_offer + self.epsilon:
            return True

        # Step 3: apply stagnation logic
        if self.prev_offer is None:
            self.prev_offer = offer
            return True

        if offer >= self.prev_offer - self.epsilon:
            self.streak += 1
        else:
            self.streak = 0

        self.prev_offer = offer
        return self.streak < self.stagnation_tolerance
