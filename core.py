import random
import pandas as pd
import itertools
import pandas as pd
import matplotlib.pyplot as plt

class Player:
    """
    This class defines a player for our 2 party game. 
    Players must have names and a proposer and responder strategy.
    The strategies available are defined in strategies/proposers.py and strategies/responders.py
    """
    def __init__(self, name, *, proposer_strategy, responder_strategy):
        self.name = name
        self.proposer_strategy = proposer_strategy
        self.responder_strategy = responder_strategy
        self.utility = 0

    def make_offer(self, game):
        return self.proposer_strategy.propose(game, self)

    def respond_to_offer(self, offer, game):
        return self.responder_strategy.respond(offer, game, self)

class Game:
    """
    This is the class for our two party game
    p represents the probability of control of the game flipping
    Each player is assumed to have a specific cost (negative int) of rejection
    The proposer has a further (negative int) cost of rejection while the receiver has a (positive int) cost of rejection
    Optionally we can add a random penalty when rejection occurs (we find this unhelpful so far) 
    We are working on integrating a bump in likelihood of control flipping when rejection occurs to see if this impacts behavior
    """
    def __init__(
        self, *, player1, 
        player2, p, player_1_bad, player_2_bad, 
        proposer_bad, receiver_bad, verbose=True,
        min_offer=0, max_offer=100, plot=False,
        proposer_random_penalty_range=(0, 0),
        receiver_random_penalty_range=(0, 0),
        p_reject_bump=0):
        
        self.player1 = player1
        self.player2 = player2
        self.p = p
        self.p_base = p
        self.p_current = p
        self.p_reject_bump = p_reject_bump
        self.p_reset_on_accept = True
        self.player_1_bad = player_1_bad
        self.player_2_bad = player_2_bad
        self.proposer_bad = proposer_bad
        self.receiver_bad = receiver_bad
        self.proposer = player1
        self.receiver = player2
        self.history = []
        self.verbose=verbose
        self.min_offer = min_offer
        self.max_offer = max_offer
        self.plot = plot
        self.proposer_random_penalty_range=proposer_random_penalty_range
        self.receiver_random_penalty_range=receiver_random_penalty_range

    def play_round(self):
        """
        This function represents one round of the game. Proposer makes an offer,
        responder accepts or rejects, then utilities are handled for both, history is logged, 
        and there is an opportunity for control of the game to switch
        """
        offer = self.proposer.make_offer(self)

        # Some proposer strategies rely on knowing what the other party proposes
        if hasattr(self.receiver.proposer_strategy, "observe_offer_from_opponent"):
            self.receiver.proposer_strategy.observe_offer_from_opponent(offer)

        # Get the response from the receiver
        accepted = self.receiver.respond_to_offer(offer, self)
        
        if accepted:
            # Reset p to baseline (for cases where p moves with rejection)
            if self.p_reset_on_accept:
                self.p_current = self.p_base
            
            self.proposer.utility += offer
            self.receiver.utility += -offer
            if self.verbose:
                print(f"{self.proposer.name} offered {offer:.2f}. {self.receiver.name} accepted.")
        else:
            
            # Bump up p
            self.p_current = min(1.0, self.p_current + self.p_reject_bump)
            
            proposer_penalty = self.get_bad_utility(self.proposer, is_proposer=True)
            receiver_penalty = self.get_bad_utility(self.receiver, is_proposer=False)

            # Generally zero unless optional param included
            proposer_penalty += random.uniform(*self.proposer_random_penalty_range)
            receiver_penalty += random.uniform(*self.receiver_random_penalty_range)

            self.proposer.utility += proposer_penalty
            self.receiver.utility += receiver_penalty

            if self.verbose:
                print(f"{self.proposer.name} offered {offer:.2f}. {self.receiver.name} rejected.")
                print(f"Bad outcome: {self.proposer.name} gets {proposer_penalty}, {self.receiver.name} gets {receiver_penalty}.")

        self.history.append({
            "round": len(self.history) + 1,
            "proposer": self.proposer.name,
            "offer": offer,
            "accepted": accepted,
            "player1_utility": self.player1.utility,
            "player2_utility": self.player2.utility,
            "next_round_switching_prob": self.p_current
        })

        # Some proposer strategies rely on knowing what happened in the last round
        self.proposer.proposer_strategy.receive_feedback(accepted, offer, self)

        # Opportunity for control of the game to switch
        self.maybe_switch_roles()        

    def maybe_switch_roles(self):
        """
        This function controls whether or not control of the game switches between players
        """
        if random.random() < self.p_current:
            self.proposer, self.receiver = self.receiver, self.proposer
            if self.verbose:
                print(f"Roles switched. Now {self.proposer.name} is proposer.")


    def get_bad_utility(self, player, is_proposer):
        if player == self.player1:
            return self.player_1_bad + (self.proposer_bad if is_proposer else self.receiver_bad)
        else:
            return self.player_2_bad + (self.proposer_bad if is_proposer else self.receiver_bad)
    
    def run(self, n_rounds):
        for _ in range(n_rounds):
            self.play_round()
        if self.plot:
            self.plot_utilities()
    
    def plot_utilities(self):
        df = pd.DataFrame(self.history)

        plt.plot(df["round"], df["player1_utility"], label=self.player1.name)
        plt.plot(df["round"], df["player2_utility"], label=self.player2.name)
        plt.xlabel("Round")
        plt.ylabel("Cumulative Utility")
        plt.title("Utility Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def run_tournament(
    proposer_strategies,
    responder_strategies,
    rounds_per_game=100,
    trials_per_match=50,
    p=0.3,
    game_kwargs=None,
):
    """
    This function runs a round robin tournament between the two players using the 
    inputted set of proposer and responder strategies. 
    Optional game_kwargs can be passed in to control optional params in the Game class
    """

    game_kwargs = game_kwargs or {}
    records = []

    strategy_pairs = list(itertools.product(proposer_strategies, responder_strategies))
    matchups = list(itertools.product(strategy_pairs, strategy_pairs))

    for ((p1_prop_name, p1_prop_fn), (p1_resp_name, p1_resp_fn)), \
        ((p2_prop_name, p2_prop_fn), (p2_resp_name, p2_resp_fn)) in matchups:

        for trial in range(trials_per_match):
            player1 = Player("Player 1", proposer_strategy=p1_prop_fn(), responder_strategy=p1_resp_fn())
            player2 = Player("Player 2", proposer_strategy=p2_prop_fn(), responder_strategy=p2_resp_fn())

            game = Game(
                player1=player1,
                player2=player2,
                p=p,
                verbose=False,
                **game_kwargs,
            )

            game.run(rounds_per_game)

            records.append({
                "p1_proposer": p1_prop_name,
                "p1_responder": p1_resp_name,
                "p2_proposer": p2_prop_name,
                "p2_responder": p2_resp_name,
                "trial": trial,
                "p1_utility": player1.utility,
                "p2_utility": player2.utility,
                "rejections": sum(not r["accepted"] for r in game.history),
            })

    return pd.DataFrame(records)
