from core import Game
class ProposerStrategy:
    """
    Parent class for Proposer Strategies. 
    """
    def propose(self, game, self_player):
        raise NotImplementedError

    def receive_feedback(self, accepted: bool, offer: float, game: Game):
        pass


class ResponderStrategy:
    """
    Parent class for Responder Strategies. 
    """
    def respond(self, offer: float, game, self_player):
        raise NotImplementedError

    def receive_feedback(self, offer: float, accepted: bool, game: Game):
        pass


