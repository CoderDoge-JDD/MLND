"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # get current move count
    move_count = game.move_count

    # count number of moves available
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # calculate weight
    w = 10 / (move_count + 1)

    # return weighted delta of available moves
    return float(own_moves - (w * opp_moves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # AVAILABLE MOVES HEURISTICS
    # The second heuristic is a simple combination of agent moves and opponent moves.
    # Basically, the number of opponent moves is more considerable than agent moves,
    # cause when the weight of opp_score is less than agent's, greedy in improving 
    # agent's moves will lead to possible failure. On the contrary, if the weight of 
    # opp_score is much more than agent's, the system will not be so much effective 
    # as well.

    # Basic decision of WIN and LOSE
    if game.is_loser(player):
    	return float("-inf")
    if game.is_winner(player):
    	return float("inf")

    # Define scores as agent and opponent legal moves
    agent_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    positive_score = agent_moves
    negative_score = opponent_moves

    # # Adjust weights of two scores and combine them
    rf = 3
    combined_score = positive_score - rf * negative_score

    return float(combined_score)


# There are some helper function for custom_score_3
def edge_alert(move, edges):
    """Specify whether a move is approaching edges or not, for
    further score addition
    """
    for edge in edges:
        if move in edge:
            return True

    return False


def board_filling(game):
	"""Calculate the filling rate of game board, for analysing 
	whether a specific move is good or not
	"""
	# Number of blank spaces
	blank_spaces = len(game.get_blank_spaces())

	# Number of all spaces
	all_spaces = game.width * game.height

	return int(blank_spaces / all_spaces * 100)

def proximity(location1, location2):
    return abs(location1[0]-location2[0])+abs(location1[1]-location2[1])

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # COMPREHENSIVE SCORE HEURISTIC
    # The third heuristic is designed with 
    # Basic decision of WIN and LOSE
    opponent = game.get_opponent(player)
    player_location   = game.get_player_location(player  )
    opponent_location = game.get_player_location(opponent) 
    playerMoves   = game.get_legal_moves(player  )
    opponentMoves = game.get_legal_moves(opponent)
    blank_spaces = game.get_blank_spaces()

    board_size = game.width * game.height

    # size of local area
    localArea = (game.width + game.height)/4

    # condition that corresponds to later stages of the game
    if board_size - len(blank_spaces) > float(0.3 * board_size):
        # filtering out moves that are within local are
        playerMoves = [move for move in playerMoves if proximity(player_location, move)<=localArea]
        opponentMoves = [move for move in opponentMoves if proximity(opponent_location, move)<=localArea]

    return float(len(playerMoves) - len(opponentMoves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass  
        return best_move


    def minimax_recursive(self, game, depth, max_player=True):
        """Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        depth : int - number which represents to what depth should we search our game tree
        max_player: boolean - to represent which player is currently on the move (considering our game tree)
        Returns
        -------
        best_value - float value which represent the best_value which is on the node of the best_move to be performed
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get current legal moves
        legal_moves = game.get_legal_moves()

        # Check terminal state: reach the deepest
        if depth == 0:
            return self.score(game, self), (-1, -1)

        
        # Check terminal state: reach the final value
        if len(legal_moves) == 0:
            return game.utility(self), (-1, -1)

        # Setting default values
        best_move = (-1, -1)
        best_value = 0

        # Starting case
        if max_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        # Essence of minmax function
        # Iterate through every move of legal moves
        for move in legal_moves:
            # Forecast potential next moves
            next_state = game.forecast_move(move)
            # Recursively call te function to compute value
            c_val, _ = self.minimax_recursive(next_state, depth-1, not max_player)

            # Maximizing process
            if max_player:
                # temp value computation
                temp_value = max(best_value, c_val)
                # Comparison with the current best value
                if temp_value != best_value:
                    # Setting new values for our best_value and new value for best_move
                    best_value, best_move = c_val, move

            # Minimizing process  
            else:
                temp_value = min(best_value, c_val)
                if temp_value != best_value:
                    best_value, best_move = c_val, move

        return best_value, best_move

    def minimax(self, game, depth, max_player=True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Call the recursive function
        best_value, best_move = self.minimax_recursive(game, depth, max_player)
        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        
        self.time_left = time_left
        best_move = None
        if len(game.get_legal_moves()) == 0:
            best_move = (-1, -1)
            return best_move
        try:
            depth = 0
            while True:
                move = self.alphabeta(game, depth)
                best_move = move
                depth += 1
            print(depth) 
        except SearchTimeout:
            pass
        return best_move



    def alphabeta_recursive(self, game, depth, alpha, beta, max_player):
        """Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        depth : int - number which represents to what depth should we search our game tree
        alpha - best already explored option along path to the root for maximizer
        beta - best already explored option along path to the root for minimizer
        max_player: boolean - to represent which player is currently on the move (considering our game tree)
        Returns
        -------
        best_value - float value which represent the best_value which is on the node of the best_move to be performed
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get current legal moves
        legal_moves = game.get_legal_moves()

        # Check terminal state: reach the deepest
        if depth == 0:
            return self.score(game, self), (-1, -1)

        
        # Check terminal state: reach the final value
        if len(legal_moves) == 0:
            return game.utility(self), (-1, -1)

        # Setting default values
        best_move = (-1, -1)
        best_value = 0

        # Starting case
        if max_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        # Essence of alphabeta function
        # Iterate through every move of legal moves
        for move in legal_moves:
            # Forecast potential next moves
            next_state = game.forecast_move(move)
            # Recursively call te function to compute value
            c_val, _ = self.alphabeta_recursive(next_state, depth-1, alpha, beta, not max_player)

            # Maximizing process
            if max_player:
                # temp value computation
                temp_value = max(best_value, c_val)
                # Comparison with the current best value
                if temp_value != best_value:
                    # Setting new values for our best_value and new value for best_move
                    best_value, best_move = c_val, move
                # Beta condition
                if beta <= best_value:
                    break
                # Update alpha
                alpha = max(alpha, best_value)

            # Minimizing process  
            else:
                temp_value = min(best_value, c_val)
                if temp_value != best_value:
                    best_value, best_move = c_val, move
                # alpha condition
                if alpha >= best_value:
                    break
                # Update beta
                beta = min(beta, best_value)

        return best_value, best_move




    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Call the recursive function
        best_value, best_move = self.alphabeta_recursive(game, depth, alpha, beta, True)
        
        return best_move
