# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random
import pandas as pd
from sklearn import tree

class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        assert False # Should not be possible to get here

    def GetWinner(self):
        """ Get the winner based on current board state.
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == 1:
                    return 1
                elif self.board[x] == 2:
                    return 2
        return 0

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for _ in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s

def UCT(rootstate, itermax, model, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for _ in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            # Use classifier 0.9 of the time and randomly at the rest 0.1
            if model and random.random() < 0.9:
                m = model.predict([state.board + [3 - state.playerJustMoved]])[0]
                if m < 0 or m > 8 or state.board[m] != 0:
                    # Fall back to random choice if model's output doesn't make sense
                    m = random.choice(state.GetMoves())
                state.DoMove(m)
            else:
                state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def UCTPlayGame(model1, model2, early_stopping = False):
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState()
    states = []

    while (state.GetMoves() != []):
        if state.playerJustMoved == 1:
            # player 2
            m = UCT(rootstate = state, itermax = 100, model = model2, verbose = False)
        else:
            # player 1
            m = UCT(rootstate = state, itermax = 100, model = model1, verbose = False)

        states.append(state.board + [3-state.playerJustMoved, m])
        state.DoMove(m)

        if early_stopping:
            winner = state.GetWinner()
            # A game can end before filling entire board
            if winner != 0:
                return states, winner

    return states, state.GetWinner()

def UCTVsDTGame(dt, dt_index, itermax):
    """ Play a game between UCT player and Decision Tree classifier.
    """
    state = OXOState()
    winner = 0

    while (state.GetMoves() != []):
        if state.playerJustMoved == dt_index:
            m = UCT(rootstate = state, itermax = itermax, model = None, verbose = False)
        else:
            m = dt.predict([state.board + [3 - state.playerJustMoved]])[0]
            if m < 0 or m > 8 or state.board[m] != 0:
                # Fall back to random choice if model's output doesn't make sense
                m = random.choice(state.GetMoves())

        state.DoMove(m)

        winner = state.GetWinner()
        # A game can end before filling entire board
        if winner != 0:
            return winner

    return winner

if __name__ == "__main__":
    print("Starting")
    # Load the data
    data = pd.read_csv('data_e.csv')
    models = []
    e_vs_e = []
    uct_vs_dt = {100: [], 200: [], 400: [], 800: []}

    for i in range(201):
        print(f"Iteration {i}")
        
        # Prepare inputs and labels
        X = data.drop(['Move'], axis=1)
        y = data['Move']

        # Train new Decision Tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(X, y)

        # Experiments (every 10 iterations)
        if(i%10 == 0 and i>0):
            # Expert (old; random) vs Expert (latest; clf)
            # - Get randomly 10 past models and play 10 games against each one
            # - Save results [wins, loses, draws]
            w = l = d = 0
            opponents = random.sample(range(len(models)), k=10)
            for o in opponents:
                for k in range(10):
                    player1 = player2 = None
                    latest_id = 1
                    # Randomly select who is starting the game
                    if random.random() < 0.5:
                        player1 = clf
                        player2 = models[int(o)]
                    else:
                        player1 = models[int(o)]
                        player2 = clf
                        latest_id = 2
                    _, winner = UCTPlayGame(player1, player2, True)
                    if winner == latest_id:
                        w += 1
                    elif winner == 0:
                        d += 1
                    else:
                        l += 1
            e_vs_e.append([w, l, d])
            # MCTS vs DT
            # - Play 100 games against each MCTS and save results as above
            for iters in uct_vs_dt:
                w = l = d = 0
                for k in range(100):
                    player_index = 2
                    # Randomly select who is starting the game
                    if random.random() < 0.5:
                        player_index = 1
                    winner = UCTVsDTGame(clf, player_index, iters)
                    if winner == player_index:
                        w += 1
                    elif winner == 0:
                        d += 1
                    else:
                        l += 1
                uct_vs_dt[iters].append([w, l, d])

        # Save current classifier to the list of models
        models.append(clf)

        # Generate new data
        states = []
        for g in range(500):
            # Doesn't matter who won in this case
            game, _ = UCTPlayGame(clf, clf)
            # Pick randomly only one state from each game
            states.append(game[random.randrange(len(game))])

        df = pd.DataFrame(states, columns=['Cell0','Cell1','Cell2','Cell3','Cell4','Cell5','Cell6','Cell7','Cell8','Player','Move'])
        # Swap player 1 and 2 to get even more data
        df_copy = df.copy()
        df_copy.loc[:, df_copy.columns != 'Move'] = df_copy.loc[:, df_copy.columns != 'Move'].replace([1,2], [2,1])
        
        # Append new data to the original dataset
        data = data.append(df)
        data = data.append(df_copy)
    
    # Save experiments results
    ex1_df = pd.DataFrame(e_vs_e, columns=['Wins','Losses','Draws'])
    ex1_df.to_csv('results_1.csv', index=False)

    for key in uct_vs_dt:
        ex2_df = pd.DataFrame(uct_vs_dt[key], columns=['Wins','Losses','Draws'])
        ex2_df.to_csv(f'results_2_{key}.csv', index=False)

    # Save all the data generated so far
    data.to_csv('data_full.csv', index=False)