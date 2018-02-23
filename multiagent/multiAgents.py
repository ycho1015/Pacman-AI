# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Food = True and False values
        # Capsules = list of a tuple
        # ScaredTimes = list of time in int value
        # GhostState = Ghost: (x,y)=(21.0, 2.0), West
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # (furthest distance between pacman and ghost) / (closest food)

        ghost_score = -1000
        for ghost in newGhostStates:
            ghost_score = max(ghost_score, manhattanDistance(newPos, ghost.getPosition()))
            if ghost.getPosition() == newPos:
                return -1000

        food_score = 1000
        for food in currentFood.asList():
            food_score = min(food_score, manhattanDistance(newPos, food))

        return ghost_score / (food_score + 1.0)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # get the max utilty value of pacman using dispatch implementation
        # if there are multiple utilty values that are the same, then return random action from them
        actions = []
        chosen_index = 0
        for action in gameState.getLegalActions(0):
            v = (self.value(gameState.generateSuccessor(0, action), 1, 0), action)
            actions.append(v)
            actions.sort(key=lambda x: x[0], reverse=True)
            best_score = actions[0][0]
            best_indices = [index for index in range(len(actions)) if actions[index][0] == best_score]
            chosen_index = random.choice(best_indices)

        return actions[chosen_index][1]

    # value function makes the decision of which minimax functions to use
    def value(self, gameState, agent_index, depth):
        if depth == self.depth or len(gameState.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(gameState)
        elif agent_index == 0:
            return self.max_value(gameState, agent_index, depth)
        else:
            return self.min_value(gameState, agent_index, depth)

    # max_value & min_value returns the utility value
    def max_value(self, gameState, agent_index, depth):
        v = float("-inf")
        for action in gameState.getLegalActions(agent_index):
            if agent_index == gameState.getNumAgents() - 1:
                v = max(v, self.value(gameState.generateSuccessor(agent_index, action), 0, depth + 1))
            else:
                v = max(v, self.value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth))
        return v

    def min_value(self, gameState, agent_index, depth):
        v = float("inf")
        for action in gameState.getLegalActions(agent_index):
            if agent_index == gameState.getNumAgents() - 1:
                v = min(v, self.value(gameState.generateSuccessor(agent_index, action), 0, depth + 1))
            else:
                v = min(v, self.value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # get the max utilty value of pacman using dispatch implementation
        # if there are multiple utilty values that are the same, then return random action from them
        actions = []
        chosen_index = 0
        # added alpha and beta constraints from minimax
        alpha, beta = float("-inf"), float("inf")

        for action in gameState.getLegalActions(0):
            v = (self.value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta), action)
            actions.append(v)
            actions.sort(key=lambda x: x[0], reverse=True)
            best_score = actions[0][0]
            best_indices = [index for index in range(len(actions)) if actions[index][0] == best_score]
            chosen_index = random.choice(best_indices)
            alpha = max(alpha, actions[0][0])

        return actions[chosen_index][1]

    # value function makes the decision of which minimax functions to use
    def value(self, gameState, agent_index, depth, alpha, beta):
        if depth == self.depth or len(gameState.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(gameState)
        elif agent_index == 0:
            return self.max_value(gameState, agent_index, depth, alpha, beta)
        else:
            return self.min_value(gameState, agent_index, depth, alpha, beta)

    # max_value & min_value returns the utility value
    def max_value(self, gameState, agent_index, depth, alpha, beta):
        v = float("-inf")
        for action in gameState.getLegalActions(agent_index):
            if agent_index == gameState.getNumAgents() - 1:
                v = max(v, self.value(gameState.generateSuccessor(agent_index, action), 0, depth + 1, alpha, beta))
            else:
                v = max(v, self.value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth, alpha, beta))

            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, agent_index, depth, alpha, beta):
        v = float("inf")
        for action in gameState.getLegalActions(agent_index):
            if agent_index == gameState.getNumAgents() - 1:
                v = min(v, self.value(gameState.generateSuccessor(agent_index, action), 0, depth + 1, alpha, beta))
            else:
                v = min(v, self.value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth, alpha, beta))

            if v < alpha:
                return v
            beta = min(beta, v)

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # get the max utilty value of pacman using dispatch implementation
        # if there are multiple utilty values that are the same, then pick a random action among them and return it
        actions = []
        chosen_index = 0
        for action in gameState.getLegalActions(0):
            v = (self.value(gameState.generateSuccessor(0, action), 1, 0), action)
            actions.append(v)
            actions.sort(key=lambda x: x[0], reverse=True)
            best_score = actions[0][0]
            best_indices = [index for index in range(len(actions)) if actions[index][0] == best_score]
            chosen_index = random.choice(best_indices)

        return actions[chosen_index][1]

    # value function makes the decision of which minimax functions to use
    def value(self, gameState, agent_index, depth):
        if depth == self.depth or len(gameState.getLegalActions(agent_index)) == 0:
            return self.evaluationFunction(gameState)
        elif agent_index == 0:
            return self.max_value(gameState, agent_index, depth)
        else:
            return self.exp_value(gameState, agent_index, depth)

    # max_value & exp_value returns the utility value
    def max_value(self, gameState, agent_index, depth):
        v = float("-inf")
        for action in gameState.getLegalActions(agent_index):
            if agent_index == gameState.getNumAgents() - 1:
                v = max(v, self.value(gameState.generateSuccessor(agent_index, action), 0, depth + 1))
            else:
                v = max(v, self.value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth))
        return v

    # we do not have the actual probability values for each successor state, so compute the average instead
    def exp_value(self, gameState, agent_index, depth):
        v = 0
        count = 0

        for action in gameState.getLegalActions(agent_index):
            count += 1

        p = 1.0 / count
        for action in gameState.getLegalActions(agent_index):
            if agent_index == gameState.getNumAgents() - 1:
                v += p * self.value(gameState.generateSuccessor(agent_index, action), 0, depth + 1)
            else:
                v += p * self.value(gameState.generateSuccessor(agent_index, action), agent_index + 1, depth)
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      possible best score would be...
      1. less number of food gives better score
      2. less number of capsule gives better score
      3. chasing for ghost after taking capsule
      4. going for winning state is gives highest score
      5. going for lose state gives worst score
    """
    "*** YOUR CODE HERE ***"
    pac_pos = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood()  # food available from current state
    current_capsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    ghost_score = 0
    for ghost in ghost_states:
        ghost_score += max(ghost_score, manhattanDistance(pac_pos, ghost.getPosition()))
        if pac_pos == ghost.getPosition():
            ghost_score = float("-inf")
        elif scared_times != 0:
            ghost_score += 100

    remaining_capsule = []
    for capsule in current_capsules:
        remaining_capsule.append(capsule)

    remaining_food = []
    for food in current_food.asList():
        remaining_food.append(food)

    # evaluation values using linear combinations
    food_score = 1.0 / (len(remaining_food) + 1.0)
    capsule_score = 1.0 / (len(remaining_capsule) + 1.0)
    ghost_score = 1.0 / (ghost_score + 1.0)

    return food_score + capsule_score + ghost_score



# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

