# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from searchAgents import mazeDistance
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
    def __init__(self):
        #If it was -1 it means we have to assign it later :) otherwise leave it
        self.numOfFoods = -1 

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ans = 0
        minGhostDis = float("inf")
        for ghost in newGhostStates:
            minGhostDis = min(minGhostDis, util.manhattanDistance(newPos, ghost.getPosition()))
        if len(newFood.asList()) == 0 and minGhostDis > 2:
            return float("inf") #Winning is everything we care about!
        if minGhostDis > 2:
            ans += 1000 #It means we have assurance about not getting caught in the next step!
        ans += minGhostDis #We want them far from us. The more they are it's better for us :)
        oldFood = currentGameState.getFood()
        if len(oldFood.asList()) > len(newFood.asList()) and minGhostDis > 2:
            ans += 1000 #Enemy is far and we have freedom to eat in peace!
        minFoodDis = float("inf")
        for foodPos in newFood.asList():
            minFoodDis = min(minFoodDis, util.manhattanDistance(newPos, foodPos))
        """It's better if we can go toward a food. The reason behind multiplying by 2 is it's more important to eat
        than fear from those funky enemies!"""
        ans -= minFoodDis * 2
        if minGhostDis < 4:
            ans -= 1000 #Lets stay out of danger!
        return ans

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

    def maxVal(self, state, exploredDepth, isFirstLayer = False):
        if exploredDepth > self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        v = -float("inf")
        legalActions = state.getLegalActions(0)
        bestAction = legalActions[0]
        for act in legalActions:
            if self.minVal(state.generateSuccessor(0, act), 1, exploredDepth) > v:
                bestAction = act
                v = self.minVal(state.generateSuccessor(0, act), 1, exploredDepth)
        if isFirstLayer:
            return bestAction
        return v

    def minVal(self, state, ind, exploredDepth):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float("inf")
        legalActions = state.getLegalActions(ind)
        for act in legalActions:
            if ind == state.getNumAgents() - 1: #It's the last enemy we have to consider
                v = min(self.maxVal(state.generateSuccessor(ind, act), exploredDepth + 1), v)
            else: #There's more enemies to consider this turn
                v = min(self.minVal(state.generateSuccessor(ind, act), ind + 1, exploredDepth), v)
        return v

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.maxVal(gameState, 1, True)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxVal(self, state, exploredDepth, alpha, beta, isFirstLayer = False):
        if exploredDepth > self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        v = -float("inf")
        legalActions = state.getLegalActions(0)
        bestAction = legalActions[0]
        for act in legalActions:
            if self.minVal(state.generateSuccessor(0, act), 1, exploredDepth, alpha, beta) > v:
                bestAction = act
                v = self.minVal(state.generateSuccessor(0, act), 1, exploredDepth, alpha, beta)
            if v > beta:
                break;
            alpha = max(alpha, v)
        if isFirstLayer:
            return bestAction
        return v

    def minVal(self, state, ind, exploredDepth, alpha, beta):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float("inf")
        legalActions = state.getLegalActions(ind)
        for act in legalActions:
            if ind == state.getNumAgents() - 1: #It's the last enemy we have to consider
                v = min(self.maxVal(state.generateSuccessor(ind, act), exploredDepth + 1, alpha, beta), v)
            else: #There's more enemies to consider this turn
                v = min(self.minVal(state.generateSuccessor(ind, act), ind + 1, exploredDepth, alpha, beta), v)
            if v < alpha:
                break
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxVal(gameState, 1, -float("inf"), float("inf"), True)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxVal(self, state, exploredDepth, isFirstLayer = False):
        if exploredDepth > self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        v = -float("inf")
        legalActions = state.getLegalActions(0)
        bestAction = legalActions[0]
        for act in legalActions:
            if self.minVal(state.generateSuccessor(0, act), 1, exploredDepth) > v:
                bestAction = act
                v = self.minVal(state.generateSuccessor(0, act), 1, exploredDepth)
        if isFirstLayer:
            return bestAction
        return v

    def minVal(self, state, ind, exploredDepth):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = 0
        legalActions = state.getLegalActions(ind)
        for act in legalActions:
            if ind == state.getNumAgents() - 1: #It's the last enemy we have to consider
                v += self.maxVal(state.generateSuccessor(ind, act), exploredDepth + 1) * (1 / len(legalActions))
            else: #There's more enemies to consider this turn
                v += self.minVal(state.generateSuccessor(ind, act), ind + 1, exploredDepth) * (1 / len(legalActions))
        return v

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.maxVal(gameState, 1, True)
        util.raiseNotDefined()

def posToInt(xy):
    return (int(xy[0]), int(xy[1]))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I used multiple parameters to evaluate a state. first of all I used score of 
    the game multiplied by 100 as a good parameter for agent. Then I used minimum food distance
    as a bad parameter for agent. It means we prefer less food distance. When number of foods
    are more than 10 I used manhattan distance but when it comes to less than 10 I used maze 
    distance. the reason behind that is when I run my old version of this file which I used 
    manhattan distance everywhere, I realized that wall are serious problem for our evaluation 
    so I switched to maze distance from last project but here was another problem and that was 
    very bad performance it was causing so I decided to use manhattan when I have many foods to
    process and use maze distance when number of foods are not that much. After that I handeled
    winning and losing states so if I can win using a state then I jump to it and if it was a 
    losing state then prevent ourselves to get there. I also used food numbers multiply 300 as a
    very bad parameter because every food we eat leaves board and reducing number of foods is a
    priority to us. Last part of this function is related to hunting those funky ghosts :)
    If there is a scary ghost near us then it's time to get to his location to hunting him and 
    if he is not scared then maybe it's better to be 3 to 6 blocks away from him (not less not more)
    """

    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghosts]
    ans = 100 * currentGameState.getScore()

    if len(foods) > 0:
        minFoodDis = float("inf")
    else:
        minFoodDis = -99999999999999999999999999999999
    if len(foods) > 10:
        for food in foods:
            minFoodDis = min(minFoodDis, manhattanDistance(food, pos))
    else:
        for food in foods:
            minFoodDis = min(minFoodDis, mazeDistance(food, pos, currentGameState))        
    ans -= minFoodDis * 3

    if currentGameState.isWin():
        betterEvaluationFunction.maximum += 1
        #print("I could win")
        return betterEvaluationFunction.maximum
    if currentGameState.isLose():
        betterEvaluationFunction.minimum -= 1
        #print("I could Lose")
        return betterEvaluationFunction.minimum
    ans -= (len(foods)) * 300

    for i in range(len(ghosts)):
        if scaredTimes[i] > 0 and mazeDistance(posToInt(ghosts[i].getPosition()), pos, currentGameState) < 3: 
            #print(util.manhattanDistance(ghosts[i].getPosition(), pos), scaredTimes[i])
            ans += scaredTimes[i] * 10 #Very close! so it could be worthy to eat
        elif scaredTimes[i] > 0: #Then it's good if they be near
            ans += scaredTimes[i]
        elif ghosts[i].getPosition() == ghosts[i].start.getPosition():
            ans += 0 #Congrats! we eat that mad ghost :)
        elif mazeDistance(posToInt(ghosts[i].getPosition()), pos, currentGameState) > 3 and mazeDistance(posToInt(ghosts[i].getPosition()), pos, currentGameState) < 6: 
            ans += mazeDistance(posToInt(ghosts[i].getPosition()), pos, currentGameState) / 10
        elif mazeDistance(posToInt(ghosts[i].getPosition()), pos, currentGameState) > 6:
            ans += 0
        else: #Stay at least 3 blocks away of the ghost
            ans -= 100 * (4 - mazeDistance(posToInt(ghosts[i].getPosition()), pos, currentGameState))
    betterEvaluationFunction.maximum = max(ans, betterEvaluationFunction.maximum)
    betterEvaluationFunction.minimum = min(ans, betterEvaluationFunction.minimum)
    return ans

    util.raiseNotDefined()
betterEvaluationFunction.maximum = 0
betterEvaluationFunction.minimum = 0

# Abbreviation
better = betterEvaluationFunction
