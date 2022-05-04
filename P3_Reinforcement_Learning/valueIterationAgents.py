# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            updatedValues = util.Counter()
            for state in self.mdp.getStates():
                maxQ = -float("inf")
                bestAction = "do nothing"
                if len(self.mdp.getPossibleActions(state)) == 0:
                    updatedValues[state] = self.values[state]
                    continue
                for action in self.mdp.getPossibleActions(state):
                    Qa = self.computeQValueFromValues(state, action)
                    if Qa > maxQ:
                        bestAction = action
                        maxQ = Qa
                updatedValues[state] = maxQ
            self.values = updatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        ans = 0
        for dest in self.mdp.getTransitionStatesAndProbs(state, action):
            destState, destT = dest
            ans += destT * (self.mdp.getReward(state, action, destState) + self.discount * self.values[destState])
        return ans
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) == 0:
            return () # try to avoid exception in terminal state
        maxQ = -float("inf")
        bestAction = "do nothing" # :))
        for action in possibleActions:
            if self.computeQValueFromValues(state, action) > maxQ:
                maxQ = self.computeQValueFromValues(state, action)
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        updatedValues = self.values
        for i in range(self.iterations):
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            maxQ = -float("inf")
            bestAction = "do nothing"
            if len(self.mdp.getPossibleActions(state)) == 0:
                updatedValues[state] = self.values[state]
                continue
            for action in self.mdp.getPossibleActions(state):
                Qa = self.computeQValueFromValues(state, action)
                if Qa > maxQ:
                    bestAction = action
                    maxQ = Qa
            updatedValues[state] = maxQ
        self.values = updatedValues

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = dict()
        for state in self.mdp.getStates(): # initialize predecessors dictionary
            predecessors[state] = set()
        for state in self.mdp.getStates(): # add predecessors
            for action in self.mdp.getPossibleActions(state):
                for dest in self.mdp.getTransitionStatesAndProbs(state, action):
                    destState, _ = dest
                    predecessors[destState].add(state)
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            maxQ = -float("inf")
            for action in self.mdp.getPossibleActions(state):
                maxQ = max(self.computeQValueFromValues(state, action), maxQ)
            if maxQ == -float("inf"):
                maxQ = 0
            diff = abs(maxQ - self.values[state])
            pq.update(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                # Update value of s
                maxQ = -float("inf")
                for action in self.mdp.getPossibleActions(s):
                    maxQ = max(self.computeQValueFromValues(s, action), maxQ)
                if maxQ == -float("inf"):
                    maxQ = 0
                self.values[s] = maxQ
            for p in predecessors[s]:
                maxQ = -float("inf")
                for action in self.mdp.getPossibleActions(p):
                    maxQ = max(self.computeQValueFromValues(p, action), maxQ)
                if maxQ == -float("inf"):
                    maxQ = 0 
                diff = abs(maxQ - self.values[p])               
                if diff > self.theta:
                    pq.update(p, -diff)
