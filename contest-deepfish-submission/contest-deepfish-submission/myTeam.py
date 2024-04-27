# myTeam.py
# ---------
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

from capture import GameState
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from game import Actions

import math
# discount reward constant
GAMMA = 0.9
EXPLORATION_CONSTANT = 1.0 / math.sqrt(2.0)
MAX_AGENT_MOVES = 300
MOVE_MAX_TIME = 1.0  # in seconds
from collections import defaultdict
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

    ####################
    # Helper Functions #
    ####################


class Node:
    def __init__(self, location=None, parent=None, action=None, cost=0):
        self.location = location
        self.parent = parent
        self.action = action
        self.cost = cost

    def getLocation(self):
        return self.location

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def getCost(self):
        return self.cost


class StateNode:
    """
    Node class adapted and modified from 'AI a modern approach' for use in these problems
    http://aima.cs.berkeley.edu/
    https://github.com/aimacode/aima-python
    Also adapted from assignment 1
    To be utilised in MCTS
    """

    def __init__(self, agent, gameState, agentPos, actionTo=None, parent=None, visits=0):
        self.agent = agent
        self.gameState = gameState
        self.agentPos = agentPos
        self.parent = parent
        self.visits = visits
        self.depth = 0
        self.actionTo = actionTo
        if actionTo == None:
            self.actionTo = self.gameState.getAgentState(self.agent.index).getDirection()
        self.legalActions = self.getAvailableActions()
        self.untriedActions = self.legalActions.copy()
        if parent:
            self.depth = parent.depth + 1
        self.reward = self.rewardFunction()
        self.q_value = self.qFunction()

    def __repr__(self):
        """
        print current state for this node position, action
        """
        return "<Node: Current State Position: {}, " \
               " Action To Here: {}>".format(self.agentPos, self.actionTo)

    def getAvailableActions(self):
        """
        Returns available actions from this state
        """
        # direction is action to state or orginal direction

        # available actions from above customised helper function
        legalActions = self.getPossibleActions(self.actionTo, self.agentPos)

        # removing stop
        legalActions.remove(Directions.STOP)

        return legalActions

    def getUntriedAction(self):
        """
        Returns unused action from legal actions in comparison with chosen action from
        """
        # chose random action out of untried actions

        untried_action = random.choice(self.untriedActions)

        remainingActions = self.untriedActions.copy()

        remainingActions.remove(untried_action)

        self.untriedActions = remainingActions.copy()

        return untried_action

    def getSuccesssorStateNode(self, action):
        """
        Returns the legal neighbour state for the action specified
        """

        neighbourPosition = Actions.getSuccessor(self.agentPos, action)

        successorStateNode = StateNode(self.agent, GameState(self.gameState), neighbourPosition, actionTo=action,
                                       parent=self)
        return successorStateNode

    def getPossibleActions(self, agentDirection, position):
        possible = []
        x, y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE):
            return [agentDirection]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not self.gameState.getWalls()[next_x][next_y]: possible.append(dir)

        return possible

    def fullyExpanded(self):
        """
        True if all actions used from this state
        """

        return not self.untriedActions

    def terminalStateNode(self, treeDepth):
        """
        Need some definition of terminal state
        All but two food eaten and team members returned to home
        Or max moves made per agent
        Or agent gets eaten?
        """

        # firstly if max moves made by either agents, then this is a terminal state
        currentMoves = self.agent.moves + treeDepth
        if currentMoves == MAX_AGENT_MOVES:
            return True


        # if game over
        if self.gameState.isOver():
            return True

        # thirdly agent gets eaten?
        # get opponents' positions
        # get opponent indices
        opponent_positions = []
        opponent_indices = []
        if self.agent.isRed:
            opponent_indices = self.gameState.getBlueTeamIndices()
        else:
            opponent_indices = self.gameState.getRedTeamIndices()

        for index in opponent_indices:
            opponent_positions.append(self.gameState.getAgentPosition(index))

        for pos in opponent_positions:
            if pos != None:
                dist_to_opponent = self.agent.getMazeDistance(self.agentPos, pos)
                if dist_to_opponent <= 1:
                    # Gonna Die
                    return True

        return False

    def rewardFunction(self):
        """
        More appropriate to put reward calc here as per lect notes etc.
        """

        R_total = 0.0

        R_food = foodReward(self.agent, self.agentPos, self.gameState)

        R_survival = survivalReward(self.agent, self.agentPos, self.gameState,
                                    self.gameState.getAgentState(self.agent.index).isPacman)

        # # total reward
        R_total = R_food + R_survival

        self.reward = R_total

        return self.reward

    def rewardFunctionFlexible(self, foodreward, survivalreward, distancereward):
        """
        More appropriate to put reward calc here as per lect notes etc.
        """
        R_food = foodReward(self.agent, self.agentPos, self.gameState)

        R_survival = survivalReward(self.agent, self.agentPos, self.gameState,
                                    self.gameState.getAgentState(self.agent.index).isPacman)

        # # total reward
        R_total = R_food + R_survival

        self.reward = R_total

        return self.reward

    def qFunction(self):
        """
        Q estimate
        """
        # updates q value
        if self.visits != 0:
            return self.reward / float(self.visits)
        else:
            return 0.0

    def __eq__(self, other):
        """
        Check duplicate vanilla states
        """
        return isinstance(other, StateNode) and self.agentPos == other.agentPos and self.actionTo == other.actionTo

    def __hash__(self):
        """
        hash value of the state for visited set membership comparison
        based on hash value of state
        """
        return hash(self.agentPos + tuple(self.actionTo))


##############################################################
# MCTS Functions (class and functions) begin here ---------- #
##############################################################


class MCTS:
    """
    Class for running MCTS
    """

    def __init__(self, exploration_weight, time_limit, max_agent_moves, agent, start, depthLimit=70,
                 iterations=50):  # Depth limit increased from 50 to 70, avoids indecisive moving
        self.agent = agent
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit  # to account for two MCTS searches, one per agent
        self.max_agent_moves = max_agent_moves
        self.depthLimit = depthLimit
        self.treeDepth = 0
        self.iterations = iterations
        self.foodListOrginal = None
        self.foodListSimulation = None
        self.start = start

    def MCTS_search(self, gameState, start_time):
        """
        entry point for search
        """
        self.gameState = gameState
        self.tree = defaultdict(list)

        agentPos = gameState.getAgentPosition(self.agent.index)

        # start_state = State(self.agent, gameState, agentPos)
        root_node = StateNode(self.agent, self.gameState, agentPos)

        # add root node to tree, with no children, as yet
        self.tree[root_node] = []

        # Creating foodlist
        self.foodListOriginal = root_node.agent.getFood(root_node.gameState).asList()

        # while timelimit not exceeded, nor max agent moves met, should only do selection for depth say 5
        time_margin = self.time_limit * 0.7
        for i in range(0, self.iterations):
            elapsed_time = time.time() - start_time

            if elapsed_time >= (self.time_limit - time_margin):
                break
            leaf_node = self.select(root_node)
            node, reward = self.simulate(leaf_node)
            self.backpropagate(node, reward)

        child = self.bestUCTChild(root_node)
        if child != root_node:
            return child.actionTo
        else:
            return Directions.STOP

    def select(self, node):

        """
        Find unexplored decendent of node
        """

        while not node.terminalStateNode(node.depth):
            if not node.fullyExpanded():
                return self.expand(node)
            else:
                node = self.bestUCTChild(node)
        return node

    def expand(self, node):
        """
        expands states not already expanded
        """

        # get untried action
        untried_action = node.getUntriedAction()

        # state for that action - deterministic
        successorNode = node.getSuccesssorStateNode(untried_action)

        # add this node to children of node
        self.tree[node].append(successorNode)

        # if node not in tree add it and children
        if successorNode not in self.tree:
            self.tree[successorNode] = []

        return successorNode

    def bestUCTChild(self, node):
        """
        Using UCT
        """

        # all children of node should already be expanded

        def uct(child):
            """
            UCT for child of node
            """
            child.q_value = child.reward / float(child.visits)
            exploitation = child.q_value  # Should be able to move the above line...
            exploration = 2.0 * self.exploration_weight * math.sqrt(
                (2.0 * math.log1p(node.visits) / float(child.visits)))

            return exploitation + exploration

        if self.tree.get(node):
            return max(self.tree.get(node), key=uct)
        else:
            return node

    def simulate(self, node):
        """
        for the result of the simulation
        """
        self.foodListSimulation = self.foodListOriginal.copy()
        # print("Length of food: " + str(len(self.foodListSimulation))) # Always 20, resetting as required
        while node.depth <= self.depthLimit:

            if node.terminalStateNode(node.depth):
                reward = node.reward
                return node, reward
            else:
                # do simulation policy
                isPacmanPolicy = node.gameState.getAgentState(node.agent.index).isPacman
                node = self.simulationPolicy(node, isPacmanPolicy)

        reward = node.reward
        # print("reward at end of simulation " + str(reward) + " depth " + str(node.depth))
        return node, reward

    def simulationPolicy(self, node, isPacmanPolicy=False):
        """
        Some kind of domain knowledge based, heuristic
        to guide simulation other than random
        This could be converted to strategy Pattern - which is essentially what it is - more formally
        """

        # # if agent is ghost, apply ghost policy
        if isPacmanPolicy:
            # print("agent is ghost")
            newNode = self.pacmanPolicy(node)
            return newNode
        else:
            newNode = self.ghostPolicy(node)
            return newNode

    def backpropagate(self, node, reward):
        """
        The backup part
        """
        # use the return path from leaf and sum rewards from bottom to top

        while node != None:
            node.visits += 1

            node.reward += reward
            reward = node.reward * 0.90  # Without discount movement may oscillate between nodes

            node = node.parent

    def ghostPolicy(self, node):
        """
        Returns move (next state) based on cases when agent is ghost
        """

        return self.defaultPolicy(node)

    def pacmanPolicy(self, node):
        """
        Set of policies for when agent is pacman
        """

        return self.defaultPolicy(node)

    def defaultPolicy(self, node):
        """
        Simply aim to eat as much food as possible (thus maximising points),
        thus return a node closer to the nearest food
        """

        # go to closest food

        # agent can be ghost or pacman
        # will return one node closer to (nearest remaining) food

        # foodList = node.agent.getFood(self.gameState).asList()
        agent_position = node.agentPos
        agent_direction = node.actionTo
        # for index in range(len(self.foodListSimulation)):
        #     if agent_position == self.foodListSimulation[index]:
        #         del (self.foodListSimulation[index])
        #         break  # Only 1 food per position, no need to continue loop

        sortedFood = sorted(self.foodListSimulation, key=lambda f: node.agent.getMazeDistance(agent_position, f))
        if sortedFood:
            goal = sortedFood[0]
        else:
            goal = self.start

        # get closer position to food
        closerPosition, action = getCloserPosition(node, agent_direction, agent_position, goal,
                                                   node.agent.getMazeDistance)

        closerNode = StateNode(self.agent, GameState(self.gameState), closerPosition, actionTo=action, parent=node)
        return closerNode


####################
# Reward Functions #
####################

"""
The following functions serve to provide the necessary rewards for Monte Carlo Tree Search simulations
The total reward is provided as a vector of two quantities - one per agent
These are further broken down into vectors, with values for each of the tactics for the agent to consider:
tactic = {opponents, capsule, food, survival}
"""
# some constants for tuning rewards
# food rewards


ALPHA = 100.0  # reward for consuming food, sometimes catches

# agent survival reward
ZETA = 2.0  # distance buffer for survival
ETA = -1.0


def foodReward(agent, position, gameState):
    """
    Reward sum of inverse of distance to nearest food
    from simulated position simuPos and inverse of distance between next food
    in order to reward tree traversal towards consecutive food pellets
    """

    R_food = 0.0

    # Rewards should be more discrete based on consuming food
    food = agent.getFood(gameState)

    # nearest pos
    pos = nearestPoint(position)

    x, y = pos[0], pos[1]
    if food[x][y]:
        R_food += ALPHA
    return R_food


def survivalReward(agent, agentPos, gameState, isPacman=False):
    """
    How to recalc distances with new position?
    Should be able to triangulate with two agents, one opponent to a position
    Only apply if one or more is within SIGHT_RANGE
    using observation
    Reward for staying away from ghost when pacman
    Much higher reward for staying away from ghost when food has been eaten
    """

    # Will now just make this a very negative reward if agent gets eaten
    # or within ZETA distance from ghost

    R_survival = 0.0
    opponent_distances = getDistancesToOpponents(agent, agentPos, gameState)

    # if edible - give -ve reward
    # if pacman or if ghost and timer has started
    if isPacman:
        for dist in opponent_distances:
            if dist <= ZETA:
                R_survival += ETA
    else:
        if gameState.getAgentState(agent.index).scaredTimer > 0:
            for dist in opponent_distances:
                if dist <= ZETA:
                    R_survival += ETA

    return R_survival


def getCloserPosition(agent_state, agent_direction, agent_position, destinationPos, heuristic=None):
    """
    Just returns one position one step closer on heuristic to destinationPos
    using MazeDistance will be shortest path
    """

    current_distance = heuristic(agent_position, destinationPos)
    best_distance = current_distance
    closer_position = agent_position
    closer_action = agent_direction
    # get successors
    availableActions = agent_state.getAvailableActions()

    for action in availableActions:
        newPosition = Actions.getSuccessor(agent_position, action)
        new_distance = heuristic(newPosition, destinationPos)
        if new_distance < best_distance:
            best_distance = new_distance
            closer_position = newPosition
            closer_action = action

    return closer_position, closer_action


def getDistancesToOpponents(agent, agentPos, gameState):
    """
    Helper function to return a list of accurate distances to two opponents if available
    """

    # get opponents' positions
    # get opponent indices
    opponent_positions = []
    opponent_indices = []
    opponent_distances = []

    if agent.isRed:
        opponent_indices = gameState.getBlueTeamIndices()
    else:
        opponent_indices = gameState.getRedTeamIndices()

    for index in opponent_indices:
        opponent_positions.append(gameState.getAgentPosition(index))

    for pos in opponent_positions:
        if pos != None:
            opponent_distances.append(agent.getMazeDistance(agentPos, pos))
    return opponent_distances
##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for search agents that chooses score-maximizing actions.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        if gameState.getAgentState(self.index).getPosition()[0] == 1:
            self.isRed = True
        else:
            self.isRed = False
        self.eatableFood = []
        self.mid_line = self.midwayLine(gameState)
        self.depth = 5
        self.atTop = False
        self.maxFood = len(self.getFood(gameState).asList())
        self.enemyDeadEnd = self.badpaths(self.isRed, gameState)
        self.steps = 300
        self.foodInDeadEnd = self.foodFinder(self.enemyDeadEnd, self.getFood(gameState).asList())
        self.crossingPoint = [random.choice(self.mid_line)]
        self.holdingFood = 0
        self.legalPositions = [p for p in gameState.getWalls().asList(False)]
        self.beliefs = {}
        for opponent in self.getOpponents(gameState):
            self.initializePrior(opponent)
        # For MCTS
        self.gameState = gameState
        self.initialFoodToEat = self.getFood(self.gameState).asList()
        self.initialCapsulesToDefend = self.getCapsulesYouAreDefending(gameState)
        self.timeLimit = MOVE_MAX_TIME
        self.mcts_search = MCTS(EXPLORATION_CONSTANT, self.timeLimit, MAX_AGENT_MOVES, self, gameState.getAgentPosition(self.index))
        self.home = gameState.getInitialAgentPosition(self.index)
        self.startDirection = gameState.getAgentState(self.index).getDirection()
        self.moves = 0

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def midwayLine(self, gameState):
        """
        Returns two points that act as a boundary line along which the agent travels
        """
        area = []
        max_y = gameState.data.layout.height
        min_y = 1
        if not self.isRed:
            mid = ((gameState.data.layout.width) // 2)
        else:
            mid = ((gameState.data.layout.width) // 2) -1
        for space in range(min_y, max_y):
            if not gameState.hasWall(int(mid), space):
                area.append((int(mid), space))
        return area

    def badpaths(self, isRed, gameState):
        """
        Returns the positions that are in a deadend
        """
        if not isRed:
            return self.blueDeadEnd(gameState)
        else:
            return self.redDeadEnd(gameState)

    def blueDeadEnd(self, gameState):
        """
        Returns the positions that are in a deadend on the red side
        """
        max_y = gameState.data.layout.height
        min_y = 1
        area = {}
        for x in range(1, (gameState.data.layout.width // 2)):
            for y in range(min_y, max_y):
                if not gameState.hasWall(x, y):
                    area[(x, y)] = []
                    if not gameState.hasWall(x + 1, y) and x + 1 < (gameState.data.layout.width // 2):
                        area[(x, y)].append((x + 1, y))
                    if not gameState.hasWall(x - 1, y):
                        area[(x, y)].append((x - 1, y))
                    if not gameState.hasWall(x, y + 1):
                        area[(x, y)].append((x, y + 1))
                    if not gameState.hasWall(x, y - 1):
                        area[(x, y)].append((x, y - 1))
        deadends = {}
        # We do this twice to make sure all deadends have the correct depth because during the first iteration a dead
        # end could be identified but it may be apart of a larger dead end which has not been found yet. So the second
        # iteration corrects this and gets the correct depth
        for position in area.keys():
            # We define a dead end as a position with only one move. This is not necessary true and is a weakness of our
            # agent
            if len(area[position]) == 1:
                depth = 1
                deadend = [(position, depth)]
                prev_position = position
                endlist = [position]
                position = area[position][0]
                deadend = (self.deadEnd(position, prev_position, area, deadends, deadend, depth, endlist))
                index = -1
                for data in deadend:
                    if data[0] in deadends:
                        if deadends[data[0]] < deadend[index][1]:
                            deadends[data[0]] = deadend[index][1]
                    else:
                        deadends[data[0]] = deadend[index][1]
                    index -= 1
        for position in area.keys():
            if len(area[position]) == 1:
                depth = 1
                deadend = [(position, depth)]
                prev_position = position
                endlist = [position]
                position = area[position][0]
                deadend = (self.deadEnd(position, prev_position, area, deadends, deadend, depth, endlist))
                index = -1
                for data in deadend:
                    if data[0] in deadends:
                        if deadends[data[0]] < deadend[index][1]:
                            deadends[data[0]] = deadend[index][1]
                    else:
                        deadends[data[0]] = deadend[index][1]
                    index -= 1
        return deadends

    def redDeadEnd(self, gameState):
        """
        Returns the positions that are in a deadend on the blue side
        """
        max_y = gameState.data.layout.height
        min_y = 1
        area = {}
        for x in range((gameState.data.layout.width // 2), gameState.data.layout.width):
            for y in range(min_y, max_y):
                if not gameState.hasWall(x, y):
                    area[(x, y)] = []
                    if not gameState.hasWall(x + 1, y):
                        area[(x, y)].append((x + 1, y))
                    if not gameState.hasWall(x - 1, y) and x - 1 >= (gameState.data.layout.width // 2):
                        area[(x, y)].append((x - 1, y))
                    if not gameState.hasWall(x, y + 1):
                        area[(x, y)].append((x, y + 1))
                    if not gameState.hasWall(x, y - 1):
                        area[(x, y)].append((x, y - 1))
        deadends = {}
        # We do this twice to make sure all deadends have the correct depth because during the first iteration a dead
        # end could be identified but it may be apart of a larger dead end which has not been found yet. So the second
        # iteration corrects this and gets the correct depth
        for position in area.keys():
            # We define a dead end as a position with only one move. This is not necessary true and is a weakness of our
            # agent
            if len(area[position]) == 1:
                depth = 1
                deadend = [(position, depth)]
                prev_position = position
                endlist = [position]
                position = area[position][0]
                deadend = (self.deadEnd(position, prev_position, area, deadends, deadend, depth, endlist))
                index = -1
                for data in deadend:
                    if data[0] in deadends:
                        if deadends[data[0]] < deadend[index][1]:
                            deadends[data[0]] = deadend[index][1]
                    else:
                        deadends[data[0]] = deadend[index][1]
                    index -= 1
        for position in area.keys():
            if len(area[position]) == 1:
                depth = 1
                deadend = [(position, depth)]
                prev_position = position
                endlist = [position]
                position = area[position][0]
                deadend = (self.deadEnd(position, prev_position, area, deadends, deadend, depth, endlist))
                index = -1
                for data in deadend:
                    if data[0] in deadends:
                        if deadends[data[0]] < deadend[index][1]:
                            deadends[data[0]] = deadend[index][1]
                    else:
                        deadends[data[0]] = deadend[index][1]
                    index -= 1
        return deadends

    def deadEnd(self, position, prev_position, area, deadends, deadend, depth, endlist):
        """
        Finds positions that are apart of a dead end or not
        """
        if self.isDeadEnd(area[position], deadends, endlist):
            for pos in area[position]:
                if pos == prev_position:
                    continue
                elif pos in endlist:
                    continue
                else:
                    prev_position = position
                    position = pos
                    depth += 1
                    deadend.append((prev_position, depth))
                    endlist.append(prev_position)
                    deadend = self.deadEnd(position, prev_position, area, deadends, deadend, depth, endlist)
        return deadend

    def isDeadEnd(self, positions, deadends, endlist):
        """
        Check if the position is in a dead end or not
        """
        count = 0
        for pos in positions:
            if pos in endlist:
                count += 1
            elif pos in deadends:
                count += 1
        if count < len(positions)-1:
            return False
        else:
            return True

    def foodFinder(self, deadend, foods):
        """
        Finds food that is in a dead end
        """
        deadFood = {}
        for food in foods:
            if food in deadend:
                deadFood[food] = deadend[food]
        return deadFood

    def initializePrior(self, agent):
        """
        Initialize all the possible positions that the enemy agent could be in
        """
        self.beliefs[agent] = util.Counter()
        for p in self.legalPositions:
            self.beliefs[agent][p] = 1.0
        self.beliefs[agent].normalize()

    def updatePosition(self, agent, position):
        """
        If we can see the enemy agent then we can update our beliefs
        """
        updatedBeliefs = util.Counter()
        updatedBeliefs[position] = 1.0
        self.beliefs[agent] = updatedBeliefs

    def newPosition(self, agent, gameState):
        """
        Update all the possible locations that the enemy agent could be in
        """
        updatedBeliefs = util.Counter()
        for oldPosition, oldProbability in self.beliefs[agent].items():
            x = oldPosition[0]
            y = oldPosition[1]
            newPos = util.Counter()
            # The agent could move in any direction
            for p in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if p in self.legalPositions:
                    # but we know if the it is a pacman or not. Therefore we can remove positions that it is not in.
                    if self.isRed:
                        if (p[0] < (gameState.data.layout.width // 2)) and gameState.getAgentState(agent).isPacman:
                            newPos[p] = 1.0
                        elif (p[0] < (gameState.data.layout.width // 2)) and not gameState.getAgentState(agent).isPacman:
                            newPos[p] = 0
                        elif (p[0] >= (gameState.data.layout.width // 2)) and gameState.getAgentState(agent).isPacman:
                            newPos[p] = 0
                        elif (p[0] >= (gameState.data.layout.width // 2)) and not gameState.getAgentState(agent).isPacman:
                            newPos[p] = 1.0
                    else:
                        if (p[0] >= (gameState.data.layout.width // 2)) and gameState.getAgentState(agent).isPacman:
                            newPos[p] = 1.0
                        elif (p[0] >= (gameState.data.layout.width // 2)) and not gameState.getAgentState(agent).isPacman:
                            newPos[p] = 0
                        elif (p[0] < (gameState.data.layout.width // 2)) and gameState.getAgentState(agent).isPacman:
                            newPos[p] = 0
                        elif (p[0] < (gameState.data.layout.width // 2)) and not gameState.getAgentState(agent).isPacman:
                            newPos[p] = 1.0
            newPos.normalize()
            for newPosition, newProbability in newPos.items():
                updatedBeliefs[newPosition] += newProbability * oldProbability
        # We also kmow if move gets eaten then the agent could be there
        lastObserved = self.getPreviousObservation()
        if lastObserved:
            lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList()
                        if food not in self.getFoodYouAreDefending(gameState).asList()]
            for f in lostFood:
                updatedBeliefs[f] = 1.0 / len(self.getOpponents(gameState))
        self.beliefs[agent] = updatedBeliefs

    def observe(self, agent, noisyDistance, gameState):
        """
        Update the likelihoods based off the current observation
        """
        myPosition = self.inSonar(self.index, gameState)
        teammatePositions = [self.inSonar(teammate, gameState) for teammate in self.getTeam(gameState)]
        updatedBeliefs = util.Counter()

        for p in self.legalPositions:
            if any([util.manhattanDistance(teammatePos, p) <= 5 for teammatePos in teammatePositions]):
                updatedBeliefs[p] = 0.0
            else:
                trueDistance = util.manhattanDistance(myPosition, p)
                positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)
                updatedBeliefs[p] = positionProbability * self.beliefs[agent][p]

        if not updatedBeliefs.totalCount():
            self.initializePrior(agent)
        else:
            updatedBeliefs.normalize()
            self.beliefs[agent] = updatedBeliefs

    def inSonar(self, agent, gameState):
        """
        Return the position of the agent
        """
        pos = gameState.getAgentPosition(agent)
        if pos:
            return pos
        else:
            return self.beliefs[agent].argMax()

class OffensiveAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def specialSituations(self, gameState, myPosition, actions, foodList, ghosts, enemies, closestEnemy):
        """
        Return special actions for specific situations
        """
        node = Node(myPosition, None, None, 0)
        newghosts = self.ghostPostions(ghosts, gameState)
        distanceToHome = self.goHome(node, self.mid_line, newghosts, gameState)
        if self.isRed:
            self.holdingFood = self.maxFood - self.getScore(gameState) - len(
                self.getFood(gameState).asList())
        else:
            self.holdingFood = self.maxFood + self.getScore(gameState) - len(
                self.getFood(gameState).asList())
        # Find a crossing point that is not patrolled by an enemy ghost
        if len(ghosts) != 0 and myPosition in self.crossingPoint and min(
                [self.getMazeDistance(myPosition, ghost) for ghost in ghosts]) < 5:
            start = self.mid_line.copy()
            start.remove(myPosition)
            self.crossingPoint = [random.choice(start)]
        # Remove point if we have crossed over into enemy side
        elif myPosition in self.crossingPoint or (self.isRed and myPosition[0] > self.mid_line[0][0]) or (
                not self.isRed and myPosition[0] < self.mid_line[0][0]):
            self.crossingPoint = []
        elif self.crossingPoint == [] and ((self.isRed and myPosition[0] <= self.mid_line[0][0]) or (
                not self.isRed and myPosition[0] >= self.mid_line[0][0])):
            start = self.mid_line.copy()
            if myPosition in self.mid_line:
                start.remove(myPosition)
            self.crossingPoint = [random.choice(start)]
        # Find a path to the crossing point
        elif self.crossingPoint:
            goTo = self.goHome(node, self.crossingPoint, newghosts, gameState)[0]
            if goTo is not None:
                return goTo
        # Go home if there is no time left or we have collected a lot of food/all of the food
        if distanceToHome[1] + 1 >= self.steps or len(foodList) <= 2:
            if distanceToHome[0] is not None:
                return distanceToHome[0]
        if len(ghosts) != 0:
            ghost_distance = min([self.getMazeDistance(myPosition, ghost) for ghost in ghosts])
        else:
            ghost_distance = closestEnemy
        # Eat as much food as possible if the enemies are scared
        if (enemies[0].scaredTimer > 0 or enemies[0].isPacman) and (
                enemies[1].scaredTimer > 0 or enemies[1].isPacman) and len(foodList) > 2:
            minDistance = min(self.getFood(gameState).asList(), key=lambda x: self.getMazeDistance(myPosition, x))
            states = [(gameState.generateSuccessor(self.index, a).getAgentState(self.index).getPosition(), a) for a in
                      actions]
            bestStates = []
            for act in states:
                if act[0] not in self.enemyDeadEnd:
                    bestStates.append(act)
                elif 2 * self.enemyDeadEnd[act[0]] < enemies[0].scaredTimer and 2 * self.enemyDeadEnd[act[0]] < enemies[1].scaredTimer:
                    bestStates.append(act)
                elif closestEnemy > 2*self.enemyDeadEnd[act[0]]:
                    bestStates.append(act)
                elif enemies[0].isPacman and enemies[1].isPacman:
                    bestStates.append(act)
            if bestStates:
                action = min(bestStates, key=lambda x: self.getMazeDistance(minDistance, x[0]))
                return action[1]
        # Find food that is safe to eat
        availableFood = []
        # Find food that we can eat without getting eaten
        for food in self.getFood(gameState).asList():
            if food not in self.foodInDeadEnd:
                availableFood.append(food)
            elif 2 * self.foodInDeadEnd[food] < ghost_distance:
                availableFood.append(food)
        # Go to capsule if we are getting chased
        if ghost_distance <= 5 and len(self.getCapsules(gameState)) > 0 and not availableFood:
            goTo = self.goHome(node, self.getCapsules(gameState), newghosts, gameState)[0]
            if goTo is not None:
                return goTo
        # If there is no food, go home
        if not availableFood:
            goTo = self.goHome(node, self.mid_line, newghosts, gameState)[0]
            self.eatableFood = []
            if goTo is not None:
                return goTo
        else:
            self.eatableFood = availableFood
        return None

    def chooseAction(self, gameState):
        """
        Choose an action depending on the gameState
        """
        startTime = time.time()
        self.steps = self.steps - 1
        myPosition = gameState.getAgentState(self.index).getPosition()
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        foodList = self.getFood(gameState).asList()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() is not None and a.scaredTimer <= 1]
        myPosition = gameState.getAgentState(self.index).getPosition()
        noisyDistances = gameState.getAgentDistances()
        # Update the enemy locations
        for opponent in self.getOpponents(gameState):
            enemyPos = gameState.getAgentPosition(opponent)
            if enemyPos:
                self.updatePosition(opponent, enemyPos)
            else:
                self.newPosition(opponent, gameState)
                self.observe(opponent, noisyDistances[opponent], gameState)
        closestEnemy = min([self.getMazeDistance(self.beliefs[opponent].argMax(), myPosition) for opponent in self.getOpponents(gameState)])
        bestActions = []
        # Check if we are in a special situation
        check = self.specialSituations(gameState, myPosition, actions, foodList, ghosts, enemies, closestEnemy)
        if check is not None:
            return check
        # Else get all available moves
        x = myPosition[0]
        y = myPosition[1]
        for a in actions:
            newPos = None
            if a == Directions.NORTH:
                newPos = (x, y+1)
            elif a == Directions.SOUTH:
                newPos = (x, y-1)
            elif a == Directions.EAST:
                newPos = (x+1, y)
            elif a == Directions.WEST:
                newPos = (x-1, y)
            # If that move will kill us, remove it
            if len(ghosts) == 0 or min([self.getMazeDistance(newPos, ghost) for ghost in ghosts]) > 1:
                # If the move is safe, add it
                if newPos not in self.enemyDeadEnd:
                    bestActions.append(a)
                # If we can enter and leave the deadend before a ghost will kill us then add the move
                elif len(ghosts) != 0 and min([self.getMazeDistance(newPos, ghost) for ghost in ghosts]) > 2*self.enemyDeadEnd[newPos]:
                    bestActions.append(a)
                # Check if we can enter and leave before the enemy comes
                elif closestEnemy > 2*self.enemyDeadEnd[newPos]:
                    bestActions.append(a)
                # If there no ghosts we can safely enter shallow dead ends
                elif len(ghosts) == 0 and 5 > 2*self.enemyDeadEnd[newPos]:
                    bestActions.append(a)
                # If we are in a dead end, exiting it is always a good move
                elif myPosition in self.enemyDeadEnd and self.enemyDeadEnd[newPos] < self.enemyDeadEnd[myPosition]:
                    bestActions.append(a)
        if not bestActions:
            bestActions = actions
        # Go to closest available food
        if len(self.eatableFood) > 0:
            minDistance = min(self.eatableFood, key=lambda x: self.getMazeDistance(myPosition, x))
            states = [(gameState.generateSuccessor(self.index, a).getAgentState(self.index).getPosition(), a) for a in
                      bestActions]
            action = min(states, key=lambda x: self.getMazeDistance(minDistance, x[0]))
            return action[1]
        # Go to food
        if len(foodList) > 0:
            minDistance = min(foodList, key=lambda x: self.getMazeDistance(myPosition, x))
            states = [(gameState.generateSuccessor(self.index, a).getAgentState(self.index).getPosition(), a) for a in
                      bestActions]
            action = min(states, key=lambda x: self.getMazeDistance(minDistance, x[0]))
            return action[1]
        # If all else fails use MCTS search
        else:
            return self.mcts_search.MCTS_search(gameState, startTime)


    def goHome(self,  node, home_line, ghosts, state):
        """
        Astar search to find home
        Based on previous assignments
        """
        open = [(self.heuristic(home_line, node.getLocation()), node)]
        closed = set()
        best_path = {node.getLocation(): 0}
        while open:
            node = open.pop(0)[1]
            if node.getLocation() not in closed or node.getCost() < best_path[node.getLocation()]:
                closed.add(node.getLocation())
                best_path[node.getLocation()] = node.getCost()
                if node.getLocation() in home_line:
                    path = self.findPath(node)
                    if path[1] == 0:
                        return None, -1
                    else:
                        return [path[0][0], path[1]]
                for successor in self.getSuccessors(node.getLocation(), state):
                    new_node = Node(successor[0], node, successor[1], node.getCost() + 1)
                    value = self.heuristic(home_line, new_node.getLocation()) + new_node.getCost()
                    if value < float("inf") and new_node.getLocation() not in ghosts:
                        open.append((value, new_node))
                        if len(open) > 1:
                            open.sort(key=lambda x: x[0])
                        best_path[new_node.getLocation()] = new_node.getCost()
        return None, -1

    def getSuccessors(self, currentPosition, state):
        """
      Based on previous assignments
      """
        walls = state.getWalls()
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = 1
                successors.append((nextState, action, cost))
        return successors

    def heuristic(self, home_line, pos):
        """
        Manhattan distance is used as the heuristic as it is admissible and easy to compute.
        """
        cost = 0
        for point in home_line:
            heuristic = util.manhattanDistance(point, pos)
            if heuristic > cost:
                cost = heuristic
        return cost

    def findPath(self, node):
        """
        Find the path from the goal node to the start node
        :param node: the goal node
        :return: A list of actions
        """
        path = []
        if node is None:
            print("Error could not find path")
            return None
        else:
            while node.getParent() is not None:
                path.append(node.getAction())
                node = node.getParent()
            return [path[::-1], len(path[::-1])]

    def ghostPostions(self, ghosts, gameState):
        """
        Simple function to find the possible position for the ghost after one turn
        """
        positions = []
        if not ghosts:
            return positions
        else:
            for ghost in ghosts:
                positions.append(ghost)
                x = int(ghost[0])
                y = int(ghost[1])
                if not gameState.hasWall(x + 1, y):
                    positions.append((x + 1, y))
                if not gameState.hasWall(x - 1, y):
                    positions.append((x - 1, y))
                if not gameState.hasWall(x, y + 1):
                    positions.append((x, y + 1))
                if not gameState.hasWall(x, y - 1):
                    positions.append((x, y - 1))
            return positions


class DefensiveAgent(OffensiveAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        if gameState.getAgentState(self.index).getPosition()[0] == 1:
            self.isRed = True
        else:
            self.isRed = False
        self.mid_line = self.midwayLine(gameState)
        self.patrol = None
        self.defendingFood = self.getFoodYouAreDefending(gameState).asList()
        self.patrolArea = [self.closestFood(gameState)]
        self.entrances = self.deadEndFinder(self.isRed, gameState)
        self.maxFood = len(self.getFood(gameState).asList())
        self.enemyDeadEnd = self.badpaths(self.isRed, gameState)
        self.steps = 300
        self.foodInDeadEnd = self.foodFinder(self.enemyDeadEnd, self.getFood(gameState).asList())
        self.crossingPoint = []
        self.holdingFood = 0
        self.legalPositions = [p for p in gameState.getWalls().asList(False)]
        self.beliefs = {}
        self.eatableFood = []
        for opponent in self.getOpponents(gameState):
            self.initializePrior(opponent)
        # For MCTS
        self.gameState = gameState
        self.initialFoodToEat = self.getFood(self.gameState).asList()
        self.initialCapsulesToDefend = self.getCapsulesYouAreDefending(gameState)
        self.timeLimit = MOVE_MAX_TIME
        self.mcts_search = MCTS(EXPLORATION_CONSTANT, self.timeLimit, MAX_AGENT_MOVES, self, gameState.getAgentPosition(self.index))
        self.home = gameState.getInitialAgentPosition(self.index)
        self.startDirection = gameState.getAgentState(self.index).getDirection()
        self.moves = 0


    def closestFood(self, gameState):
        """
        Find closest Food
        """
        foods = self.getFoodYouAreDefending(gameState).asList()
        dists = [(self.getMazeDistance(position, food), position) for position in self.mid_line for food in foods]
        closestFood = min(dists, key=lambda x: x[0])
        return random.choice([x[1] for x in dists if x[0] == closestFood[0]])

    def pacmanFinder(self, gameState, myPosition):
        """
        Find the pacman that is the closest to the ghost
        """
        location = []
        for opponent in self.getOpponents(gameState):
            position = self.beliefs[opponent].argMax()
            if self.isRed:
                if position[0] < gameState.data.layout.width // 2:
                    location.append(position)
            else:
                if position[0] >= gameState.data.layout.width // 2:
                    location.append(position)
        if location:
            dists = [(self.getMazeDistance(loc, myPosition), loc) for loc in location]
            dist = min(dists, key=lambda x: x[0])
            return random.choice([x[1] for x in dists if x[0] == dist[0]])
        else:
            dists = [(self.getMazeDistance(loc, position), loc) for loc in location for position in self.mid_line]
            dist = min(dists, key=lambda x: x[0])
            return random.choice([x[1] for x in dists if x[0] == dist[0]])


    def chooseAction(self, gameState):
        agentPos = gameState.getAgentPosition(self.index)
        # If we are at the patrol point, reset the partol.
        if agentPos == self.patrol:
            self.patrol = None
        # If we are scared, go on the offensive
        if gameState.getAgentState(self.index).scaredTimer > 0:
            return OffensiveAgent.chooseAction(self, gameState)
        # Look for enemy Pacmen
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        # If we see an enemy Pacman, chase the closest one
        if len(invaders) > 0:
            dists = [(a.getPosition(), self.getMazeDistance(agentPos, a.getPosition())) for a in invaders]
            self.patrol = min(dists, key=lambda x: x[1])[0]
            if self.patrol in self.entrances and self.getMazeDistance(self.patrol, self.entrances[self.patrol]) < 4:
                self.patrol = self.entrances[self.patrol]
            elif self.patrol in self.entrances and self.getMazeDistance(self.patrol, agentPos) < 5:
                self.patrol = agentPos
        # Else if we notice some food has gone missing, head to the missing food area
        elif len(self.defendingFood) != len(self.getFoodYouAreDefending(gameState).asList()):
            missing = set(self.defendingFood) - set(self.getFoodYouAreDefending(gameState).asList())
            if len(missing) > 0:
                self.patrol = missing.pop()
            self.defendingFood = self.getFoodYouAreDefending(gameState).asList()
        # If nothing to patrol, go on the offensive
        if self.patrol is None and not enemies[0].isPacman and not enemies[1].isPacman:
            return OffensiveAgent.chooseAction(self, gameState)
        # If there is something to patrol, go to it.
        else:
            noisyDistances = gameState.getAgentDistances()
            # Update the opponent positions
            for opponent in self.getOpponents(gameState):
                enemyPos = gameState.getAgentPosition(opponent)
                if enemyPos:
                    self.updatePosition(opponent, enemyPos)
                else:
                    self.newPosition(opponent, gameState)
                    self.observe(opponent, noisyDistances[opponent], gameState)
            self.steps = self.steps - 1
            if self.patrol is None:
                self.patrol = self.pacmanFinder(gameState, agentPos)
            if agentPos == self.patrol:
                return Directions.STOP
            actions = gameState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            if gameState.getAgentState(self.index).isPacman:
                node = Node(agentPos, None, None, 0)
                ghosts = [a.getPosition() for a in enemies if a.getPosition() is not None]
                newghosts = self.ghostPostions(ghosts, gameState)
                direction = self.goHome(node, [self.patrol], newghosts, gameState)[0]
                if direction is not None:
                    return direction
                else:
                    return random.choice(actions)
            else:
                node = Node(agentPos, None, None, 0)
                if self.isRed:
                    direction = self.goHome(node, [self.patrol], [(x, y) for x in range(gameState.data.layout.width // 2, gameState.data.layout.width) for y in range(1, gameState.data.layout.height)], gameState)[0]
                else:
                    direction = self.goHome(node, [self.patrol], [(x, y) for x in range(1, gameState.data.layout.width // 2) for y in range(1, gameState.data.layout.height)], gameState)[0]
                if direction is not None:
                    return direction
                else:
                    return random.choice(actions)

    def deadEndFinder(self, isRed, gameState):
        """
        Find dead ends that are on our side
        """
        max_y = gameState.data.layout.height
        min_y = 1
        area = {}
        if isRed:
            for x in range(1, gameState.data.layout.width // 2):
                for y in range(min_y, max_y):
                    if not gameState.hasWall(x, y):
                        area[(x, y)] = []
                        if not gameState.hasWall(x+1, y) and x + 1 < (gameState.data.layout.width // 2):
                            area[(x, y)].append((x+1, y))
                        if not gameState.hasWall(x-1, y):
                            area[(x, y)].append((x-1, y))
                        if not gameState.hasWall(x, y+1):
                            area[(x, y)].append((x, y+1))
                        if not gameState.hasWall(x, y-1):
                            area[(x, y)].append((x, y-1))
            deadends = {}
            for position in area.keys():
                if len(area[position]) == 1:
                    deadend = [position]
                    prev_position = position
                    position = area[position][0]
                    while len(area[position]) == 2:
                        if area[position][0] != prev_position:
                            prev_position = position
                            position = area[position][0]
                        else:
                            prev_position = position
                            position = area[position][1]
                        deadend.append(prev_position)
                    for pos in deadend:
                        deadends[pos] = position
            return deadends
        else:
            for x in range((gameState.data.layout.width // 2), gameState.data.layout.width):
                for y in range(min_y, max_y):
                    if not gameState.hasWall(x, y):
                        area[(x, y)] = []
                        if not gameState.hasWall(x + 1, y):
                            area[(x, y)].append((x + 1, y))
                        if not gameState.hasWall(x - 1, y) and x - 1 >= (gameState.data.layout.width // 2):
                            area[(x, y)].append((x - 1, y))
                        if not gameState.hasWall(x, y + 1):
                            area[(x, y)].append((x, y + 1))
                        if not gameState.hasWall(x, y - 1):
                            area[(x, y)].append((x, y - 1))
            deadends = {}
            for position in area.keys():
                if len(area[position]) == 1:
                    deadend = [position]
                    prev_position = position
                    position = area[position][0]
                    while len(area[position]) == 2:
                        if area[position][0] != prev_position:
                            prev_position = position
                            position = area[position][0]
                        else:
                            prev_position = position
                            position = area[position][1]
                        deadend.append(prev_position)
                    for pos in deadend:
                        deadends[pos] = position
            return deadends

    def deadendRoom(self, area, deadend, deadends, position, prev_position):
        for pos in area[position]:
            if pos == prev_position:
                continue
            elif pos in deadend:
                continue
            elif pos in deadends:
                continue
            else:
                prev_position = position
                position = pos
                deadend.append(prev_position)
                self.deadendRoom(area, deadend, deadends, position, prev_position)
        return deadend
