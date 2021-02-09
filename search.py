# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def _dfs(problem, state, visited, solution):

    if problem.isGoalState(state):
        return True # return sucessfull path

    if visited[state] == 1:
        return False # return unsucessfull path

    visited[state] = 1
    for successor in problem.getSuccessors(state):
        if visited[successor[0]] == 0:
            if problem.isGoalState(successor[0]) or _dfs(problem, successor[0], visited, solution):
                solution.append(successor[1])
                return True
    
    return False


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visited = util.Counter()
    solution = []
    cost = _dfs(problem, problem.getStartState(), visited, solution)
    solution.reverse()
    return solution

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Use a Queue, so the search explores all nodes on one level before moving to the next level 
    queue = util.Queue()
    # Make an empty list of explored nodes
    visited = []
    # Place the starting point in the queue
    queue.push((problem.getStartState(), []))
    while queue:
        node, solution = queue.pop()
        if not node in visited:
            visited.append(node)
            if problem.isGoalState(node):
                return solution
            for successor in problem.getSuccessors(node):
                coordinate, direction, cost = successor
                nextSolution = solution + [direction]
                queue.push((coordinate, nextSolution))
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited         = util.Counter()
    parent          = util.Counter()
    parent_action   = util.Counter()
    
    dist    = dict()

    dist[problem.getStartState()] = 0

    visited[problem.getStartState()] = 1

    queue = util.Queue()
    queue.push(problem.getStartState())

    goal = (float("inf"), float("inf"))

    while queue:
        p = queue.pop()

        if problem.isGoalState(p):
            goal = p
            break

        for s in problem.getSuccessors(p):
            if s[0] not in dist or dist[s[0]] > dist[p] + s[2]:
                parent[s[0]] = p
                parent_action[s[0]] = s[1]

                dist[s[0]] = dist[p] + s[2]
                if visited[s[0]] == 0:
                    visited[s[0]] = 1
                    queue.push(s[0])

    if goal == (float("inf"), float("inf")):
        print("ERROR")
        return []

    last = goal

    solution = []

    while parent[last] != problem.getStartState():
        solution.append(parent_action[last])
        last = parent[last]

    solution.append(parent_action[last])
    solution.reverse()

    return solution


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def expandNodeGreedy(problem, state, heuristic=nullHeuristic):
    childs = util.PriorityQueue()

    for successor in problem.getSuccessors(state):
        node = (successor[0], successor[1])
        heu = heuristic(successor[0], problem)

        if problem.isGoalState(node):
            solution = util.Stack()
            solution.push(node[1])
            return solution

        childs.push(node, heu) 

    while not childs.isEmpty():
        node = childs.pop()
        solution = expandNodeGreedy(problem, node[0], heuristic)

        if(not solution.isEmpty()):
            solution.push(node[1])
            return solution

    return util.Stack()


def _greedySearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest heuristic first."""
    "*** YOUR CODE HERE ***"
    s = expandNodeGreedy(problem, problem.getStartState(), heuristic)

    solution = list()

    while not s.isEmpty():
        solution.append(s.pop())
    
    return solution


def greedySearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest heuristic first."""
    "*** YOUR CODE HERE ***"
    state = problem.getStartState()
    isVisited = util.Counter()
    solution = []

    while(not problem.isGoalState(state)):
        best = float('inf')
        isVisited[state] = 1
        action = None
        for neighbor in problem.getSuccessors(state):
            cost = heuristic(neighbor[0], problem)
            
            if best > cost and isVisited[neighbor[0]] == 0:
                best = cost
                action = neighbor[1]
                state = neighbor[0]
        
        solution.append(action)

    return solution

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    queue = util.PriorityQueue()

    isClosed = util.Counter()

    node = (problem.getStartState(), [])
    cost = heuristic(problem.getStartState(), problem)

    queue.push(node, cost)
    
    while not queue.isEmpty():
        node, solution = queue.pop()

        if problem.isGoalState(node):
            return solution

        if isClosed[node] == False:
            isClosed[node] = True

            for successor in problem.getSuccessors(node):
                coordinate, direction, cost = successor
                nextSolution = solution + [direction]

                dn = problem.getCostOfActions(nextSolution)
                cn = heuristic(coordinate, problem)
                gn = dn + cn

                queue.push((coordinate, nextSolution), gn)
    return []



# def aStarSearch(problem, heuristic=nullHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     state = problem.getStartState()

#     openSet     = util.PriorityQueue()
#     closeSet     = set()
#     openSet.push(state, 0)

#     cameFrom            = dict()
#     cameFromAction      = dict()
#     distances              = dict()
#     distances[state]       = 0

#     fScore = dict()
#     fScore[state] = heuristic(state, problem)

#     current = None

#     while not openSet.isEmpty():
#         current = openSet.pop()

#         if(problem.isGoalState(current)):
#             return reconstruct_path(cameFrom, cameFromAction, current)

#         # print(len(openSet.heap))
#         # print(problem.getSuccessors(current))
#         closeSet.add(current)

#         for neighbor in problem.getSuccessors(current):
#             # print("state", _state)
#             # print("closeSet", closeSet)
    
#             distance = distances[current] + neighbor[2]
#             # print("tentative", distance)

#             if neighbor[0] not in distances or distance <= distances[neighbor[0]]:
#                 cameFrom[neighbor[0]] = current
#                 cameFromAction[neighbor[0]] = neighbor[1]

#                 distances[neighbor[0]] = distance
#                 fScore[neighbor[0]] = distances[neighbor[0]] + heuristic(neighbor[0], problem)
#                 # print("_state", _state)
                
#                 if neighbor[0] not in closeSet:
#                     openSet.push(neighbor[0], fScore[neighbor[0]])

#     # return reconstruct_path(cameFrom, cameFromAction, current)
#     return []

# def _aStarSearch(problem, heuristic=nullHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     _state           = problem.getStartState()
#     _open            = util.Counter()
#     _closed          = util.Counter()
#     _node            = util.Counter()

#     _tree            = util.Counter()
    

#     _node['parent']  = 'Null'
#     _node['action']  = 'Null'
#     _node['dist']    = 0
#     _node['cost']    = 0 + heuristic(_state, problem)
#     _node['state']   = _state
 
#     _frontier = util.PriorityQueue()
#     _frontier.push(_node, 0)

#     _open[_state] = 1
#     _closed[_state] = 0

#     solution = list()


#     while(True):
#         if _frontier.isEmpty():
#             goal = current['state']
#             break

#         current = _frontier.pop()

#         if problem.isGoalState(current['state']):
#             print("GOAL")
#             break

#         print(current['state'])

#         _open[current['state']]      = 0
#         _closed[current['state'][0]]    = 1

#         for successor in problem.getSuccessors(current['state']):
#             # if _closed[successor[0]] == 0:
#                 _node            = util.Counter()
#                 _node['state']   = successor[0]
#                 _node['dist']    = successor[2] + current['dist']
#                 _node['action']  = successor[1]
#                 _node['parent']  = current

#                 if _closed[successor[0][0]] == 0:
#                     _open[_node['state']] = 1
#                     _frontier.push(_node, _node['dist'] + heuristic(_node['state'], problem))
        

#     while goal != problem.getStartState():
#         solution.append(_parent[goal]['action'])
#         goal = _parent[goal]['state']
    

#     solution.reverse()
#     solution.pop()

#     print(solution)
#     return solution


# import graphProblem

# import math
# def distance(p0, p1):
#     return math.sqrt(math.pow(p0[0] - p1[0], 2) + math.pow(p0[1] - p1[1], 2))


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"


    distances = list()
    distances_food = list()

    distances_food.append(0)
    
    for food in foodGrid.asList():
        distances.append(util.manhattanDistance(position, food))
        for tofood in foodGrid.asList():
            distances_food.append(util.manhattanDistance(food, tofood))

    if len(distances):
        return min(distances) + max(distances_food) 
    else:
        return max(distances_food)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
gs = greedySearch
astar = aStarSearch
