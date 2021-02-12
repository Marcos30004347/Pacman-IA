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
    queue = util.Queue()

    pushed = util.Counter()

    queue.push((problem.getStartState(), []))
    pushed[problem.getStartState()] = 1

    while not queue.isEmpty():
        node, solution = queue.pop()
        if problem.isGoalState(node):
            return solution

        for neighbor in problem.getSuccessors(node):
            newSolution = solution + [neighbor[1]]

            if pushed[neighbor[0]] == 0:
                pushed[neighbor[0]] = 1
                queue.push((neighbor[0], newSolution))

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    node = (problem.getStartState(), [])

    wasVisited = util.Counter()
    distances  = dict()

    queue = util.PriorityQueue()
    queue.push(node, 0)

    distances[problem.getStartState()] = 0

    while True:
        if queue.isEmpty():
            return []
        
        state, solution = queue.pop()
        
        if problem.isGoalState(state):
            return solution
        
        wasVisited[state] = 1

        for neighbor in problem.getSuccessors(state):
            if neighbor[0] not in distances or distances[state] + neighbor[2] < distances[neighbor[0]]:
                distances[neighbor[0]] = distances[state] + neighbor[2]
                if wasVisited[neighbor[0]] == 0:
                    queue.push((neighbor[0], solution + [neighbor[1]]) , distances[neighbor[0]])
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def greedySearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    wasVisited = util.Counter()
    solution = []

    startstate = problem.getStartState()

    if problem.isGoalState(startstate):
        return 'Stop'
    queue.push(
        [
            #(state, action, distance), priority
            [(startstate, 'Stop', 0), 0]
        ], 
        heuristic(startstate, problem)
    )

    while not queue.isEmpty():
        nodes = queue.pop()

        node = nodes[-1]
        if problem.isGoalState(node[0][0]):
            for path in nodes[1:]:
                # Append actions
                solution.append(path[0][1])
            return solution

        if wasVisited[node[0][0]] == 0:
            wasVisited[node[0][0]] = 1

            for successor in problem.getSuccessors(node[0][0]):
                if wasVisited[successor[0]] == 0:
                    cost = node[1] + successor[2]
                    priority = heuristic(successor[0], problem)
                    currentpath = nodes[:] # copy current history
                    currentpath.append([successor, cost])
                    queue.push(currentpath, priority)
    return []

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

    dist = list()
    dist_foods = list()

    dist_foods.append(0)
    
    for food in foodGrid.asList():
        dist.append(util.manhattanDistance(position, food))
        for to_food in foodGrid.asList():
            dist_foods.append(util.manhattanDistance(food, to_food))

    if len(dist):
        return min(dist) + max(dist_foods) 
    else:
        return max(dist_foods)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
gs = greedySearch
astar = aStarSearch
