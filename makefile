run:
	python3 pacman.py --layout tinyMaze

dfs:
	python3 pacman.py --layout mediumMaze --pacman SearchAgent --agentArgs fn=dfs

bfs:
	python3 pacman.py --layout openMaze --pacman SearchAgent --agentArgs fn=bfs

ucs:
	python3 pacman.py --layout mediumMaze --pacman SearchAgent --agentArgs fn=ucs

heu:
	python3 pacman.py --layout mediumMaze --pacman SearchAgent --agentArgs fn=gs,heuristic=manhattanHeuristic

astar:
	python3 pacman.py --layout trickySearch --pacman SearchAgent --agentArgs fn=astar,heuristic=manhattanHeuristic

flood:
	python3 pacman.py --layout trickySearch --pacman SearchAgent --agentArgs fn=astar,heuristic=foodHeuristic

tests:
	python autograder.py --question passo1
	python autograder.py --question passo2
	python autograder.py --question passo3
	python autograder.py --question passo4
	python autograder.py --question passo5
	python autograder.py --question passo6
	python autograder.py --question passo7