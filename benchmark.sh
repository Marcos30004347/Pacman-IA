declare -a StringArray=("tinyMaze" "mediumMaze" "openMaze" "smallMaze" "contoursMaze" "bigMaze")

if [ -d ./benchmarks ];
then
    rm -rf ./benchmarks
fi

mkdir -p ./benchmarks

for fase in ${StringArray[@]};
do
    for algo in bfs dfs ucs gs astar;
    do
        START=$(date +%s.%N)
        python3 pacman.py --layout $fase --pacman SearchAgent --agentArgs fn=$algo >> ./benchmarks/$fase-benchmark.txt;
        END=$(date +%s.%N)
        DIFF=$(echo "$END - $START" | bc)
        echo "Total Time: $DIFF" >> ./benchmarks/$fase-benchmark.txt;
        echo -e "" >> ./benchmarks/$fase-benchmark.txt;
    done;
done


declare -a StringArray=("tinyMaze" "mediumMaze" "openMaze" "smallMaze" "contoursMaze" "bigMaze")

for fase in ${StringArray[@]};
do

    for algo in gs astar;
    do
        for heu in manhattanHeuristic;
        do
            echo "ALGO $algo"
            START=$(date +%s.%N)
            python3 pacman.py --layout $fase --pacman SearchAgent --agentArgs fn=$algo,heuristic=$heu >> ./benchmarks/$fase-benchmark.txt;
            END=$(date +%s.%N)
            DIFF=$(echo "$END - $START" | bc)
            echo "Total Time: $DIFF" >> ./benchmarks/$fase-benchmark.txt;
            echo -e "" >> ./benchmarks/$fase-benchmark.txt;
        done;
    done;
done

declare -a StringArray=("trickySearch" "tinySearch" "smallSearch")

for fase in ${StringArray[@]};
do
    for algo in foodHeuristic;
    do
        START=$(date +%s.%N)
	    python3 pacman.py --layout $fase --pacman SearchAgent --agentArgs fn=astar,prob=FoodSearchProblem,heuristic=$algo >> ./benchmarks/$fase-benchmark.txt;
        END=$(date +%s.%N)
        DIFF=$(echo "$END - $START" | bc)
        echo "Total Time: $DIFF" >> ./benchmarks/$fase-benchmark.txt;
        echo -e "" >> ./benchmarks/$fase-benchmark.txt;
    done;
done

