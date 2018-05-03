/*
 * This is the class which represent a searching agent. 
 * The goal and start fields represent the start and current cell respectively.
 * The current field represent the current cell of the agent.
 * The 2D array agentgrid represent the grid as perceived by the agent. It store the 
 * information known to the agent as the currecnt situation.
 * Using currentPath and currentPathIndex we keep track of the path travelled by the 
 * agent. It serve two purposes. During A* searches it store the presumed path and 
 * it also stored the current path travelled by the agent.
 *  expandedNodes field is used for keeping tack of total number of expanded nodes
 *  during all A* searches.
 *  gstar field stores g* value of the previous A* search. This is used only in Adaptive A*
 *  search where we need to update the heuristic using the g* value of the previous search. 
 */
public class SearchAgent {
	GridCell goal, start, current;
	GridCell[][] agentgrid;
	GridCell[] currentPath;
	int currentPathIndex;
	int expandedNodes;
	int gstar;
	int closeIndex;
	GridCell[] closed;
	BinaryHeap priorityQueue;
	/*
	 * This is the constuctor method of Search agent.Here
	 * the agent do its intial setup.
	 */
	public SearchAgent(){
		//Initialize the agent grid cell where agent store its 
		//knowledge about the environment
		agentgrid = new GridCell[Astar.h][Astar.w];		
		//Intialize the start and goal cell and put it into 
		//agent grid cell
		agentgrid[Astar.gy][Astar.gx] = new GridCell(Astar.gx,Astar.gy);
		goal = agentgrid[Astar.gy][Astar.gx];
		agentgrid[Astar.sy][Astar.sx] = new GridCell(Astar.sx,Astar.sy);
		start = agentgrid[Astar.sy][Astar.sx];
		//Initialize the current Path. This will store the presumed path
		//when the A* returns and the agent traverse through this path and update 
		//it's knoledge about the environment and store that knowledge in agentgrid.
		currentPath = new GridCell[Astar.h * Astar.w];
		closed = new GridCell[Astar.h * Astar.w];
		currentPathIndex = -1;
		//At start the agent has only one cell, its start location
		//in the current path.
		currentPath[++currentPathIndex] = start;
		//the current location of the agent is set to start
		current = start;
		//Updating the starting position of the agent.The agent has the knowledge 
		//of cells adjacent to the start node.		
		GridCell[] successor = null;
		if(Astar.doBackwardSearch){
			goal.h = manhattanDistance(goal.x,goal.y, start.x, start.y);
			successor = getSuccessorAstar(current,start);						
		}
		else{
			start.h = manhattanDistance(goal.x,goal.y, start.x, start.y);
			successor = getSuccessorAstar(current,goal);			
		}
		//This will check if any cell is blocked near the current cell.
		for (int i=0; i<4; i++) {
			if(successor[i] == null){
				break;
			}			
			if(Astar.envGrid[successor[i].y][successor[i].x] == Astar.BLOCKED){
				successor[i].isblocked = true;										
			}			
		}
		priorityQueue = new BinaryHeap(Astar.h * Astar.w);
		expandedNodes = 0;
	}
	/*
	 * This method do the searching in the grid. It calls the A* search method after setting the 
	 * start and goal cells. After this it prepare the path and update the knowledge of the 
	 * agent about the adjacent cells found in the path. It will repost the dicovery of a path 
	 * follwoing which the agent can travel from start to goal cell and print the grid with the path. 
	 * If there is no path exists then it will report that no path found.
	 */
	void doSearch(){
		//A flag which is set to true when agentReached the goal.
		boolean agentReahcedGoal = false;
		//counter for counting the no of different A* iterations
		int searchID = 0;
		//int preexpandednode = 0;
		//In this while loop the agent set the start and goal cell and call the main 
		//A* seach function. It will the prepare the presumed path and traverse it.
		//While traversing the path it also update the agent knowledge about the 
		//environment grid.
		while(!agentReahcedGoal){			
			//This flag store the result of the main A* search.
			boolean reuslt;
			closeIndex = -1;
			//If the search is backward then set current noad as goal, otherwise set current 
			//as start node.
			if(!Astar.doBackwardSearch){				
				reuslt = aStarSearch(++searchID, current, goal);
			}
			else{				
				reuslt = aStarSearch(++searchID, goal, current);				
			}
			//System.out.println("No of nodes expanded in iteration " +searchID+" is "+  (expandedNodes-preexpandednode));
			//preexpandednode = expandedNodes;
			//If A* returns a path to goal node.
			if(reuslt){
				//set the g* value and prepare the path return by the A* search
				if(Astar.doBackwardSearch){
					//If the search is backward then g* is g value of the current cell
					gstar = current.g;
					preparePathBackward();
				}
				else{
					//If the search is backward g* is g value of the current cell
					gstar = goal.g;
					preparePathForward(goal);
				}
				//Here agent is traversing the path returned by the A* search and 
				//updating its knowledge about the blocked cell.
				for(int i = 0; i < currentPathIndex; ++i){
					//Get the adjacent cells of the i th element in the path.
					GridCell[] ssr = getSuccessor(currentPath[i]);
					for (int j=0; j<4; j++) {
						if(ssr[j] == null){
							break;
						}	
						//If the corrsponding cell in the environment grid is blocked then 
						//mark that cell blocked in the agent grid also. 
						if(Astar.envGrid[ssr[j].y][ssr[j].x] == Astar.BLOCKED){
							ssr[j].isblocked = true;							
						}
					}
					//if the path is blocked then set the current location as previous cell
					// to the blocked cell and start a new A* search
					if(currentPath[i + 1].isblocked == true){						
						current = currentPath[i];
						currentPathIndex = i;
						agentReahcedGoal = false;
						break;
					}
					agentReahcedGoal = true;
					if(i != 0){
						Astar.showGrid[currentPath[i].y][currentPath[i].x] = '0';
					}
				}			
				//Agent reached goal search is successful 
				if(agentReahcedGoal){
					System.out.println("PATH FOUND");
					System.out.println("Total No. of nodes expanded " + expandedNodes);
					Astar.showGrid[Astar.sy][Astar.sx] = 's';
					Astar.printgrid(Astar.showGrid);
					showPath();
				}
				//Agent is still not able to reach the goal. Start a new A* search.
				else{
					Astar.showGrid[current.y][current.x] = '0';
					//Astar.printgrid(Astar.showGrid);
					//Update the h values in the case of adaptive A* search
					if(Astar.isAdaptive){
						for(int i = 0; i <= closeIndex; ++i ){
							closed[i].h = gstar - closed[i].g;
						}
					}
				}				
			}
			//There is no way to reach the goal. Agent give up.
			else{
				System.out.println("NO PATH POSSIBLE");
				System.out.println("Total No. of nodes expanded " + expandedNodes);
				Astar.showGrid[Astar.sy][Astar.sx] = 's';
				Astar.printgrid(Astar.showGrid);
				showPath();
				break;
			}
		}
	}
	/*
	 * This is the main A* search It expands the node in non-increasing order of 
	 * their f value using a binary heap implementation of min priority queue.
	 * It start it search from startState and find a possible path to goalState.
	 */
	boolean aStarSearch(int count, GridCell startState, GridCell goalState){
		priorityQueue.clear();
		boolean success = false;
		startState.g = 0;		
		priorityQueue.insert(startState);
		goalState.h = 0;
		//System.out.print("expands ");
		while(!priorityQueue.isEmpty()){
			//Extract the cell having minimum value of f. For tie breaking rules see
			//the comments for method compareTwoGridCell.
			GridCell min  = priorityQueue.extractMin();
			//set the current search id of the A* search.
			min.astarCount = count;
			closed[++closeIndex] = min;
			//System.out.print("((" + min.y+","+min.x +")("+ min.h+ "," +min.g+"))");
			expandedNodes++;			
			
			if(min.equals(goalState)){
				success = true;
				break;
			}
			//Get the successor of the just expanded cell.
			GridCell[] successor = getSuccessorAstar(min, goalState);
			for(int i=0; i<4; i++){
				if(successor[i] == null){
					break;
				}				
				int si = priorityQueue.find(successor[i]); 
				if(si != -1){
					//if successure is already in the heap then update its 
					//g value if it reduces.
					if(successor[i].g > (min.g + 1)){
						successor[i].parent = min;						
						priorityQueue.decreaseKey(si, min.g + 1);
					}
				}
				else{					
					//in case of backward seach we have to update the h value in every iteration
					//of A* search
					if(Astar.doBackwardSearch ){
						successor[i].h = manhattanDistance(goalState.x,goalState.y, successor[i].x, successor[i].y);
					}

					//update the parent and g value of the newly added cell and insert it into the heap.
					successor[i].parent = min;
					successor[i].g = min.g + 1;
					priorityQueue.insert(successor[i]);				
				}								
			}			
		}	
		//System.out.println();
		return success;
	}
	/*
	 * This method is used by the A* search. It will return the adjacent cells of an
	 * expanded cell.If the cell is not already there in the agent grid then it will
	 * create it on need basis and update it h value using Manhattan distance.
	 */
	GridCell[] getSuccessorAstar(GridCell gc, GridCell goalState){
		GridCell[] ret = new GridCell[4];
		int x = gc.x;
		int y = gc.y;
		int i = 0;
		GridCell temp = null;
		if(x+1 < Astar.w){	
			temp = agentgrid[y][x+1]; 
			if(temp == null){
				agentgrid[y][x+1] = new GridCell(x+1, y);
				temp = agentgrid[y][x+1];
				temp.h = manhattanDistance(goalState.x,goalState.y, temp.x, temp.y);
				ret[i++] = temp;				
			}
			else if(!temp.isblocked && (temp.astarCount < gc.astarCount)){
				ret[i++] = temp;							
			}
		}
		if(y-1 >= 0){
			temp = agentgrid[y-1][x]; 
			if(temp == null){
				agentgrid[y-1][x] = new GridCell(x, y-1);
				temp = agentgrid[y-1][x];
				temp.h = manhattanDistance(goalState.x,goalState.y, temp.x, temp.y);
				ret[i++] = temp;				
			}
			else if(!temp.isblocked && (temp.astarCount < gc.astarCount)){
				ret[i++] = temp;
			}
		}
		if(x-1 >= 0){
			temp = agentgrid[y][x-1]; 
			if(temp == null){
				agentgrid[y][x-1] = new GridCell(x-1, y);
				temp = agentgrid[y][x-1];
				temp.h = manhattanDistance(goalState.x,goalState.y, temp.x, temp.y);
				ret[i++] = temp;				
			}
			else if(!temp.isblocked && (temp.astarCount < gc.astarCount)){
				ret[i++] = temp;
			}
		}
		if(y+1 < Astar.h){
			temp = agentgrid[y+1][x]; 
			if(temp == null){
				agentgrid[y+1][x] = new GridCell(x, y+1);
				temp = agentgrid[y+1][x];
				temp.h = manhattanDistance(goalState.x,goalState.y, temp.x, temp.y);
				ret[i++] = temp;				
			}
			else if(!temp.isblocked && (temp.astarCount < gc.astarCount)){
				ret[i++] = temp;
			}
		}
		return ret;
	}
	//This method return the adajcent nodes while the agent is traversing the presumed path 
	//returned by A*.
	GridCell[] getSuccessor(GridCell gc){
		GridCell[] ret = new GridCell[4];
		int x = gc.x;
		int y = gc.y;
		int i = 0;
		GridCell temp = null;
		if(x+1 < Astar.w){	
			temp = agentgrid[y][x+1]; 
			if(!temp.isblocked){				
				ret[i++] = temp;
			}
		}
		if(y+1 < Astar.h){
			temp = agentgrid[y+1][x]; 
			if(!temp.isblocked){				
				ret[i++] = temp;
			}
		}
		if(x-1 >= 0){
			temp = agentgrid[y][x-1]; 
			if(!temp.isblocked){				
				ret[i++] = temp;
			}
		}
		if(y-1 >= 0){
			temp = agentgrid[y-1][x]; 
			if(!temp.isblocked){				
				ret[i++] = temp;
			}
		}
		return ret;
	}
	//This method prepare the presumed path in case of backward search.
	private void preparePathBackward() {
		GridCell gc = current;
		while(!gc.equals(goal)){
			currentPath[++currentPathIndex] = gc.parent;
			gc = gc.parent;
		}		
	}
	//This method prepare the presumed path in case of backward search.
	void preparePathForward(GridCell gc) {
		if(gc.equals(current)) {			
			return;
		}
		preparePathForward(gc.parent);
		currentPath[++currentPathIndex] = gc;
	}
	//This method print the agent path.
	private void showPath() {		
		System.out.println("Agent Path");
		for(int i = 0; i <= currentPathIndex; ++i){
			if(i != 0){
				System.out.print("->");
			}
			System.out.print("(" +  currentPath[i].y + "," +  currentPath[i].x + ")");
		}				
		System.out.println();
	}
	//This method calculate the Manhattan distance.
	int manhattanDistance(int gx, int gy, int x, int y){		
		int xval =  gx - x; 
		if(xval < 0) xval = -xval;
		int yval =  gy - y;
		if(yval < 0) yval = -yval;
		return (xval + yval);
	}
}