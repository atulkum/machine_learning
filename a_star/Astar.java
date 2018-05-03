import java.io.*;
/*
 * This is the driver class for this maze program. It reads the input grid 
 * from the file and fill the envGrid data and it also update various flags. 
 * After that it call the search method.
 * largerG field indicate that tie is broken in favour of larger g or smaller g.
 * doBackwardSearch field indicete thta the search is forward or backward.
 * isAdaptive firled indicate is the search is Adaptive A* or not.
 * envGrid is the environmental grid.
 * showGrid stores the agent path information.
 * h, w are height and width of the grid respectively.
 * sx, sy, gx, gy are the x and y co-ordinates of the goal and start cells.
 *  
 */
public class Astar {
	public static boolean largerG = true;
	public static boolean doBackwardSearch = false;
	public static boolean isAdaptive = false;
	public static String filename = null;
	//"C:\\uscProjects\\HW2561\\grids\\6.txt";
	public static int[][] envGrid;
	public static char[][] showGrid;
	public static int h, w;
	public static int sx, sy, gx, gy;
	public static final int BLOCKED = 4;
	public static final int OPEN = 3;
	public static final int GOAL= 2;
	public static final int START= 1;
	/*
	 * Main method whoch parse the argumnets, set the various flag accordingly,
	 * Read the input grid file and clal the serach method.
	 */
	public static void main(String[] args){
		if(args.length < 6){
			System.out.println("Use: java Astar -i inputfile -d {f|b} -t {s|l} [-A]");
			return;
		}
		for(int i=0; i < args.length; ++i){
			char[] arg = args[i].toCharArray();
			if (arg[0] == '-') {
				switch (arg[1]) {  
					case 'i': filename = args[++i];
	                  break;
					case 'd': 
						{
							char[] nextArg = args[++i].toCharArray();
							if(nextArg[0] == 'b'){
								doBackwardSearch=true;
							}
						}
	                  break;
					case 't':
						{
							char[] nextArg = args[++i].toCharArray();
							if(nextArg[0] == 's'){							
								largerG=false;
							}
						}
	                  break;
					case 'A': isAdaptive = true;
	                  break;

					default : System.out.println("Unaccepted Parameter " + arg[1]);
	                  break;
				}
			}
		}
		readGridFile(filename);
		//printgrid(envGrid);		
		SearchAgent agent = new SearchAgent();
		agent.doSearch();		
	}
	//Method to prinf the grid.
	static void printgrid(int[][] graph) {
		for(int y = 0; y < h; ++y){
			for(int x = 0; x < w; ++x){
				System.out.print(graph[y][x] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	//Method to prinf the grid.
	static void printgrid(char[][] graph) {
		for(int y = 0; y < h; ++y){
			for(int x = 0; x < w; ++x){
				System.out.print(graph[y][x] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	// This method read the grid file and fill up the information in envGrid.
	static void readGridFile(String file) {
		BufferedReader reader=null;
		try {
			reader =  new BufferedReader(new FileReader(file));
		    String line = reader.readLine();
		    if(line!= null){
		    	line = line.trim();
		    	h = Integer.parseInt(line);
		    }
		    line = reader.readLine();
		    if(line!= null){
		    	line = line.trim();
		    	w = Integer.parseInt(line);
		    }
		    int i = 0;
		    envGrid = new int[h][w];
		    showGrid = new char[h][w];
		    while (( line = reader.readLine()) != null){
		    	line  = line.trim();
		    	if(line.equals("")){
		    		continue;
		    	}
		    	//String[] row = line.split(" ");
		    	char[] row = line.toCharArray();
				for(int j = 0; j < w; ++j){					
					switch(row[2*j]){
						case '_': {
							envGrid[i][j] = OPEN;
							showGrid[i][j] = '_';
						}
						break;
						case 'x':{
							envGrid[i][j] = BLOCKED;
							showGrid[i][j] = 'x';
						}
						break;
						case 'g': {
							envGrid[i][j] = GOAL;
							gx = j;
							gy = i;
							showGrid[i][j] = 'g';
						}
						break;
						case 's': {
							envGrid[i][j] = START;
							sx = j;
							sy = i;
							showGrid[i][j] = 's';
						}
						break;
					}
				}
				i++;
		    }
		}
		catch(IOException e){
			e.printStackTrace();
		}
		finally {
			try {
				if(reader != null){ 
					reader.close();
				}
			} catch (IOException e) {				
				e.printStackTrace();
			}
		}		    
	}
}
