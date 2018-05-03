/*
 * This is the GridCell class which represent a cell into the grid.
 * The x and y represent x and y co-ordinate respectively. The origin is the 
 * upper left corner.The value of y are from 0 to (height-1) and it goes from 
 * top to bottom. The x values are from 0 to (width-1) and it grows from left 
 * to right. 
 * g represent the g value, i.e. the distance from the start node in
 * an A* search.
 * h is the hueristic value used during A* search. It is calculated using 
 * Manhattan's distance in A* search and using (g*-g) in the case of 
 * Adaptive A* search(after the first A* search).
 * The parent field is the parent to this grid cell.The parent to the start cell 
 * in an A* search is null.
 * astarCount field is the A* search id of an A* search.   
 * 
 */
public class GridCell {
	int x, y;
	int g;
	int h;
	GridCell parent;
	boolean isblocked;
	int astarCount;
	//Constructor to create an GridCell object
	public GridCell(int x, int y){
		this.x = x;
		this.y = y; 		
		isblocked=false;
		parent=null;		
		astarCount = 0;
	}
	//equals method for the GridCell object.
	//This method check that two grid cells are same or not
	//according to their x and y co-ordinates.
	public boolean equals(Object obj) {
		GridCell gc = (GridCell)obj;
		return ((this.x == gc.x) && (this.y == gc.y));		
	}
}
