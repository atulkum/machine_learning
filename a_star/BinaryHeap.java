/*
 * This is the BinaryHeap class which implements the binary heap 
 * implementation of the  min priority queue, which is used during 
 * the A* search. The binaryheap contain the pointers to the 
 * GridCell objects in an array. The array store a complete 
 * binary tree where the parent and child index are given as below:
 * p=parent index
 * c= child index
 * then p=c/2
 * and c = p*2 and p*2 + 1.
 *
 * The size field store the maximum size upto which this heap 
 * can grows.
 * The current field store the current size of the heap.
 * The elements field is an array of GridCell object which stores 
 * the complete binary tree. The indexing of the elements start from
 * 1. So the root(or the minimum element) is placed at the index 1.     
 */
public class BinaryHeap {
	int size;
	int current;
	GridCell[] elements;
	//Construuctor for the BinaryHeap class which create the elements 
	//array to the size passed as maxElement
	public BinaryHeap(int maxElement){
		size = maxElement;
		elements = new GridCell[size + 1];
		current = 0;
	}
	void clear(){
		current = 0;
	}
	//This function return the root node (which contain the minimum element).
	GridCell getMin(){
		return elements[1];
	}
	//This function return true if the heap is empty.
	boolean isEmpty(){
		return current == 0;
	}
	//This function insert a new element in the binary heap.
	//It first copy the passed element at the "current + 1" th index
	//and then heapify the heap by bubbling up the element.It compares
	//the child element to it's parent and if the parent is larger than the 
	//child then both are swaped, until it reaches at the root node.
	boolean insert(GridCell gc){		
		if(find(gc) != -1){
			System.out.println("Duplicate gridcell at (" + gc.y +","+gc.x+")");
			return false;
		}
		current++;
		if(size == current){
			System.out.println("Heap is full");
			return false;
		}
		elements[current] = gc;
		
		int c = current;
		while(c != 1){
			int p = c/2;
			int cmp = compareTwoGridCell(elements[c], elements[p]); 
			if( cmp < 0){
				GridCell temp = elements[c];
				elements[c] = elements[p];
				elements[p] = temp;
			}
			else{
				break;
			}
			c = p;			
		}
		return true;
	}
	/*
	 * This method extract the minimum element from the heap.
	 * It first copy the last element to the root element and 
	 * then heapify the binary heap by going dowin down and comparing
	 * the parents to its smaller child and swapping if required,
	 * until it reaches a leave node.
	 */
	GridCell extractMin(){
		if(isEmpty()) {
			System.out.println("Heap is empty. extractMin fails.");
			return null;
		}
		GridCell ret = elements[1];
		elements[1] = elements[current];
		current--;
		int p = 1;		
		while(p < current){
			int c;
			int cmp;
			if(2*p > current){
				break;
			}
			else if(2*p+1 > current){
				c = 2*p;
			}
			else{
				cmp = compareTwoGridCell(elements[2*p], elements[2*p + 1]);
				if(cmp < 0){
					c = 2*p;
				}
				else{
					c = 2*p + 1;
				}
			}
			cmp = compareTwoGridCell(elements[c], elements[p]);
			
			if(cmp < 0){
				GridCell temp = elements[c];
				elements[c] = elements[p];
				elements[p] = temp;
			}
			else{
				break;
			}
			p = c;			
		}
		return ret;
	}
	/*
	 * This method decrease the value of a GridCell. It first decrease 
	 * the g value of the cell and then heapify the heap.
	 */
	boolean decreaseKey(/*GridCell gc*/int gci, int g){						
		//int gci = find(gc);
		if(gci == -1){
			System.out.println("Gridcell not found");
			return false;
		}		
		elements[gci].g = g;
		int c = gci;		
		while(c != 1){
			int p = c/2;
			int cmp = compareTwoGridCell(elements[c], elements[p]); 
			if( cmp < 0){
				GridCell temp = elements[c];
				elements[c] = elements[p];
				elements[p] = temp;
			}
			else{
				break;
			}
			c = p;			
		}
		return true;		
	}
	//This method find the existance of a cell in the binary heap by
	//comparing the passed cell to every cell in the elements array.
	int find(GridCell gc){
		for(int i = 1; i <= current; ++i){
			if(elements[i].equals(gc)){
				return i;
			}
		}
		return -1;
	}
	//This method print the whole heap data structure.
	void printHeap(){
		if(isEmpty()){
			System.out.println("Heap is empty");
		}
		for(int i = 1; i <= current; ++i){
			System.out.print("(" +"(" +  elements[i].y + "," +  elements[i].x + ")( " + elements[i].g+ " ," + elements[i].h);
			if(elements[i].isblocked == true){
				System.out.println("),blocked)");
			}
			else{
				System.out.print(",unblocked)");
			}
		}
		System.out.println();
		System.out.println("+++++++++++++++++++++++++++++++++++++++++++++++++++++");
	}
	/*
	 * This method compare two grid cells f value. If f values are equal then tie is broken 
	 * according to the larger g value or smaller g value as indicated by the user input.
	 * if g values are also equal then the tie is broken in this order of the cells, 
	 * right, down, left and up respectively.
	 */
	static int compareTwoGridCell(GridCell grid1,GridCell grid2) {
		int f1 = grid1.g + grid1.h;
		int f2 = grid2.g + grid2.h;
		if(f1 != f2){
			return (f1-f2);
		}
		else {
			//larger g or smaller g
			if(grid1.g !=  grid2.g){
				if(Astar.largerG == true){
					return (grid2.g -  grid1.g);
				}
				else{
					return (grid1.g -  grid2.g);
				}
			}
			else {
				if(grid1.x != grid2.x){
					return grid2.x - grid1.x;
				}
				else{
					return grid2.y - grid1.y;
				}				
			}
		}
	}

}
