import java.util.HashMap;


public class MappedRecord {
	int id;
	public MappedRecord(int id1, int id2){
		measuremap = new HashMap<Integer, Double>();
		this.id1 = id1;
		this.id2 = id2;
		
		isDuplicate = (id1 == id2);
		
		id = Math.min(id1, id2) + 37*Math.max(id1, id2);
	}
	boolean isDuplicate;
	int id1;
	int id2;
	HashMap<Integer, Double> measuremap;
}
