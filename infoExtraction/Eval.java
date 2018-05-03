import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

public class Eval {	
	public static void main(String[] args){
		/*args = new String[]{				
				"C:/uscProjects/548/homework/output.txt",
				"89"		
		};			
		int[] ct = new int[2];
		ct[0] = ct[1] = 0;
		try {
			readFile(args[0], ct);		
		} catch (IOException e) {
			e.printStackTrace();
		}
		int gt = Integer.parseInt(args[1].trim());
		
		double TP = ct[0];		
		System.out.println(TP + " "  + ct[1]);
		
		double precion = TP/ct[1];
		double recall = TP/gt;
				
		double F_Measure = (2*precion*recall)/(precion + recall);
		System.out.println(Math.round(F_Measure*100) + "%");*/
		double ccratio = 0;//.001;
		if (ccratio > 0){
			System.out.println(ccratio);
		}
	}

	static void readFile(String file, int[] count) throws IOException {
		LineNumberReader rdr = new LineNumberReader(new FileReader(file));
		String line = null;
		while((line = rdr.readLine()) != null) {
			count[1]++;						
			String[] val = line.split(" ");			
			if(Integer.parseInt(val[0].trim()) == Integer.parseInt(val[1].trim())){
				count[0]++;
			}
			else{
				System.out.println(line);
			}
		}		
	}
}
