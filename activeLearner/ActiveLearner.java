
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.TreeMap;
import java.util.Random;

public class CoraController {
	public static ArrayList<Record> data = new ArrayList<Record>();
	
	public static ArrayList<MappedRecord> duplicate = new ArrayList<MappedRecord>();	
	public static ArrayList<MappedRecord> single = new ArrayList<MappedRecord>();
	
	public static ArrayList<MappedRecord> train = new ArrayList<MappedRecord>();
	public static ArrayList<MappedRecord> test = new ArrayList<MappedRecord>();
	
	public static final String testfilename = "coratest";
	public static final String trainfilename = "coratrain";
	public static final String restdatafilename = "cora.txt";
	
	public static Random rand = new Random();
	public static Mapper mapper;
	
	public static final String newline = System.getProperty("line.separator");
	
	public static final String restWekaPrefix = 
	"@relation \"cora1\"" + newline + newline
	+ "@attribute author NUMERIC" + newline
	+ "@attribute volume NUMERIC" + newline
	+ "@attribute title NUMERIC" + newline
	+ "@attribute institution NUMERIC" + newline
	+ "@attribute venue NUMERIC" + newline
	+ "@attribute address NUMERIC" + newline
	+ "@attribute publisher NUMERIC" + newline
	+ "@attribute year NUMERIC" + newline
	+ "@attribute pages NUMERIC" + newline
	+ "@attribute editor NUMERIC" + newline
	+ "@attribute note NUMERIC" + newline
	+ "@attribute month NUMERIC" + newline
		+	"@attribute class {\'0\',\'1\'}" + newline + newline + newline + newline
		+	"@data";
 	
	public static void readNodes(String file) throws IOException{		
		LineNumberReader rdr = new LineNumberReader(new FileReader(file));		
		String line = null;		
		
		while ((line = rdr.readLine()) != null) {
			if(!line.trim().equals("")){				
				String[] linedata = line.split("\",");				
				Record ent = new Record(); 
				String temp = linedata[linedata.length - 1].replaceAll("\\p{Punct}+", " ").trim();
				ent.id = temp.hashCode();
				for(int i = 0 ; i < linedata.length - 1; ++i){
					temp = linedata[i].replaceAll("\\p{Punct}+", " ").trim();
					ent.strmap.put(i + 1, temp);					
				}				
				data.add(ent);		
			}
		}
		rdr.close();
		
	}

	public static void map(){
		for(int i = 0 ; i  < data.size(); ++i){
			Record rec1 = data.get(i);
			for(int j = i + 1 ; j  < data.size(); ++j){
				Record rec2 = data.get(j);
	
				MappedRecord mrec = new MappedRecord(rec1.id, rec2.id);
				int n = rec1.strmap.size();
				
				for(int k = 0 ; k < n; ++k){					
					int index = k + 1;
					mrec.measuremap.put(index, mapper.getScore(rec1.strmap.get(index), rec2.strmap.get(index)));
				}
				if(mrec.isDuplicate) duplicate.add(mrec);
				else single.add(mrec);
				
				test.add( mrec);
			}
		}	
		System.out.println("CORA : Single " + single.size() + " duplicate " + duplicate.size());
	}
	public static void initialize(){
		int n = 2;
		int id = 0;
		for(int i = 0; i < n; ++i){
			id = rand.nextInt(duplicate.size());
			MappedRecord mrec = duplicate.remove(id);
			train.add( mrec);			
			test.remove(mrec);		
		}				
		for(int i = 0; i < n; ++i){
			id = rand.nextInt(single.size());
			MappedRecord mrec = single.remove(id);
			train.add( mrec);
			test.remove(mrec);			
		}
	}	
	public static void dumpInfileWeka(String file, ArrayList<MappedRecord> rcs, String prefix, int iteration) {
		try{
			PrintWriter out = new PrintWriter(new FileOutputStream(file + iteration + ".arff"));
			out.println(prefix);

			for(int i = 0; i  < rcs.size(); ++i){
				MappedRecord mrec = rcs.get(i);
				
				StringBuffer temp = new StringBuffer();								
				int n = mrec.measuremap.size();
				for(int k = 0 ; k < n; ++k){
					int index = k + 1;					
					temp.append(mrec.measuremap.get(index)).append(",");									
				}
				if(mrec.isDuplicate) temp.append("\'1\'");
				else temp.append("\'0\'");

				out.println(temp.toString());	
			}										
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public static void main(String[] args){
		try {
			mapper = new Mapper(1);
			Random randmain = new Random(443436);
			readNodes(restdatafilename);
			for(int k = 0 ; k  < 1; ++k){
				String trf = trainfilename + k;
				String tsf = testfilename + k;
				
				if(args[0].equals("random")){
					trf = trf + "_rand_";
					tsf = tsf + "_rand_";
				}
				else if(args[0].equals("dt")){
					trf = trf + "_dt_";
					tsf = tsf + "_dt_";			
				}
				else if(args[0].equals("svm")){
					trf = trf + "_svm_";
					tsf = tsf + "_svm_";			
				}				
				
				duplicate.clear();	
				single.clear();				
				train.clear();
				test.clear();
				
				rand.setSeed(randmain.nextLong());							
				map();
				initialize();		
				dumpInfileWeka(trf, train, restWekaPrefix, 0);
				dumpInfileWeka(tsf, test, restWekaPrefix, 0);

				for(int i = 0 ; i < 20; ++i){
					int rem = 0;
					if(args[0].equals("random")){
						rem = rand.nextInt(test.size());
					}
					else if(args[0].equals("dt")){
						WekaDecisionTree.doVotingCora(trf, tsf, i);
					
						if(!WekaDecisionTree.two.isEmpty()){
							rem = WekaDecisionTree.two.get(0);
						}
						else if(!WekaSVM.rest.isEmpty()){
							rem = WekaDecisionTree.rest.get(0);
						}
					}
					else if(args[0].equals("svm")){
						WekaSVM.doVotingCora(trf, tsf, i);
						
						if(!WekaSVM.two.isEmpty()){
							rem = WekaSVM.two.get(0);
						}
						else if(!WekaSVM.rest.isEmpty()){
							rem = WekaSVM.rest.get(0);
						}						
					}					
					MappedRecord mrec = test.remove(rem);
					train.add(mrec);
					if(args[0].equals("random")){
						if((i+1)%5 == 0){
							dumpInfileWeka(trf, train, restWekaPrefix, i + 1);
							dumpInfileWeka(tsf, test, restWekaPrefix, i + 1);					
						}
					}
					else{
						dumpInfileWeka(trf, train, restWekaPrefix, i + 1);
						dumpInfileWeka(tsf, test, restWekaPrefix, i + 1);
					}
				}
			}
		} catch (IOException e) {			
			e.printStackTrace();
		}
	}
	/*
	 * public static void dumpInfileSVMLight(String file, TreeMap<Integer, MappedRecord> rcs) {
		try{			
			PrintWriter out = new PrintWriter(new FileOutputStream(file));			
			for(Integer key: rcs.keySet()){
				StringBuffer temp = new StringBuffer();
				MappedRecord mrec = rcs.get(key);
				if(mrec.isDuplicate) temp.append("1");
				else temp.append("0");
				temp.append(" ");
				int n = mrec.measuremap.size();
				for(int k = 0 ; k < n; ++k){
					int index = k + 1;
					temp.append((index) + ":" + mrec.measuremap.get(index));
					if(k < (n-1)){
						temp.append(" ");
					}
				}				
				out.println(temp.toString());	
			}					
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}				
	}
	
	 * 
	 */
}
