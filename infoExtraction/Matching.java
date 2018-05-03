import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import uk.ac.shef.wit.simmetrics.similaritymetrics.Levenshtein;
import uk.ac.shef.wit.simmetrics.similaritymetrics.QGramsDistance;
import uk.ac.shef.wit.simmetrics.similaritymetrics.Soundex;

public class Matching { 
	public static EntryList source1;
	public static EntryList source2;
	public static void main(String[] args){
		////////////////////////////////////////
		args = new String[]{
			
			"C:/uscProjects/548/hw6/hw6_secret_set/re1.txt",
			"C:/uscProjects/548/hw6/hw6_secret_set/re2.txt",		
			"re3.txt"
		};
		///////////////////////////////////////		
		ArrayList<OutEntry> output = new ArrayList<OutEntry>();
		try {
			source1 = readNodes(args[0]);
			source2 = readNodes(args[1]);
			
			for(Entry e1 : source1.tree.values()){
				for(Entry e2 : source2.tree.values()){
					if(match(e1,e2)){
						output.add(makeOutput(e1, e2));
					}
				}
			}		
			PrintWriter out = new PrintWriter(new FileOutputStream(args[2]));			
			for(OutEntry oe: output){
				oe.print(out);
			}
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	public static boolean match(Entry e1, Entry e2) {
		if(e1.plabel.equals("") && e2.plabel.equals("")){
			return false;
		}	
		if(matchString(e1.plabel,e2.plabel, true)){
			return true;
		}
		else if(matchString(e1.pname + " " + e1.plabel, e2.pname + " " + e2.plabel, false)){
			return true;
		}
		else{
			Entry ep1 = source1.tree.get(e1.root);
			Entry ep2 = source2.tree.get(e2.root);
			if(ep1 != null && ep2 != null && !ep1.plabel.trim().equals("") 
				&& !ep2.plabel.trim().equals("") 
				&& matchString(ep1.plabel, ep2.plabel, true)){
				
				double measure = new QGramsDistance().getSimilarity(
						ep1.plabel + " " + e1.plabel, ep2.plabel + " " + e2.plabel);			
				if(measure > 0.78){
					return true;
				}								
			}		
		}
		String inst1 = e1.instances.toString().trim();
		String inst2 = e2.instances.toString().trim();
		if((inst1.contains("california") || inst1.contains("ca")) && (inst1.contains("california") || inst1.contains("ca"))){
			return true;
		}
		if(!inst1.equals("") && !inst2.equals("") && matchString(inst1, inst2, true)){
			return true;
		}
		return false;		
	}

	public static boolean matchString(String src1, String src2, boolean isaffix) {		
		//System.out.println("Match " + src1 + " " + src2);
		if(src1.equals("") || src2.equals("")){
			return false;
		}	
		double measure = 0;		
		if(src1.toString().equalsIgnoreCase(src2.toString())){
			return true;
		}
		if(isaffix && (src1.toString().startsWith(src2.toString()) || src2.toString().startsWith(src1.toString())
			||src1.toString().endsWith(src2.toString()) || src2.toString().endsWith(src1.toString()) 
			||src1.toString().contains(src2.toString()) || src2.toString().contains(src1.toString()))){
			//measure = new QGramsDistance().getSimilarity(src1.toString(), src2.toString());
			//if(measure > 0.78){
				return true;
			//}			 
		}								
		int labelsizesum = src1.length() + src2.length();
		if(labelsizesum <= 8){
			//levemstein			
			measure = new Levenshtein().getSimilarity(src1.toString(), src2.toString());			
			if(measure > 0.9){
				return true;
			}
			else{
				//synonym
				measure = new Soundex().getSimilarity(src1.toString(), src2.toString());				
				if(measure > 0.9){
					return true;
				}					
			}
		}			
		else{
			//3-gram
			measure = new QGramsDistance().getSimilarity(src1.toString(), src2.toString());
			if(measure > 0.78){
				return true;
			}
			else{
				//synonym				
				measure = new Soundex().getSimilarity(src1.toString(), src2.toString());				
				if(measure > 0.9){
					return true;
				}					
			}
		}		
		return false;
	}
	public static OutEntry makeOutput(Entry e1, Entry e2){
		OutEntry oe = new OutEntry();
		oe.source1id = e1.nodeid;
		oe.src1label = e1.label;
		oe.source2id = e2.nodeid;
		oe.src2label = e2.label;
		return oe;
	}
	public static EntryList readNodes(String file) throws IOException{
		HashMap<Integer, Entry> tree = new HashMap<Integer, Entry>();
		LineNumberReader rdr = new LineNumberReader(new FileReader(file));		
		String line = null;
		int nnodes = 0;
		int root = 0;
		int count = 0;
		boolean isStart = false;
		boolean isStruct = false;
		
		while ((line = rdr.readLine()) != null) {
			if(line.trim().equals("# number of nodes")){
				line = rdr.readLine();
				nnodes = Integer.parseInt(line.trim());
			}
			else if(line.trim().equals("# root")){
				line = rdr.readLine();
				root = Integer.parseInt(line.trim());
			}
			else if(line.trim().equals("# left-right order should be retained")){				
				line = rdr.readLine();
				isStart = true;
			}
			else if(isStart && (count < nnodes)){
				if(!line.trim().equals("")){
					//System.out.println(line);
					String[] linedata = line.replaceAll("\"", " ").split(",");
					Entry ent = new Entry(); 
					for(int i = 0; i < linedata.length; ++i){			
						ent.setVal(i, linedata[i].trim()); 				
					}
					tree.put(ent.nodeid, ent);
					count++;
				}
			}
			else if(line.trim().equals("# (level by level top-down, start from root)")){
				line = rdr.readLine();
				isStruct = true;		
			}
			else if(isStruct && !line.trim().equals("# end")){
				//System.out.println(line);
				String[] temp1 = line.trim().split("-");
				int r = Integer.parseInt(temp1[0]);
				String[] temp2 = temp1[1].trim().split(",");
				
				for(String each : temp2){
					tree.get(Integer.parseInt(each)).root = r;
				}
			}
			else{
				isStart = false;
				isStruct = false;
			}
		}
		rdr.close();
		EntryList el = new EntryList(tree,nnodes, root);		
		return el;
	}
}
class EntryList{
	public HashMap<Integer, Entry> tree;
	public int nnodes;
	public int root;
	public EntryList(HashMap<Integer, Entry> tree, int nnodes, int root) {
		super();
		this.tree = tree;
		this.nnodes = nnodes;
		this.root = root;
	}
}
class Entry{
	public int root;
	public int nodeid;
	public String name;
	public String label;
	public StringBuffer instances = new StringBuffer();
	
	public String pname;
	public String plabel;
	
	public ArrayList<String> instance =  new ArrayList<String>();
	
	public void setVal(int i, String val){
		switch(i){
		case 0: nodeid = Integer.parseInt(val);break;
		case 1: name = val; 
			pname = val.toLowerCase();
		break;
		case 2: label = val; 
			plabel = val.toLowerCase();
			plabel = plabel.replace("(s)", "s");
			plabel = plabel.replace(":", "");
		break;
		default:{
			instance.add(val.toLowerCase());
			instances.append(val.toLowerCase() + " ");
		}
		break;
		}
	}
	public void print(){
		System.out.println(root + "," + nodeid + "," + name + "," + label);
	}
	/*public String[] stem(String ip){
		SnowballStemmer stemmer = new englishStemmer();
		String[] ts = ip.split(" ");
		for(int i = 0 ; i  < ts.length; ++i){			
			if (ts[i].length() > 0) {
				stemmer.setCurrent(ts[i].toString());				
				stemmer.stem();				
				ts[i] = stemmer.getCurrent();
			}
		}
		return ts;
	}*/	
}

class OutEntry{
	public String src1label;
	public String src2label;
	
	public int source1id;
	public int source2id;
		
	public void print(PrintWriter out){
		//System.
		out.println(src1label + "," + src2label + "," + source1id + "," + source2id);
	}
}


//public static boolean matchString(String[] s1, String[] s2) {
////System.out.println("Match " +s1 + " | |" + s2 );
//if(s1[0].equals("") || s2[0].equals("")){
//	return false;
//}	
//double measure = 0;
//
//StringBuffer sb1 = new StringBuffer();
//StringBuffer sb2 = new StringBuffer();
//
//for(String s : s1){
//	sb1.append(s + " ");
//}
//String src1 = sb1.toString().trim();
//for(String s : s2){
//	sb2.append(s + " ");
//}
//String src2 = sb1.toString().trim();
//
//if(src1.toString().equalsIgnoreCase(src2.toString())){
//	return true;
//}
//if(src1.toString().startsWith(src2.toString()) || src2.toString().startsWith(src1.toString())
//		||src1.toString().endsWith(src2.toString()) || src2.toString().endsWith(src1.toString()) 
//		||src1.toString().contains(src2.toString()) || src2.toString().contains(src1.toString())){
//	measure = new QGramsDistance().getSimilarity(src1.toString(), src2.toString());
//	if(measure > 0.58){
//		return true;
//	}			 
// }		
//				
//int labelsizesum = src1.length() + src2.length();
//if(labelsizesum <= 8){
//	//levemstein
//	/*measure = 0;
//	for(String s: s1){
//		double max = -1;	
//		for(String ss: s2){					
//			double temp = new Levenshtein().getSimilarity(s, ss);
//			if(temp > max){
//				max = temp;
//			}
//		}
//		measure += max;
//	}
//	measure /= s1.length;*/
//	
//	measure = new Levenshtein().getSimilarity(src1.toString(), src2.toString());
//	
//	if(measure >= 0.1){
//		return true;
//	}
//	else{
//		//synonym
//		/*measure = 0;
//		for(String s: s1){
//			double max = -1;	
//			for(String ss: s2){					
//				double temp = new Soundex().getSimilarity(s, ss);
//				if(temp > max){
//					max = temp;
//				}
//			}
//			measure += max;
//		}
//		measure /= s1.length;*/
//		measure = new Soundex().getSimilarity(src1.toString(), src2.toString());				
//		if(measure > 0.9){
//			return true;
//		}					
//	}
//}			
//else{
//	//3-gram
//	measure = new QGramsDistance().getSimilarity(src1.toString(), src2.toString());
//	if(measure > 0.78){
//		return true;
//	}
//	else{
//		//synonym
//		/*measure = 0;
//		for(String s: s1){
//			double max = -1;	
//			for(String ss: s2){					
//				double temp = new Soundex().getSimilarity(s, ss);
//				if(temp > max){
//					max = temp;
//				}
//			}
//			measure += max;
//		}
//		measure /= s1.length;*/
//		
//		measure = new Soundex().getSimilarity(src1.toString(), src2.toString());
//		
//		if(measure > 0.9){
//			return true;
//		}					
//	}
//}		
//return false;
//}