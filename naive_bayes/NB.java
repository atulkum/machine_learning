import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class NB {
	public static double epsilon = 1e-10;
	
	private Map<String, Map<String, Double>> classTermFrequency = new HashMap<String, Map<String, Double>>();
	private Map<String, Double> prior = new HashMap<String, Double>();
	private Set<String> vocab = new HashSet<String>();
	
	public static void main(String[] args) throws IOException{
		Map<String, List<String>> classDocMap = new HashMap<String, List<String>>();
		
		BufferedReader br = new BufferedReader(new FileReader(new File("/Users/atulk/Documents/workspace/NB/data/trainingdata.txt")));
		String line = br.readLine();
		while ((line = br.readLine()) != null) {
			int i = line.indexOf(" ");
			String classId = line.substring(0, i);
			String doc = line.substring(i+1, line.length() - i-2);
			if(!classDocMap.containsKey(classId)){
				classDocMap.put(classId, new ArrayList<String>());
			}
			classDocMap.get(classId).add(doc);
		}
		br.close();
		NB nbClassifier = new NB();
		/*nbClassifier.trainMultinomialNB(classDocMap);
		System.out.println(nbClassifier.predictMultinomialNB("This is a document"));
		System.out.println(nbClassifier.predictMultinomialNB("this is another document"));
		System.out.println(nbClassifier.predictMultinomialNB("documents are seperated by newlines"));*/
		
		nbClassifier.trainBernoulliNB(classDocMap);
		System.out.println(nbClassifier.classTermFrequency);
		
		System.out.println(nbClassifier.predictBernoulliNB("This is a document"));
		System.out.println(nbClassifier.predictBernoulliNB("this is another document"));
		System.out.println(nbClassifier.predictBernoulliNB("documents are seperated by newlines"));
		
		System.out.println(nbClassifier.predictBernoulliNB("document"));
		System.out.println(nbClassifier.predictBernoulliNB("another document"));
		System.out.println(nbClassifier.predictBernoulliNB("documents seperated newlines"));
		
		
		/*classDocMap.put("1", new ArrayList<String>());
		classDocMap.put("2", new ArrayList<String>());
		classDocMap.get("1").add("Chinese Beijing Chinese");	 
		classDocMap.get("1").add("Chinese Chinese Shanghai");	 
		classDocMap.get("1").add("Chinese Macao"); 
		classDocMap.get("2").add("Tokyo Japan Chinese");
		
		NB nbClassifier = new NB();
		nbClassifier.trainMultinomialNB(classDocMap);
		System.out.println(nbClassifier.predictMultinomialNB("Chinese Chinese Chinese Tokyo Japan"));
		nbClassifier.trainBernoulliNB(classDocMap);
		System.out.println(nbClassifier.predictBernoulliNB("Chinese Chinese Chinese Tokyo Japan"));*/
	} 
	
	public void trainMultinomialNB(Map<String, List<String>> classDocMap){
		Set<String> classes = classDocMap.keySet();
		int n = classDocMap.size();
		
		double normalizationFactor = 0.0;
		for(String c: classes){ 
			for(String terms:classDocMap.get(c)){
				String[] t = terms.split("\\s+");
				vocab.addAll(Arrays.asList(t));
				normalizationFactor += t.length;
			}
		}
		normalizationFactor += vocab.size();
		
		for(String c : classes){
			List<String> classDoc = classDocMap.get(c);
			double nc = classDoc.size();
			prior.put(c,  Math.log(nc/n));
			Map<String, Double> tf = new HashMap<String, Double>();
			for(String terms: classDoc){
				for(String term: terms.split("\\s+")){
					if(!tf.containsKey(term)){
						tf.put(term, 0.0);
					}
					tf.put(term, tf.get(term) + 1);
				}
			}
			for(String t: vocab){
				double nct = 0.0;
				if(tf.containsKey(t)){
					nct = tf.get(t);
				}
				if(!classTermFrequency.containsKey(c)){
					classTermFrequency.put(c, new HashMap<String, Double>());
				}
				classTermFrequency.get(c).put(t, (nct + 1.0)/normalizationFactor);
			}
		}
	}
	public String predictMultinomialNB(String doc){
		Map<String, Double> scores = new HashMap<String, Double>();
		Set<String> classes = classTermFrequency.keySet();
		String[] tokens = doc.split("\\s+");
		for(String c: classes ){
			double score =prior.get(c);
			for(String t:tokens){
				if(vocab.contains(t)){
					score += Math.log(classTermFrequency.get(c).get(t));
				}
			}
			scores.put(c, score);
		}
		Double max = Collections.max(scores.values());
		String ret = null;
		for(String c : scores.keySet()){
			if(Math.abs(scores.get(c) - max) <= epsilon){
				ret = c;
				break;
			}
		}
		return ret;
	}
	public void trainBernoulliNB(Map<String, List<String>> classDocMap){
		Set<String> classes = classDocMap.keySet();
		int n = classDocMap.size();
		for(String c: classes){ 
			for(String terms:classDocMap.get(c)){
				String[] t = terms.split("\\s+");
				vocab.addAll(Arrays.asList(t));
			}
		}
		
		for(String c : classes){
			List<String> classDoc = classDocMap.get(c);
			double nc = classDoc.size();
			prior.put(c,  Math.log(nc/n));
			Map<String, Double> tf = new HashMap<String, Double>();
			for(String terms: classDoc){
				Set<String> allTerm = new HashSet<String>(Arrays.asList(terms.split("\\s+")));
				for(String t: allTerm){
					if(!tf.containsKey(t)){
						tf.put(t, 0.0);
					}
					tf.put(t, tf.get(t) + 1);
				}
			}
			for(String t: vocab){
				double nct = 0.0;
				if(tf.containsKey(t)){
					nct = tf.get(t);
				}
				if(!classTermFrequency.containsKey(c)){
					classTermFrequency.put(c, new HashMap<String, Double>());
				}
				classTermFrequency.get(c).put(t, (nct + 1.0)/(nc + 2));
			}
		}
	}
	public String predictBernoulliNB(String doc){
		Map<String, Double> scores = new HashMap<String, Double>();
		Set<String> classes = classTermFrequency.keySet();
		
		Set<String> tokens = new HashSet<String>(Arrays.asList(doc.split("\\s+")));
		
		for(String c: classes ){
			double score =prior.get(c);
			for(String t:vocab){
				if(tokens.contains(t)){
					score += Math.log(classTermFrequency.get(c).get(t));
				}else{
					score += Math.log(1 - classTermFrequency.get(c).get(t));
				}
			}
			scores.put(c, score);
		}
		System.out.println(scores);
		Double max = Collections.max(scores.values());
		String ret = null;
		for(String c : scores.keySet()){
			if(Math.abs(scores.get(c) - max) <= epsilon){
				ret = c;
				break;
			}
		}
		return ret;
	}
}
