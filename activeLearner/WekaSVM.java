import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;

import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class WekaSVM {
	public static HashMap<Integer, Integer> miss = new HashMap<Integer, Integer>();
	public static ArrayList<Integer> two = new ArrayList<Integer>();
	public static ArrayList<Integer> rest = new ArrayList<Integer>();
	
	public static void classifyInstancesTree(String rem, String trainfile, String testfile) throws Exception{
		Instances train = new Instances(new BufferedReader(new FileReader(trainfile)));
		Instances test = new Instances(new BufferedReader(new FileReader(testfile)));
		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);

		Remove rm = new Remove();
		rm.setAttributeIndices(rem);  //"1" = remove 1st attribute

		SMO lib = new SMO();		     

		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(lib);
		
		fc.buildClassifier(train);		
		 
		for (int i = 0; i < test.numInstances(); i++) {
			double pred = fc.classifyInstance(test.instance(i));
			if((int) test.instance(i).classValue() !=  (int) pred){
				Integer label = miss.get(i);
				if(label == null) miss.put(i, 1);
				else miss.put(i, label + 1);							
			}
		}					
	}

	public static void doVoting(String trainFile, String testFile, int iteration){
		two.clear();
		rest.clear();
		try {			
			WekaDecisionTree.classifyInstancesTree("1,2,3", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			WekaDecisionTree.classifyInstancesTree("1,2,4", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			WekaDecisionTree.classifyInstancesTree("1,3,4", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			WekaDecisionTree.classifyInstancesTree("2,3,4", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			//PrintWriter out = new PrintWriter(new FileOutputStream("j48output.txt"));
			//out.println(miss);
			//out.close();
			
			for(Integer lineno: miss.keySet()){
				if(miss.get(lineno) == 2){
					two.add(lineno);
				}
				else{
					rest.add(lineno);
				}
			}
			
		} catch (Exception e) {			
			e.printStackTrace();
		}
	}
	public static void doVotingCora(String trainFile, String testFile, int iteration){
		two.clear();
		rest.clear();
		try {			
			WekaDecisionTree.classifyInstancesTree("1,2,3,5,6,7,8,9,10,11,12", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			WekaDecisionTree.classifyInstancesTree("1,2,4,5,6,7,8,9,10,11,12", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			WekaDecisionTree.classifyInstancesTree("1,3,4,5,6,7,8,9,10,11,12", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			WekaDecisionTree.classifyInstancesTree("2,3,4,5,6,7,8,9,10,11,12", 
					trainFile + iteration + ".arff", testFile + iteration + ".arff");
			//PrintWriter out = new PrintWriter(new FileOutputStream("j48output.txt"));
			//out.println(miss);
			//out.close();
			
			for(Integer lineno: miss.keySet()){
				if(miss.get(lineno) == 2){
					two.add(lineno);
				}
				else{
					rest.add(lineno);
				}
			}
			
		} catch (Exception e) {			
			e.printStackTrace();
		}
	}
	public static void main(String[] args){
		doVoting("train", "test", 0);
	}
}