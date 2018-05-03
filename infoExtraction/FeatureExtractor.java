import java.io.*;
import java.util.*;


public class FeatureExtractor{
      
	/**
	* @param token A token to extract its useful features
	* return a vector of features associated with the input token
	*
	**/	
	static String FEATURE_STAR = "HAVE_STAR";  
	static String FEATURE_HAVEDOT = "HAVE_DOT";
	static String FEATURE_HAVESLASH = "HAVE_SLASH";
	static String FEATURE_HAVEDOLLAR = "HAVE_DOLLAR";
	static String FEATURE_HAVEPARENTHESIS = "HAVE_PARENTHESIS";
	static String FEATURE_HAVEPLUS = "HAVE_PLUS";
	static String FEATURE_HAVECAPS = "HAVE_CAPS";
	static String FEATURE_HAVENUM = "HAVE_NUMS";
	//static String FEATURE_HAVECOMMA = "HAVE_COMMA";
	
	public static Vector<String> extractTokenFeatures(String token)
	{
			Vector<String> associatedFeatures = new Vector<String>();
			// Extract some features you think that might help improve CRF model performance.
			// You may use regular expression to detect a pattern of an input token.
			// After feature extraction complete, please also put the names of all 
			// associated features to the vector associatedFeatures.
			// e.g. if( a present token has ".") associatedFeatures.addElement("DOTFEATURE");
			
			if(token.length() > 0){
				char st = token.charAt(0); 
				if( Character.isUpperCase(st) ){
					associatedFeatures.add(FEATURE_HAVECAPS);
				}
				if(Character.isDigit(st)){
					associatedFeatures.add(FEATURE_HAVENUM);
				}
			}		
			if(token.indexOf("*") != -1){
				associatedFeatures.add(FEATURE_STAR);
			}						
			if(token.indexOf("/") != -1){
				associatedFeatures.add(FEATURE_HAVESLASH);
			}
			if(token.indexOf("$") != -1){
				associatedFeatures.add(FEATURE_HAVEDOLLAR);
			}
			if(token.indexOf(".") != -1){
				associatedFeatures.add(FEATURE_HAVEDOT);
			}	
			if(token.indexOf("(") != -1 || token.indexOf(")") != -1){
				associatedFeatures.add(FEATURE_HAVEPARENTHESIS);
			}
			if(token.indexOf("+") != -1){
				associatedFeatures.add(FEATURE_HAVEPLUS);
			}
			//if(token.indexOf(",") != -1){
				//associatedFeatures.add(FEATURE_HAVECOMMA);
			//}
			return associatedFeatures;
	}
	
	
	public static void main(String args[]) throws Exception
	{			
		if( args == null || args.length != 2)
		{
			System.out.println("Usage FeatureExtractor [labelfile (input)] [label with token features file (output)]");
			return;
		}
		LineNumberReader lr = null;
		try
		{
			 lr = new LineNumberReader(new FileReader(args[0]));			
		}catch(Exception e)
		{
			System.out.println("Cannot open the file '"+args[0]+"'");
			return;
		}
		
		PrintWriter out = new PrintWriter(new FileOutputStream(args[1]));
		String curLine;
		while ((curLine = lr.readLine()) != null) {
			String[] tokens = curLine.split("\\s");
			if(tokens.length == 1 && tokens[0].trim().length() == 0) {
				out.println();
			}
			else
			{				
				out.print(tokens[0]);
				Vector tokenfeatures = extractTokenFeatures(tokens[0]);	
				for(int i=0;i<tokenfeatures.size();i++)
				{
					out.print(" "+tokenfeatures.elementAt(i));
				}
				if(tokens.length > 1) out.print(" "+tokens[tokens.length-1]); // put token's label back
				out.println();
			}						 			
		}
		lr.close();
		out.close();	 				
	}
}





