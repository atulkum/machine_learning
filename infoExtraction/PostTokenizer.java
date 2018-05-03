import java.io.*;
import java.util.*;

public class PostTokenizer{
	static String FEATURE_STAR = "STAR";  
	static String FEATURE_HAVEDOT = "HAVEDOT";
	static String FEATURE_HAVESLASH = "HAVESLASH";
	static String FEATURE_HAVEDOLLAR = "HAVEDOLLAR";
	static String FEATURE_HAVEPARENTHESIS = "HAVEPARENTHESIS";
	static String FEATURE_HAVEPLUS = "HAVEPLUS";
	
	
	
	public static void main(String args[]) throws Exception
	{
		args = new String[2];
		args[0] = "posts2.txt";
		args[1] = "label2.txt";
		if( args == null || args.length != 2)
		{
			System.out.println("Usage PostTokenizer [postfile(input)] [tokenizedfile(output)]");
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
			String[] tokens = curLine.split("[ -]");
			for(int i=0;i<tokens.length;i++)
			{
				if(tokens[i] != null ||	tokens[i].trim().length() > 0)
				{
					out.print(tokens[i]);						
					out.println();
					out.flush();
				}				
			}	
			out.println(); // make a blank line between posts
		}
		lr.close();
		out.close();	 				
	}
	
}





