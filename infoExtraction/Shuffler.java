import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;


public class Shuffler {
	public static void main(String[] args){
		ArrayList<String> posts = new ArrayList<String>();
		try {
			BufferedReader rdr = new BufferedReader(new FileReader("posts.txt"));
			String line = null;
			while ((line = rdr.readLine()) != null) {
				posts.add(line);
			}
			rdr.close();
			Collections.shuffle(posts);
			PrintWriter out1 = new PrintWriter(new FileWriter("posts1.txt"));
			PrintWriter out2 = new PrintWriter(new FileWriter("posts2.txt"));
			int n = posts.size()/2;
			int i = 0;
    		for(String ln : posts){    			
    			if(i < n){
    				out1.println(ln);
    			}
    			else{
    				out2.println(ln);
    			}
    			++i;
    		}
    		out1.close();
    		out2.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
