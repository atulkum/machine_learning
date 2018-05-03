import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.io.PrintWriter;

public class GenerateTable {	
	public static final String[] labels = {"HOTELNAME", "LOCALAREA", "STARS", "PRICE", "DATE"};
	public static void main(String[] args){		
		if(args.length != 4){
			System.out.println("Usage: java GenerateTable DataFile1.txt label2.txt DataFile2.txt label1.txt");
			System.exit(0);
		}					    
	    try {
	    	StringBuffer sb = new StringBuffer();
	        sb.append("<html>\n<body>\n");sb.append("<title>homework2</title>\n");	 	    	 	    
			sb.append("<table border=\"1\">\n");
			sb.append("<tr bgcolor=#00FFFF>\n");		
			sb.append("<th>HOTELNAME</th>\n");
			sb.append("<th>LOCALAREA</th>\n");
			sb.append("<th>STARS</th>\n");
			sb.append("<th>PRICE</th>\n");
			sb.append("<th>START_DATE</th>\n");
			sb.append("<th>END_DATE</th>\n");			
			sb.append("</tr>\n");
	    	
			processData(sb, args[0], args[1]);
			processData(sb, args[2], args[3]);
			
			sb.append("</table>\n</body>\n</html>\n");
			PrintWriter out = new PrintWriter(new File("table.html"));
			out.println(sb.toString());
		    out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}	    
	}
	
	private static void processData(StringBuffer sb, String datafile, String labelfile) {		
		sb.append("<tr>\n");			
		
		StringBuffer hotelname = null;
		StringBuffer localarea = null;
		String stars = null;
		String price = null;
		String startdate = null;
		String enddate = null;		
		
		try{
			LineNumberReader dataReader = new LineNumberReader(new FileReader(datafile));
			LineNumberReader labelReader = new LineNumberReader(new FileReader(labelfile));
			String dataLine;
			String labelLine;
			boolean isIn = false;
			boolean isStartDateFound = false;
			while ((dataLine = dataReader.readLine()) != null) {
				labelLine = labelReader.readLine();
			
				dataLine = dataLine.split("\\s")[0].trim();
				labelLine = labelLine.trim();
				//System.out.println(dataLine + " " + labelLine);
				if(!isIn){
					if(labelLine.length() > 0){
						//start a new post
						isIn = true;
						isStartDateFound = false;
						hotelname = new StringBuffer();
						localarea = new StringBuffer();
						stars = null;
						price = null;
						startdate = null;
						enddate = null;	
					}
				}
				if(isIn){
					if(labelLine.length() == 0){
						//end of the post
						String[] data = new String[6];
						data[0] = hotelname.toString();
						data[1] = localarea.toString();
						data[2] = (stars != null)? stars :"";
						data[3] = (price != null)? price :"";
						data[4] = (startdate != null)? startdate :"";
						data[5] = (enddate != null)? enddate :data[4];						
						writeHtml(sb, data);
						isIn = false;
					}
					else{
						if(labelLine.equals(labels[0])){
							hotelname.append(dataLine + " ");
						}
						else if(labelLine.equals(labels[1])){
							localarea.append(dataLine + " ");
						}
						else if(labelLine.equals(labels[2])){
							stars = dataLine;
						}
						else if(labelLine.equals(labels[3])){
							price = dataLine;
						}
						else if(labelLine.equals(labels[4])){
							if(isStartDateFound){
								if(dataLine.length() > 0 && dataLine.indexOf("/") != -1){
									enddate = dataLine;
								}
							}
							else{
								if(dataLine.length() > 0 && dataLine.indexOf("/") != -1){
									isStartDateFound = true;
									startdate = dataLine;
								}
							}
						}
					}
				}
			}		
		}
		catch(Exception e){
			e.printStackTrace();
		}		
		sb.append("</tr>\n");
	}
	
	private static void writeHtml(StringBuffer sb, String[] data) {
		if(data[5].length() > 0 && data[4].length() > 0){
			String[] tok1 = data[4].split("/");
			String[] tok2 = data[5].split("/");
				
			for(int i = tok2.length; i > 0; --i){
				if(tok2[i-1].length() > 0){
					tok1[i - 1] = tok2[i-1];
				}
			}						
			StringBuffer stb = new StringBuffer();
			stb.append(tok1[0]);
			for(int i = 1; i < tok1.length; ++i){
				stb.append("/" + tok1[i]);
			}			
			data[5] = stb.toString();
		}
		//System.out.println(data[4] + " " + data[5]);
		
		sb.append("<tr>\n");
		for(int i = 0; i < 6; ++i){
			sb.append("<td>" + data[i] + "</td>\n");											
		}
		sb.append("</tr>\n");
	}
}

