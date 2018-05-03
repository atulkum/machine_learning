import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class homework1 {
	public static final String newsstr = "elements/news";
	public static final String titlestr = "title";
	public static final String summarystr = "summary";
	public static final String latimesXml = "LATimes.xml";
	public static final String reutersXml = "Reuters.xml";
	public static final String prefixUrl = "http://www.dapper.net/RunDapp?dappName=";
	public static final String suffixUrl = "&v=1&v_searchstr=";
	public static final String latimesDapper = "atulk_latimes_548hw1";
	public static final String reutersDapper = "atulk_reuter_548hw1";
		
	public static void main(String[] args){
		if(args.length != 1){
			System.out.println("Usage: java homework1 <Serach Term>");
			System.exit(0);
		}
		
		String serachTerm = args[0];		
	    
	    try {
	    	getXML(latimesDapper, serachTerm, latimesXml);
	    	getXML(reutersDapper, serachTerm , reutersXml);
	    	
	    	StringBuffer sb = new StringBuffer();
	        sb.append("<html>\n");sb.append("<body>\n");sb.append("<title>homework1</title>\n");
	 	    sb.append("<p><b>Search Term: " + serachTerm +"</b></p>\n");
	 	
			readXML(sb, latimesXml, "LA Times News Results");		    
			sb.append("\n</p></br>\n");
			readXML(sb, reutersXml, "Reuters News Results");
	    	
			PrintWriter out = new PrintWriter(new File("Results.html"));
			out.println(sb.toString());
		    out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}	    
	}
	static public void getXML(String dapperName,String searchterm, String filename) throws IOException{
		String urlstr = prefixUrl + dapperName + suffixUrl + searchterm;
		URL url = new URL(urlstr);

    	URLConnection URLconnection = url.openConnection ( ) ;
    	HttpURLConnection httpConnection = (HttpURLConnection)URLconnection;

    	if ( httpConnection.getResponseCode ( ) == HttpURLConnection.HTTP_OK) {
    		InputStream in = httpConnection.getInputStream ( );	  
    		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
    		PrintWriter out = new PrintWriter(new FileWriter(filename));
    		
    		String line;
    		while((line = reader.readLine()) != null){
    			out.println(line);
    		}
    		reader.close();
    		out.close();
    	} else {
    		System.out.println( "XML file not available" );	                 
    	} 
	}
	static public void readXML(StringBuffer sb, String xmlfile, String heading) throws Exception{		
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		dbf.setNamespaceAware(true);
		dbf.setValidating(false);
		DocumentBuilder db = dbf.newDocumentBuilder();
		XPathFactory xpf4dom = XPathFactory.newInstance();
		XPath xpath4dom = xpf4dom.newXPath();
		XPathExpression newsExpr = xpath4dom.compile(newsstr);		   
		XPathExpression titleExpr = xpath4dom.compile(titlestr);		   
		XPathExpression summaryExpr = xpath4dom.compile(summarystr);		   

		Document doc = db.parse(xmlfile);
		NodeList news = (NodeList)newsExpr.evaluate(doc, XPathConstants.NODESET);		    		    
		int nnews = news.getLength();		    		    
		sb.append("<p><b>" + heading + "</b></p>\n");
		sb.append("<table border=\"1\">\n");
		sb.append("<tr bgcolor=#00FFFF>\n");
		sb.append("<th>News Title</th>\n");
		sb.append("<th>News Summary</th>\n");
		sb.append("<th>Link</th>\n");
		sb.append("</tr>\n");
		for(int i = 0; i < nnews; ++i){
			Node item = news.item(i);		    	
			Node title = (Node)titleExpr.evaluate(item, XPathConstants.NODE);
			Node summary = (Node)summaryExpr.evaluate(item, XPathConstants.NODE);
			
			sb.append("<tr>\n");			
			
			sb.append("<td>");
			if(title != null){
				sb.append(title.getTextContent());
			}
			sb.append("</td>\n");			
			
			sb.append("<td>");
			if(summary != null){
				sb.append(summary.getTextContent());
			}
			sb.append("</td>\n");
			
			sb.append("<td>");
			if(title != null){
				sb.append(title.getAttributes().getNamedItem("href").getTextContent());
			}
			sb.append("</td>\n");
			
			sb.append("</tr>\n");
		}
		sb.append("</table>");
	}
}
