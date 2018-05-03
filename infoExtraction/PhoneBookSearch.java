import java.io.BufferedReader;
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


public class PhoneBookSearch {
	public static final String phone = "phone";
	public static final String name = "name";
	public static final String address = "address";
	public static final String prefixUrl = "http://service.openkapow.com/atulcse/";
	
	public static final String wpEntry = "result/wp";
	public static final String ypEntry = "result/yp";
		
	public static final String wpXml = "kumaratulhw4wp.xml";
	public static final String ypXml = "kumaratulhw4yp.xml";
				
	public static final String wpDapper = "kumaratulhw4wp.xml?";
	public static final String ypDapper = "kumaratulhw4yp.xml?";
		
	public static void main(String[] args){		
		if(args.length < 2){
			System.out.println("Usage: java PhoneBookSearch mode=<yp/wp> name=<name> address=<address>");
			System.exit(0);
		}
		String modeStr = args[0].split("=")[1];		
		
		String nameTerm = null;
		String addressTerm = null;
		
		for(String arg: args){
			String[] search = arg.split("=");
			if(search[0].equals("name")){
				nameTerm = search[1]; 
			}
			else if(search[0].equals("address")){
				addressTerm = search[1];
			}
		}	
	    try {
	    	if(modeStr.equals("yp")){
	    		getXML(ypDapper, nameTerm, addressTerm, ypXml);
	    		readXML(ypXml, ypEntry);
	    	}
	    	else{
	    		getXML(wpDapper, nameTerm, addressTerm, wpXml);
	    		readXML(wpXml, wpEntry);
	    	}	    						    					    	
		} catch (Exception e) {
			e.printStackTrace();
		}  	    
	}
	
	static public void getXML(String dapperName,String nameterm, String addressterm, String filename) throws IOException{		
		StringBuffer urlstr = new StringBuffer();
		urlstr.append(prefixUrl);
		urlstr.append(dapperName);
		
		if(nameterm != null && addressterm != null){
			urlstr.append("name=" + nameterm +"&address=" + addressterm);
		}
		else if(nameterm != null){
			urlstr.append("name=" + nameterm);
		}
		else if(addressterm != null){
			urlstr.append("address=" + addressterm);
		}
		String urlst = urlstr.toString().trim().replace(" ", "+");
		URL url = new URL(urlst);
		
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
    		System.out.println( "Dapper Server Timed Out..." );	                 
    	} 
	}
	static public void readXML(String xmlfile, String starttag) throws Exception{		
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		dbf.setNamespaceAware(true);
		dbf.setValidating(false);
		DocumentBuilder db = dbf.newDocumentBuilder();
		XPathFactory xpf4dom = XPathFactory.newInstance();
		XPath xpath4dom = xpf4dom.newXPath();

		XPathExpression phonesExpr = xpath4dom.compile(starttag);		   
		XPathExpression phoneExpr = xpath4dom.compile(phone);
		XPathExpression nameExpr = xpath4dom.compile(name);		   
		XPathExpression addressExpr = xpath4dom.compile(address);		   
				
		Document doc = db.parse(xmlfile);
		NodeList entries = (NodeList)phonesExpr.evaluate(doc, XPathConstants.NODESET);		    		    
		int nEntry = entries.getLength();
		System.out.println("*************************");
		for(int i = 0; i < nEntry; ++i){
			Node item = entries.item(i);		    	
			
			Node phone = (Node)phoneExpr.evaluate(item, XPathConstants.NODE);
			Node name = (Node)nameExpr.evaluate(item, XPathConstants.NODE);
			Node address = (Node)addressExpr.evaluate(item, XPathConstants.NODE);
			
			if(phone != null && (name != null || address != null)){
				if(!phone.getTextContent().trim().equals("") &&
					(!name.getTextContent().trim().equals("") ||
					!address.getTextContent().trim().equals(""))){
					
					System.out.println("Phone Number: " + phone.getTextContent());
					
					if(!name.getTextContent().trim().equals("")){
						System.out.println("Name: " + name.getTextContent());
					}
					if(!address.getTextContent().trim().equals("")){
						System.out.println("Address: " + address.getTextContent());
					}
					System.out.println("*************************");
				}								
			}			
		}		
	}
}
