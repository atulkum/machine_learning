import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.table.AbstractTableModel;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;


public class PhoneBookSearchGUI {
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
	public static MyModel model = new MyModel();	
	
	public static void main(String[] args){		
		JFrame frame = new JFrame("Phone Book");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);		
		frame.setSize(750, 200);
		JPanel pane = new JPanel();
		pane.setSize(720, 125);
		pane.setLayout(null);
		frame.getContentPane().add(pane);				
		
		JPanel ypane = new JPanel();				
		ypane.setSize(360, 125);
		ypane.setLocation(10, 10);
		ypane.setLayout(null);
		pane.add(ypane);		
		
		JPanel wpane = new JPanel();
		wpane.setSize(360, 125);
		wpane.setLocation(380, 10);
		wpane.setLayout(null);
		pane.add(wpane);
	
		JLabel yname = new JLabel("Yellow Pages");
		yname.setSize(100, 20);
		yname.setLocation(70, 0);
		ypane.add(yname);
		
		JLabel bname = new JLabel("Business Name");
		bname.setSize(100, 20);
		bname.setLocation(0, 30);
	
		final JTextField bnametf = new JTextField();	
		bnametf.setSize(200, 20);
		bnametf.setLocation(110, 30);
	
		ypane.add(bname);
		ypane.add(bnametf);					
		
		JLabel yaddress = new JLabel("Address");
		yaddress.setSize(100, 20);
		yaddress.setLocation(0, 50);

		final JTextField yaddresstf = new JTextField();
		yaddresstf.setSize(200, 20);
		yaddresstf.setLocation(110, 50);

		ypane.add(yaddress);
		ypane.add(yaddresstf);
		
		JLabel wname = new JLabel("White Pages");
		wname.setSize(100, 20);
		wname.setLocation(70, 0);
		wpane.add(wname);
		
		JLabel fname = new JLabel("First Name");		
		fname.setSize(100, 20);
		fname.setLocation(0, 30);

		final JTextField fnametf = new JTextField();
		fnametf.setSize(200, 20);
		fnametf.setLocation(110, 30);

		wpane.add(fname);
		wpane.add(fnametf);			
		
		JLabel lname = new JLabel("Last Name");
		lname.setSize(100, 20);
		lname.setLocation(0, 50);

		final JTextField lnametf = new JTextField();
		lnametf.setSize(200, 20);
		lnametf.setLocation(110, 50);

		wpane.add(lname);
		wpane.add(lnametf);
		
		JLabel waddress = new JLabel("Address");
		waddress.setSize(100, 20);
		waddress.setLocation(0, 70);

		final JTextField waddresstf = new JTextField();		
		waddresstf.setSize(200, 20);
		waddresstf.setLocation(110, 70);

		wpane.add(waddress);
		wpane.add(waddresstf);							
		
		JButton playw = new JButton("Get Results"); 
		playw.setSize(100, 20);
		playw.setLocation(350, 140);
		pane.add(playw);
		playw.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent event){
				boolean isy = false;
				String nameTerm = null;
				String addressTerm = null;
				
				if(!bnametf.getText().trim().equals("")){
					isy = true;
					nameTerm = bnametf.getText().trim();					
				}
				if(!yaddresstf.getText().trim().equals("")){
					isy = true;
					addressTerm = yaddresstf.getText().trim();					 
				}
				if(isy){
					try {
						getXML(ypDapper, nameTerm, addressTerm, ypXml);
						readXML(ypXml, ypEntry);
					} catch (Exception e) {
						e.printStackTrace();
					}
					showTable();
				}
				else{
					if(!fnametf.getText().trim().equals("") || !lnametf.getText().trim().equals("")){
						isy = false;
						if(!fnametf.getText().trim().equals("") && !lnametf.getText().trim().equals("")){
							nameTerm = fnametf.getText().trim() + " " + lnametf.getText().trim();
						}
						else if(!fnametf.getText().trim().equals("")){
							nameTerm = fnametf.getText().trim();
						}
						else{
							nameTerm = lnametf.getText().trim();
						}
					}
					if(!waddresstf.getText().trim().equals("")){
						isy = false;
						addressTerm = waddresstf.getText().trim();					 
					}
					if(nameTerm != null || addressTerm != null){
						try {
							getXML(wpDapper, nameTerm, addressTerm, wpXml);
							readXML(wpXml, wpEntry);
						} catch (Exception e) {
							e.printStackTrace();
						}	
						showTable();
					}					
				}
			}

			private void showTable() {
				JFrame frame = new JFrame("Phone Book");
				JPanel panel = new JPanel();
				panel.setLayout(new GridLayout(1, 0));
				JTable table = new JTable(model);
				table.setPreferredScrollableViewportSize(new Dimension(500, 70));									
				JScrollPane scrollPane = new JScrollPane(table);					
				panel.add(scrollPane);					
				panel.setOpaque(true); 
				frame.setContentPane(panel);
				frame.pack();
				frame.setVisible(true);				
			}
		});		
		frame.setVisible(true);
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
		
		ArrayList<String> phonearr = new ArrayList<String>();
		ArrayList<String> namearr = new ArrayList<String>();
		ArrayList<String> addrarr = new ArrayList<String>();
		
		for(int i = 0; i < nEntry; ++i){
			Node item = entries.item(i);		    	
			
			Node phone = (Node)phoneExpr.evaluate(item, XPathConstants.NODE);
			Node name = (Node)nameExpr.evaluate(item, XPathConstants.NODE);
			Node address = (Node)addressExpr.evaluate(item, XPathConstants.NODE);
			
			if(phone != null && (name != null || address != null)){
				if(!phone.getTextContent().trim().equals("") &&
					(!name.getTextContent().trim().equals("") ||
					!address.getTextContent().trim().equals(""))){
					
					phonearr.add(phone.getTextContent());					
					namearr.add(name.getTextContent());					
					addrarr.add(address.getTextContent());					
				}								
			}			
		}
		Object[][] data = new Object[phonearr.size()][3];
		int i = 0;
		for(String p : phonearr){
			data[i++][0] = p; 
		}
		i = 0;
		for(String n : namearr){
			data[i++][1] = n; 
		}
		i = 0;
		for(String a : addrarr){
			data[i++][2] = a; 
		}
		model.setData(data);
	}
}

class MyModel extends AbstractTableModel {
	private String[] headings = { "Phone number", "Name", "Address" };
	private Object[][] data;

	public int getColumnCount() {
		return headings.length;
	}

	public int getRowCount() {
		return data.length;
	}

	public String getColumnName(int col) {
		return headings[col];
	}

	public Object getValueAt(int row, int col) {
		return data[row][col];
	}
	public void setData(Object[][] data){
		this.data = data;
	}
}
