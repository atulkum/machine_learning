import java.io.*;
import java.util.*;
public class Trie{
	Node start;
	int totalNodes;
	HashMap groups = new HashMap();
	HashSet finalg = new HashSet();
	HashSet nonfinalg = new HashSet();
	ArrayList transition;
	Node finalNode;
	void build(String fileName) throws IOException{
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(new DataInputStream(
					new FileInputStream(fileName))));

		String line = null;
		while((line = reader.readLine()) != null){
			line = line.trim();
			if(!line.equals("")){
				String[] labels = line.split("\\s+");
				start = insert(start, labels, 0);
			}
		}
		shrinkFinal();
	}
	Node insert(Node root, String[] labels, int n){
		if (root == null){
			totalNodes++;
			root = new Node();
		}
		if (labels.length == n) {
			root.isFinal = true;
		}
		else {
			if(labels[n].charAt(0) == '-'){
				root.link[26] = insert(root.link[26], labels, n+1);
				root.link[26].parent = root;
			}
			else{
				root.link[labels[n].charAt(0) - 'A'] = insert(root.link[labels[n].charAt(0) - 'A'], labels, n+1);
				root.link[labels[n].charAt(0) - 'A'].parent = root;
			}
			root.hasChild = true;
		}
		return root;
	}
	void shrinkFinal(){
		finalNode = new Node();
		finalNode.isFinal = true;
		setFinal(start, finalNode);
	}
	void setFinal(Node root, Node fnode){
		if(root == null){
			return;
		}
		for(int i = 0; i < 27; i++){
			if(root.link[i] != null){
				if(!root.link[i].hasChild){
					root.link[i] = fnode;
				}
				setFinal(root.link[i], fnode);
			}
		}
	}
	//print fsa without minimization
	void printAll(){		
		System.out.println(finalNode.id);
		print(start);
		System.out.println("(" + finalNode.id + " (" + start.id +" \"_\"))");
	}
	void print(Node root){
		if(root == null){
			return;
		}
		for(int i = 0; i < 27; i++){
			if(root.link[i] != null){
				String tran = "(" + root.id + " (" + root.link[i].id + " \"";				
				if(i != 26){
					tran = tran + (char)(i + 'A');
				}
				else{
					tran = tran + '-';
				}
				tran = tran + "\"))";

				System.out.println(tran);
				if(root.link[i].isFinal && root.link[i].hasChild){
					tran = "(" + root.link[i].id + " (" + finalNode.id + " *e*))";
					System.out.println(tran);
				}
				print(root.link[i]);
			}
		}
	}

	void minimize(){
		minInit(start);
		nonfinalg.add(start);
		start.group = 1;
		/*System.out.println();
		Iterator itr = nonfinalg.iterator();
		while(itr.hasNext()){
			Node ele = (Node)itr.next();
			System.out.print(ele.id + ",");
		}
		System.out.println();
		itr = finalg.iterator();
		while(itr.hasNext()){
			Node ele = (Node)itr.next();
			System.out.print(ele.id + ",");
		}
		System.out.println("**************************");*/
		groups.put(new Integer(2), finalg);
		groups.put(new Integer(1), nonfinalg);

		int groupid = 2;
		int groupsize;
		while(true){
			groupsize = groups.size();
			//new groups
			HashMap newGrps = new HashMap();
			Iterator keys = groups.keySet().iterator();
			while(keys.hasNext()){
				Integer gid = (Integer)keys.next();
				HashSet onegroup = (HashSet)groups.get(gid);
				//System.out.println(gid.intValue());
				if(onegroup.size() == 1){
					newGrps.put(gid, onegroup);
					continue;
				}
				ArrayList tobedeleted = new ArrayList();
				Iterator inGrpItr = onegroup.iterator();
				while(inGrpItr.hasNext()){
					Node member = (Node)inGrpItr.next();
					if(!member.hasChild  || member.group != gid.intValue()){
						continue;
					}

					HashSet newgroup = new HashSet();
					groupid++;
					member.group = groupid;
					newgroup.add(member);
					newGrps.put(new Integer(groupid), newgroup);
					tobedeleted.add(member);
					Iterator inGrpItr2 = onegroup.iterator();
					while(inGrpItr2.hasNext()){
						Node othermember = (Node)inGrpItr2.next();
						if(othermember.id == member.id || othermember.group != gid.intValue()) {
							continue;
						}
						boolean allMatching = true;
						for(int j = 0; j < 27; ++j){
							if((othermember.link[j] != null && member.link[j] == null)||
									(othermember.link[j] == null && member.link[j] != null)){
								allMatching = false;
							}
							if(othermember.link[j] != null && member.link[j] != null
									&&(othermember.link[j].group != member.link[j].group)){
								allMatching = false;
							}
						}
						if(allMatching){
							othermember.group = member.group;							
							newgroup.add(othermember);
							tobedeleted.add(othermember);
						}
					}
				}
				//remove the group members who are no longer here
				for(int i = 0;i <tobedeleted.size(); ++i ){
					onegroup.remove(tobedeleted.get(i));
				}
				if(onegroup.size() > 0){
					newGrps.put(gid, onegroup);
				}
			}
			groups = newGrps;
			if(groupsize == groups.size()){
				break;
			}
		}

		/*Iterator keys = groups.keySet().iterator();
		while(keys.hasNext()){
			Integer groupn = (Integer)keys.next();
			HashSet grp = (HashSet)groups.get(groupn);
			System.out.print(groupn + ":");
			Iterator inGrpItr = grp.iterator();
			while(inGrpItr.hasNext()){
				Node ele = (Node)inGrpItr.next();
				System.out.print(ele.id + "|" + ele.group + ",");
			}
			System.out.println();
		}*/
	}

	void minInit(Node root){
		if(root == null){
			return;
		}
		for(int i = 0; i < 27; i++){
			if(root.link[i] != null){
				if(root.link[i].isFinal){
					finalg.add(root.link[i]);
					root.link[i].group = 2;
				}
				else{
					nonfinalg.add(root.link[i]);
					root.link[i].group = 1;
				}
				minInit(root.link[i]);
			}
		}
	}
	//print fsa with minimization
	void printAllMin(){
		transition = new ArrayList();				
		printMin(start);
		//removing the duplicate transitions
		LinkedHashSet uniquetrans = new LinkedHashSet(transition);
		System.out.println(finalNode.group);
		Iterator itr = uniquetrans.iterator();
		while(itr.hasNext()){			
			System.out.println((String)itr.next());
		}
		System.out.println("(" + finalNode.group + " (" + start.group + " \"_\"))");
	}
	void printMin(Node root){
		if(root == null){
			return;
		}
		for(int i = 0; i < 27; i++){
			if(root.link[i] != null){				
				String tran = "(" + root.group + " (" + root.link[i].group + " \"";				
				if(i != 26){
					tran = tran + (char)(i + 'A');
				}
				else{
					tran = tran + '-';
				}
				tran = tran + "\"))";				
				transition.add(tran);				
				if(root.link[i].isFinal && root.link[i].hasChild){
					tran = "(" + root.link[i].group + " (" + finalNode.group + " *e*))";
					transition.add(tran);
				}
				printMin(root.link[i]);
			}
		}
	}

	public static void main(String[] args){		
		if(args.length < 1){
			System.out.println("USAGE: java  -Xms1024m -Xmx1024m Trie <filename> [m]  >  <output file name>");
			System.out.println("filename - the dictionary filename.");
			System.out.println("m is optional if you want minimized FSA.");
			System.out.println("minimized FSA needs -Xms1024m -Xmx1024m as it needs lots of memory.");
			return;
		}
		Trie t = new Trie();
		boolean isMinimized = false;
		try {
			t.build(args[0]);
			if(args.length > 1 && args[1].endsWith("m")){
				isMinimized = true;
			}
			if(	isMinimized){
				t.minimize();
				t.printAllMin();
			}
			else{
				t.printAll();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

class Node{
	boolean isFinal;
	Node[] link;
	int id;
	Node parent;
	int group;
	static int numNodes = 0;
	boolean hasChild;
	public Node(){
		isFinal = false;
		id = numNodes++;
		link = new Node[27];
		hasChild = false;
	}
}