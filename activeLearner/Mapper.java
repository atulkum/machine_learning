import uk.ac.shef.wit.simmetrics.similaritymetrics.Levenshtein;

import com.wcohen.ss.JaroWinkler;
import com.wcohen.ss.SoftTFIDF;


public class Mapper {
	public static SoftTFIDF tfidf_measure = new SoftTFIDF(new JaroWinkler(), 0.9);
	public static Levenshtein lev_measure = new Levenshtein();
	
	private int measure = -1;
	
	public Mapper(int m){
		measure = m;
	}
	public double getScore(String s1, String s2){
		if(s1 == null|| s2 == null){
			return 0;
		}
		if(s1.trim().equals("")){
			if(s2.trim().equals("")){
				return 1;
			}
			else{
				return 0;
			}
		}
		if(s2.trim().equals("")){
			if(s1.trim().equals("")){
				return 1;
			}
			else{
				return 0;
			}
		}
		double score = 0;
		switch(measure){
		case 1:	score = tfidf_measure.score(s1, s2);break;
		case 2: score = lev_measure.getSimilarity(s1, s2);break;
		}		
		return score;
	}	
}
