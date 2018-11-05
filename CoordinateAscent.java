import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;

public class CoordinateAscent {
	
	public double[] weightVector = new double[46];
	
	public double dotProduct(double[] v1, double[] v2) {
		double dp = 0.0;
		for(int i=0; i<v1.length; i++) {
			dp = dp + (v1[i] * v2[i]);
		}
		return dp;
	}
	
	public HashMap<DataPoint, Double> sortMap(HashMap<DataPoint, Double> inputMap){
		HashMap<DataPoint, Double> sortedMap = new HashMap<DataPoint, Double>(inputMap);
		List<Entry<DataPoint, Double>> sortedDocs = new LinkedList<Entry<DataPoint, Double>>(inputMap.entrySet());
		Collections.sort(sortedDocs, new Comparator<Entry<DataPoint, Double>>(){

			public int compare(Entry<DataPoint, Double> o1, Entry<DataPoint, Double> o2) {
				//Sort by largest score first - descending value
				return o2.getValue().compareTo(o1.getValue());
			}
			
		});
		for(Entry<DataPoint, Double> e : sortedDocs) {
			sortedMap.put(e.getKey(), e.getValue());
		}
		return sortedMap;
	}
	
	@SuppressWarnings("unchecked")
	public void train(List<DataPoint>[] trainingData) {
		double[] idealWeightVector = new double[46];
		for(int w=0; w<weightVector.length; w++) {
		//For each parameter -- adjust the knob and score all the queries in the training data
			//Find a total score by adding the score for all training data and dividing by the number of training samples
			//We need to find the w with the maxScore
			double totalScore = 0.0;
			
			//You can either increase or decrease the value v for a weight vector
			//The upper and lower bounds can also be changed or figured out later
			for(int v = 0; v < 200; v=v+2) {
				double tempScore = 0.0;
				//Change the values of the weight vector
				weightVector[w] = v;
			
				//For each query
				for(List<DataPoint> queryList : trainingData) {
					//For each data point - get the feature vector
					HashMap<DataPoint, Double> scoredDocs = new HashMap<DataPoint, Double>();
					for(DataPoint d : queryList) {
						double[] featureVector = d.getFeatureVector();
						double score = dotProduct(featureVector, weightVector);
						scoredDocs.put(d, score);
					}
					
					//Get the ranked list for the query - sort for every change in the weight vector - computationally expensive
					HashMap<DataPoint, Double> finalRankedList = sortMap(scoredDocs);
					//This is for all the training data. This does not generalize well for other training sets.
					tempScore = tempScore + Metric.scoreRanking((List<DataPoint>) finalRankedList.keySet());
					
				}
				tempScore = tempScore / trainingData.length;
				
				if (tempScore > totalScore) {
					//We have a better value for w -- keep iterating
					idealWeightVector[w] = v;
					totalScore = tempScore;
				}
				else {
					//Move to the next weight -- as we have hit the maximum for this feature already
					break;
				}
					
			}
		}
		//Copy over the ideal Weight Vector
		weightVector = idealWeightVector;
	}
	
	@SuppressWarnings("unchecked")
	public List<DataPoint> predict(String query, List<String> candidateDocIds){
		HashMap<DataPoint, Double> scoredDocs = new HashMap<DataPoint, Double>();
		for(String d: candidateDocIds) {
			double[] candidateVector = DataPoint.buildFeatureVector(query, d);
			DataPoint dp = new DataPoint(d);
			double score = dotProduct(candidateVector, weightVector);
			scoredDocs.put(dp, score);
		}
		//Get the ranked list for the query
		HashMap<DataPoint, Double> finalRankedList = sortMap(scoredDocs);
		return ((List<DataPoint>) finalRankedList.keySet());
	}
	
	public static List<DataPoint>[] loadAll(){
		return null;
	}
	
	public static void printResult(List<DataPoint> finalList) {
		System.out.println("The ranked list is....");
		for(DataPoint d : finalList) {
			System.out.println(d.docId);
		}
	}
	
	public static void main(String[] args) {
		List<DataPoint>[] trainingData = CoordinateAscent.loadAll();
		String query = null;
		List<String> candidateDocIds = null;
		
		CoordinateAscent cAscent = new CoordinateAscent();
		cAscent.train(trainingData);
		List<DataPoint> searchResult = cAscent.predict(query, candidateDocIds);
		CoordinateAscent.printResult(searchResult);
	}
	
}
