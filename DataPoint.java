
public class DataPoint {
	String docId;
	String relevanceLabel;
	
	public DataPoint(String docId) {
		docId = docId;
	}
	
	public double getFeatureValue(String featureId) {
		return 0.0;
	}
	
	public double[] getFeatureVector() {
		//Based on the Letor Dataset
		return new double[46];
	}
	
	public static double[] buildFeatureVector(String query, String DocId) {
		return new double[46];
	}
}
