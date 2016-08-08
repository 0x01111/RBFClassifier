package RBFClassifier;

import java.util.HashMap;

public class RBFClassifier {
	
	protected RBFNetwork network;
	
	/**
	 * 建立RBF分类模型
	 * @param num_inputs The number of inputs into the network
	 * @param num_gaussian The number of hidden nodes in the network
	 * @param num_output The number of categories (the output layer size).
	 */
	public RBFClassifier(int num_inputs, int num_gaussian, int num_output, float learning_rate, float gaussian_width) {
		network = new RBFNetwork(num_inputs, num_gaussian, num_output, learning_rate, gaussian_width);
	}
	
	/**
	 * 分类
	 * @param input The state
	 * @return The class fitting that input
	 */
	public int classify(float[] input) {
		
		//	First read from the the RBF network
		float[] class_vals = network.get_output(input);
		
		//	Find the argmax value (the most likely category)
		int best = 0;
		for (int i = 0; i < class_vals.length; i++) {
			if (class_vals[best] < class_vals[i]) {
				best = i;
				
			}
		}
		
		return best;
	}
	
	/**
	 * 训练模型
	 * @param input
	 * @param outcome
	 */
	public void learn(float[] input, float[] outcome) {
		float[] out = network.get_output(input);
		
		
		network.back_propogate(
				
				//	The expected is what the network thinks will happen
				outcome,
				
				//	The actual is given based on the data
				out);
	}

	/**
	 * Sets the weights of the hidden layer based on the data. Also writes
	 * the data to a file to read for later.
	 * @param filename The output file
	 * @param data The experimental data
	 * @param num_gaussian
	 */
	public void set_weights_w(String filename, float[][] data, int num_gaussian) {
		
		//	make k-means clusters
		float[][] kmeans = Kmeans.kmeans(data, num_gaussian);
		
		Main.float_to_csv(filename, kmeans);

		//	Simply these clusters to the hidden layer
		for (int i = 0; i < kmeans.length; i++) {
			network.gnodes[i].set_centers(kmeans[i]);
		}
	}
	
	/**
	 * Sets the weights of the hidden layer based on the data. Also writes
	 * the data to a file to read for later.
	 * @param filename The input file
	 */
	public void set_weights_r(String filename) {
		float[][] centers = Main.csv_to_float("centers.csv");
		for (int i = 0; i < centers.length; i++) {
			network.gnodes[i].set_centers(centers[i]);
		}
	}
	


}

