package RBFClassifier;

public class RBFNetwork {
	protected GaussianNode[] gnodes; // 高斯层结点
	protected OutputNode[] onodes; // 输出层结点
	protected float learning_rate; // 学习率 
	
	/**
	 * 构造函数 
	 * @param num_inputs 输入层个数
	 * @param num_gaussian 高斯层个数
	 * @param num_outputs 输出层个数 
	 */
	public RBFNetwork(int num_inputs, int num_gaussian, int num_outputs, float learning_rate, float gaussian_width) {
		this.learning_rate = learning_rate;
		GaussianNode.C = gaussian_width;
		
		//初始化高斯中心
		gnodes = new GaussianNode[num_gaussian];
		for (int i = 0; i < num_gaussian; i++) {
			//	Inputs = input space
			gnodes[i] = new GaussianNode();
		}
		
		//初始化输出层权值
		onodes = new OutputNode[num_outputs];
		for (int i = 0; i < num_outputs; i++) {
			onodes[i] = new OutputNode(num_gaussian);
		}
	}
	
	/**
	 * 根据训练数据，计算输出值  
	 * @param input
	 * @return The outputs
	 */
	public float[] get_output(float[] input) {

		//	前馈过程，输入层 到 高斯层，计算高斯层的输入值 
		float gaussian[] = new float[gnodes.length];
		for (int i = 0; i < gaussian.length; i++) {
			gaussian[i] = gnodes[i].output(input);
		}
		
		// 前馈过程，高斯层到输入层，计算最后的输出值
		float output[] = new float[onodes.length];
		for (int i = 0; i < output.length; i++) {
			output[i] = onodes[i].output(gaussian);
		}
		
		return output;
	}
	
	/**
	 * 根据梯度下降法 更新隐含层权值
	 * 
	 * 
	 * @param expected What was thought to be the outcome
	 * @param outcome The actual outcome
	 */
	public void back_propogate(float[] expected, float[] outcome) {
		
		//	计算每个节点的训练误差 
		for (int i = 0; i < onodes.length; i++) {
			
			//	Get the error of this classification
			float error = expected[i] - outcome[i];
			
			//	更新权值  Update the weights
			for (int j = 0; j < onodes[i].weights.length; j++){
				
				//	The weight update function
				onodes[i].weights[j] += 
						learning_rate *
						error *
						onodes[i].last_input[j];// * onodes[i].weights[j];
			}
		}
	}
}
