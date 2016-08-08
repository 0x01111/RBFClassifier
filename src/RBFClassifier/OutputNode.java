package RBFClassifier;

/**
 * 输出结点
 * 
 */
class OutputNode {
	protected float[] weights;// 权值
	protected float[] last_input; // 最后输出值 
	
	public OutputNode(int num_inputs) {
		weights = new float[num_inputs];
		for (int i = 0; i < num_inputs; i++) {
			
			//	初始化权值-0.5 到 0.5 之间
			weights[i] = (float)(Math.random() - .5);
		}
	}
	
	public float output(float[] goutput) {
		last_input = goutput;
		
		float sum = 0f;
		
		//	Dot product weights*goutput
		for (int i = 0; i < goutput.length; i++) {
			sum += weights[i] * goutput[i];
		}
		
		return sum;
	}
}


