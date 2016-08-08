package RBFClassifier;

/**
 * 隐含层 RBF结点
 *
 */
class GaussianNode {
	float[] centers; // 高斯中心 
	protected static float C; // 高斯中心 标准差 
	
	public GaussianNode() { }
	
	public float output(float[] raw) {
		
		//	Gaussify the distance between input layer and the gauss centers
		return gaussian_function(
			Kmeans.euclidean_distance(raw, centers)
		);
	}
	
	public void set_centers(float[] centers) {
		this.centers = centers;
	}
	// 计算高斯距离
	private static float gaussian_function(double in) {
		return (float)Math.exp(- (in * in) / (2 * C * C));
	}
}
