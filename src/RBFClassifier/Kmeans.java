package RBFClassifier;

import java.util.HashMap;

public class Kmeans {
	/**
	 * 根据Kmeans确定隐含层的类中心 
	 * @param data 输入数据集 
	 * @param num_gaussian 聚类的数量 
	 * @return  返回隐含层高斯中心 
	 */
	public static float[][] kmeans(float[][] data, int num_gaussian) {
		
		//	聚类中心，每行是一个中心
		float[][] centers = new float[num_gaussian][data[0].length];
		
		//	初始化中心
		System.arraycopy(data, 0, centers, 0, centers.length);
		
		//	Prepare the loop
		HashMap<float[], Integer> old_assignments = null;
		boolean changed = true;
		
		// 聚类中心没有改变的时候，聚类结束
		
		while (changed) {
			
			changed = false;
			HashMap<float[], Integer> assignments = new HashMap<>(); // 保存分配的中心 以及数据
			float[][] new_centers = new float[num_gaussian][data[0].length];
			
			//	Assignments
			int[] center_count = new int[num_gaussian];
			for (float[] f : data) {
				
				//	找到最近的中心 
				int min_index = 0;
				double min_dis = euclidean_distance(centers[0], f);
				for (int i = 1; i < centers.length; i++) {
					double distance = euclidean_distance(centers[i], f);
					if (distance < min_dis) {
						min_dis = distance;
						min_index = i;
					}
				}
				
				//	Save the this data point's closest center
				assignments.put(f, min_index);
				
				//	Did the point change closest cluster since last time?
				if (old_assignments == null || old_assignments.get(f) != min_index) {
					changed = true;
				}
				
				//	新的聚类中心，这个类中心没有计算均值 
				for (int i = 0; i < f.length; i++) {
					new_centers[min_index][i] += f[i];
				}
				
				//	计算该类中心的数据量 
				center_count[min_index]++;
				
			}
			
			//	类中心计算均值
			for (int i = 0; i < num_gaussian; i++) {
				for (int j = 0; j < new_centers[i].length; j++) {
					
					//	Average the centers
					if (center_count[i] != 0) {
						new_centers[i][j] /= center_count[i];
					}
				}
			}
			
			//	更新
			old_assignments = assignments;
			centers = new_centers;
		}
		
		//	返回最好的类中心
		return centers;
	}
	
	/**
	 * 计算欧拉距离
	 *
	 * @param a One vector
	 * @param b Another vector
	 * @return 欧拉距离
	 */
	public static double euclidean_distance(float[] a, float[] b) {
		double sum = 0.0;
		for (int i = 0; i < a.length; i++) {
			sum += Math.pow(a[i]-b[i],2);
		}
		return Math.sqrt(sum);
	}
}
