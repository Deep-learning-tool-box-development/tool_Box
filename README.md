# Including algorithms
- 推荐使用: </br>
*Convolution Neural Network (CNN)*</br>
*Deep Belief Network (DBN)*  </br>
*Particle swarm optimization (PSO)*  </br>
*Simulated Annealing(SA)*  </br>
*K-nearest neighbors  (KNN)*  </br>
*Support Vector Machine (SVM)*  </br>
# Install packages
```python
pip install tensorflow (version: 2.X)
pip install keras
pip sklearn
pip install --upgrade --user numpy pandas h5py (升级包)
pip install matplotlib
pip install scipy
```
# Environment
```
matplotlib 3.3.4
numpy 1.19.5
pandas 1.2.3
sklearn 0.22.2.post1
tensorflow 2.4.1
tensorflow.keras 2.4.0
scipy 1.6.2
```
# User guide
算法接口文档 as reference

user only need to change option between "PSO" and "SA" under cnn_main or dbn_main or svm_main. 

# 测试结果

**cnn_pso:**

<img src="C:\Users\zhudifan\AppData\Roaming\Typora\typora-user-images\image-20210414190451173.png" alt="image-20210414190451173" style="zoom: 67%;" />

**cnn_sa:**

<img src="C:\Users\zhudifan\AppData\Roaming\Typora\typora-user-images\image-20210414190551716.png" alt="image-20210414190551716" style="zoom: 50%;" />



# 参考资料

[Tensorflow基本函数](http://www.cnblogs.com/wuzhitj/p/6431381.html), 
[RBM原理](https://blog.csdn.net/itplus/article/details/19168937), 
[Hinton源码](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html), 


[Tensorboard](https://blog.csdn.net/sinat_33761963/article/details/62433234) 

[EDBN](https://www.sciencedirect.com/science/article/pii/S0019057819302903?via%3Dihub)