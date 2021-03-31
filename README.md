# 包含网络
- 推荐使用: </br>
*Deep Belief Network (DBN)*  </br>

# 所依赖包
```python
pip install tensorflow (version: 2.X)
pip install keras

pip install --upgrade --user numpy pandas h5py (升级包)
```
# 用于任务
`use_for = 'classification'` 用于分类任务 </br>


# 版本信息


## User：
用户可以通过`model.py`文件控制一些功能的开关： </br>
`self.show_pic` => show curve in 'Console'? </br>
`self.tbd` => open/close tensorboard </br>
`self.save_model` => save/ not save model </br>
`self.plot_para` => plot W image or not </br>
`self.save_weight` => save W matrix or not </br>
`self.do_tSNE` => do t-SNE or not

## Version 2022.03.31:



# 测试结果
用于`minst`数据集分类，运行得到正确率可达98.78% </br>
用于`Urban Sound Classification`语音分类，正确率达73.37% </br>
(这个跑完console不会显示结果，因为是网上的比赛数据集，需上传才能得到正确率)</br>
用于`Big Mart Sales III`预测，RMSE为1152.04 </br>
(这个也是网上的数据集，也没有test_Y)</br></br>

跑的结果并不是太高，有更好的方法请赐教 </br>
语音分类未尝试语谱法，欢迎做过的和我交流 </br>

# 数据地址


# 参考资料
[Tensorflow基本函数](http://www.cnblogs.com/wuzhitj/p/6431381.html), 
[RBM原理](https://blog.csdn.net/itplus/article/details/19168937), 
[Hinton源码](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html), 


[Tensorboard](https://blog.csdn.net/sinat_33761963/article/details/62433234) 

[EDBN](https://www.sciencedirect.com/science/article/pii/S0019057819302903?via%3Dihub)
