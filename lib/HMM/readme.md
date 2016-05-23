#### HMM

HMM(隐马尔可夫模型, Hide Markov Model), 包括前向算法、viterbi算法、Baum-Welch算法。   
算法细节介绍可以参考: [隐马尔可夫模型](http://www.cosmosshadow.com/机器学习/2015/08/10/隐马尔可夫模型.html)   

hmmlearn.py来源: http://www.cnblogs.com/hanahimi/p/4011765.html   
HMMArthur.py是我根据上面一个修改而成   
主要修改点: 

* 矩阵化，代码量更小一些
* 优化函数参数传入
* 修改"BackwardWithScale函数传scale含义不明"