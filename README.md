# learn_GloVe
自己写一个GloVe的训练程序  
  
# 理论要点
GloVe的基本思路，是通过联合矩阵的特性，来训练嵌入矩阵的。  
共现矩阵的最重要特性：  
ratio(i,j,k)的值与词之间的相关性有密切关系。  
令：  
P(i,k) = X(i,k) / X(i)  
P(j,k) = X(j,k) / X(i)  
ratio(i,j,k) = P(i,k) / P(j,k)  
则：  
若i,k相关、且j,k相关 —— ratio(i,j,k)接近1  
若i,k不相关、且j,k不相关 —— ratio(i,j,k)接近1  
若i,k相关，j,k不相关 —— ratio(i,j,k)很大  
若i,k不相关，j,k相关 —— ratio(i,j,k)很小  
  
利用这一个特性，构造损失函数：  
J = sum(log(P_ij) - dot(v_i,v_j))  
展开P(i,j)推导  
J = sum(dot(v_i,v_j) + b_i + b_j - log(X_ij))  
加上权重函数f(X_ij)  
J = sum(f(X_ij) * (dot(v_i,v_j) + b_i + b_j - log(X_ij)))  
其中，权重函数为：  
f(x) = power((x / x_max), 0.75)  if x < x_max  
f(x) = 1                         if x >= x_max  
  
接下来的任务，就是构造联合矩阵，然后通过训练，使损失函数最小。  
  
