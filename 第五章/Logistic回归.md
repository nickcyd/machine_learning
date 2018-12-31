[TOC]

# Logistic回归

## 博客园地址：[https://www.cnblogs.com/chenyoude/](https://www.cnblogs.com/chenyoude/)
## git 地址：[https://github.com/nickcyd/machine_learning](https://github.com/nickcyd/machine_learning)
## 微信：a1171958281
**代码中涉及的数学公式可以自己下载 Typora 这款软件后，把内容复制到.md文件内通过 Typora 打开**

## Logistic 回归

### 本章内容
* Sigmoid 函数和 Logistic 回归分类器
* 最优化理论初步
* 梯度下降最优化算法
* 数据中的缺失项处理


### 回归算法
* 回归算法：假设现在有一些数据点，我们用一条直线对这些点进行拟合（该线称为最佳拟合直线），这个拟合过程就称作回归。与分类算法一样同属于监督学习。


### Logistic 回归的一般过程
1. 收集数据：采用任意方法收集数据。
2. 准备数据：由于需要进行距离计算，因此要求数据类型为数值型。
3. 分析数据：采用任意方法对数据进行分析。
4. 训练算法：大部分时间讲用于训练，训练的目的是为了找到最佳的分类回归系数。
5. 测试算法：一旦训练步骤完成，分类将会很快。
6. 使用算法：基于训练好的回归系数对这些数值进行简单的回归计算，判定他们属于哪个类别，在此基础上做一些其他分析工作。


### Logistic的优缺点
* 优点：计算代价不高，易于理解和实现。
* 缺点：容易欠拟合，分类精度可能不高。
* 适用数据类型：数值型和标称型。


## 基于 Logistic 回归和 Sigmoid 函数的分类

### Sigmoid 函数
* 海维赛德阶跃函数(单位阶跃函数)：输出只有0或1的函数，并且0到1的过程属于跳跃过程，即非0即1。
* Sigmoid 函数：x=0时，sigmoid 值为0.5；随着 x 的增大，对应值将逼近1；随着 x 的减小，对应值将逼近0。
* Sigmoid 函数公式：$\sigma(z)={\frac{1}{1+e^{-z}}}$。


### Logistic 回归分类器
* Logistic 回归分类器：我们在每个特征上都乘以一个回归系数  *之后详细介绍*，然后把所有的结果值相加，将这个总和代入 sigmoid 函数，进而得到一个范围在0\~1之间的数值。大于0.5的数据被分入1类，小于0.5即被归入0类。


### 图5-1 两种坐标尺度下的 Sigmoid 函数图
![两种坐标尺度下的 Sigmoid 函数图](/Users/mac/Desktop/machine_learning/第五章/img/两种坐标尺度下的 Sigmoid 函数图.png)
* 通过图5-1 下面一张图可以看出，如果横坐标的尺度足够大，在 x=0出 sigmoid 函数看起来很像阶跃函数。


## 基于最优化方法的最佳回归系数确定
* Sigmoid函数的输入记为 z，可由该公式得出：$z=w_0x_0+w_1x_1+w_2x_2+\cdots+w_nx_n$。
* 上述公式向量写法：$z=w^Tx​$  *向量 x 是分类器的输入数据，向量 w 是我们需要找的最佳参数（系数）*。

### 梯度上升法

* 梯度上升法：沿着函数的梯度方向探寻某函数的最大值。即求函数的最大值。
* 如果梯度记为$\nabla$，则函数$f(x,y)$的梯度公式：$\nabla f(x,y)=\begin{pmatrix} {\frac{\part f(x,y)}{\part x}} \\ {\frac{\part f(x,y)}{\part y}} \\ \end{pmatrix}$。
* ${\frac{\part f(x,y)}{\part x}}$：沿 x 的方向移动${\frac{\part f(x,y)}{\part x}}$，函数$f(x,y)$必须要在待计算的点上有定义并且可微。
* ${\frac{\part f(x,y)}{\part y}}$：沿 x 的方向移动${\frac{\part f(x,y)}{\part y}}$，函数$f(x,y)$必须要在待计算的点上有定义并且可微。


### 图5-2 梯度上升图
![梯度上升图](/Users/mac/Desktop/machine_learning/第五章/img/梯度上升图.png)
* 通过图5-2 可以看出梯度上升算法到达每个点后都会重新估计移动的方向。
* 梯度上升算法的迭代公式：$w:=w+\alpha \nabla_wf(w)$，该公式将一直被迭代执行，直至达到某个停止条件为止。
* $\alpha$：移动量的大小，称为步长。


### 梯度下降算法
* 梯度下降算法：沿着函数的梯度方向探寻某函数的最小值。即求函数的最小值。
* 梯度下降算法的迭代公式：$w:=w-\alpha \nabla_wf(w)$


## 训练算法：使用梯度上升找到最佳参数

### 图5-3 数据集图

![数据集图](/Users/mac/Desktop/machine_learning/第五章/img/数据集图.png)
* 图5-3中有100个样本点，每个点包含两个数值型特征 X1和X2。


### 梯度上升算法的伪代码

```python
每个回归系数初始化为1
重复 R 次：
    计算整个数据集的梯度
    使用 alpha*gradient 更新回归系数的向量
    返回回归系数 
    
```


### 程序5-1 Logistic 回归梯度上升优化算法
```python
import os
import numpy as np
import matplotlib.pyplot as plt
from path_settings import machine_learning_PATH

data_set_path = os.path.join(machine_learning_PATH, '第五章/data-set')
testSet_path = os.path.join(data_set_path, 'testSet.txt')
horseColicTraining_path = os.path.join(data_set_path, 'horseColicTraining.txt')
horseColicTest_path = os.path.join(data_set_path, 'horseColicTest.txt')


def load_data_set():
    """导入数据集"""
    data_mat = []
    label_mat = []

    # 循环导入.txt文本数据构造成列表
    fr = open(testSet_path)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))

    return data_mat, label_mat


def sigmoid(in_x):
    return 1 / (1 + np.exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    # 生成特征矩阵
    data_matrix = np.mat(data_mat_in)
    # 生成标记矩阵并反置
    label_mat = np.mat(class_labels).transpose()

    # 计算data_matrix的行列
    m, n = np.shape(data_matrix)

    # 设置移动的步长为0.001
    alpha = 0.001
    # 设置最大递归次数500次
    max_cycles = 500

    # 初始化系数为1*3的元素全为1的矩阵
    weights = np.ones((n, 1))

    # 循环迭代梯度上升算法
    for k in range(max_cycles):
        # 计算真实类别与预测类别的差值
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        
        # 调整回归系数
        weights = weights + alpha * data_matrix.transpose() * error

    return weights


def test_grad_ascent():
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat)
    print(weights)
    """
    [[ 4.12414349]
     [ 0.48007329]
     [-0.6168482 ]]
    """


if __name__ == '__main__':
    test_grad_ascent()

```


### 分析数据：画出决策边界
* 该节将通过代码画出决策边界


### 程序5-2 画出数据集和 Logistic 回归最佳拟合直线的函数
```python
def plot_best_fit(wei):
    # getA==np.asarrayz(self)
    # 使用__class__.__name__为了判断是梯度上升和随机梯度上升
    if wei.__class__.__name__ == 'matrix':
        weights = wei.getA()
    elif wei.__class__.__name__ == 'ndarray':
        weights = wei
    else:
        weights = wei

    data_mat, label_mat = load_data_set()

    # 把特征集转换成数组
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]

    # 循环数据集分类
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 0.1是步长
    x = np.arange(-3, 3, 0.1)
    # 假设 sigmoid 函数为0，并且这里的 x，y 相当于上述的 x1和x2即可得出 y 的公式
    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def test_plot_best_fit():
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat)
    plot_best_fit(weights)


if __name__ == '__main__':
    # test_grad_ascent()
    test_plot_best_fit()
    
```

### 图5-4 梯度上升算法500次迭代后的结果

![梯度上升算法500次迭代后的结果](/Users/mac/Desktop/machine_learning/第五章/img/梯度上升算法500次迭代后的结果.png)
* 通过图5-4 可以看出我们只分错了2-4个点。


### 训练算法：随机梯度上升
* 梯度上升法每次更新回归系数时都需要遍历整个数据集，如果样本或者特征数过多就应该考虑使用随机梯度上升算法。
* 随机梯度上升：一次仅用一个样本点来更新回归系数，不需要重新读取整个数据集。


### 随机梯度上升算法伪代码
```python
所有回归系数初始化为1
对数据集中每个样本
    计算该样本的梯度
    使用 alpha*gradient 更新回归系数值
返回回归系数值

```

### 程序5-3 随机梯度上升算法
```python
def stoc_grad_ascent0(data_matrix, class_labels):
    """随机梯度上升算法"""
    m, n = np.shape(data_matrix)

    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # 使用 sum 函数得出一个值，只用计算一次
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]

    return weights


def test_stoc_grad_ascent0():
    data_arr, label_mat = load_data_set()
    weights = stoc_grad_ascent0(np.array(data_arr), label_mat)
    plot_best_fit(weights)


if __name__ == '__main__':
    # test_grad_ascent()
    # test_plot_best_fit()
    test_stoc_grad_ascent0()

```
* 梯度上升和随机梯度上升：从代码中我们可以看到前者变量 h 和误差 error 都是向量，而后者全是数值；前者是矩阵转换，后者则是 numpy 数组。


### 图5-5 随机梯度上升算法图

![随机梯度上升算法图](/Users/mac/Desktop/machine_learning/第五章/img/随机梯度上升算法图.jpg)
* 图5-5可以看出随机梯度上升算法的最佳拟合直线并非最佳分类线


### 程序5-4 改进的随机梯度上升算法
```python
def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    """改进随机梯度上升算法，默认迭代150次"""
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            # 每次迭代减小 alpha 的值，但最小为0.01，确保新数据依然有影响。缓解系数波动的情况
            alpha = 4 / (1 + j + i) + 0.01

            # 随机选取值进行更新
            rand_index = int(np.random.uniform(0, len(data_index)))

            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]

            # 删除更新后的值
            del (data_index[rand_index])

    return weights


def test_stoc_grad_ascent1():
    data_arr, label_mat = load_data_set()
    weights = stoc_grad_ascent1(np.array(data_arr), label_mat)
    plot_best_fit(weights)


if __name__ == '__main__':
    # test_grad_ascent()
    # test_plot_best_fit()
    # test_stoc_grad_ascent0()
    test_stoc_grad_ascent1()

```

### 图5-6 改进随机梯度上升算法图
![改进随机梯度上升算法图](/Users/mac/Desktop/machine_learning/第五章/img/改进随机梯度上升算法图.jpg)
* 图5-6可以看出150次的跌打就能得到一条很好的分类线，而梯度上升算法需要迭代500次。


## 示例：从疝气病预测病马的死亡率

* 疝气病：描述马胃肠痛的术语
* 数据集中包含368个样本和28个特征，并且有30%的值是缺失的


### 示例：使用 Logistic 回归估计马疝病的死亡率
1. 收集数据：给定数据文件
2. 准备数据：用 Python 解析文本文件并填充缺失值
3. 分析数据：可视化并观察数据
4. 训练算法：使用优化算法，找到最佳的系数
5. 测试算法：观察错误率，根据错误率决定是否会退到训练阶段；改变迭代的次数和步长等参数来得到更好的回归系数
6. 使用算法：实现一个简单的程序来手机马的症状并输出预测结果


### 准备数据：处理数据中的缺失值
* 数据的获取是相当昂贵的，扔掉和重新获取都是不可取的
* 以下几种方法可以解决数据的缺失的问题
1. 使用可用特征的均值来填补缺失值
2. 使用特殊值来填补缺失值
3. 忽略有缺失值的样本
4. 使用相似样本的均值填补缺失值
5. 使用另外的机器学习算法预测缺失值
* 预处理第一件事：用0替代所有的缺失值，因为缺失值为0时回归系数的更新公式不会更新并且 sigmoid(0)=0.5，他对结果的预测不具有任何倾向性
* 预处理第二件事：对于数据标记缺失的数据舍弃，因为标记很难确定采用某个合适的值来替换。
* 预处理后的文件：对于原始数据文件可以去 http://archive.ics.uci.edu/ml/datasets/Horse+Colic 获取，此处只提供预处理之后的文件


### 测试算法：用 Logistic 回归进行分类
```python
def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colic_test():
    """马疝病造成马死亡概率预测"""
    fr_train = open(horseColicTraining_path)
    fr_test = open(horseColicTest_path)

    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        # 切分所有特征并把特征加入 line_arr 列表中
        curr_line = line.strip().split('\t')  # type:list
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        # 分开处理特征和标记
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    print(train_weights)

    error_count = 0
    num_test_vec = 0
    for line in fr_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')  # type:list
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))

        # 通过比较样本标记与输入系数与特征相乘值 sigmoid 函数得到的标记判断是否预测失误
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1

    error_rate = (float(error_count) / num_test_vec)
    print('测试集的错误率: {}'.format(error_rate))
    # 测试集的错误率: 0.373134328358209

    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0
    for k in range(num_tests):
        error_sum += colic_test()
    print('迭代 {} 次后平均错误率为: {}'.format(num_tests, error_sum / float(num_tests)))
    # 迭代 10 次后平均错误率为: 0.3656716417910448


if __name__ == '__main__':
    # test_grad_ascent()
    # test_plot_best_fit()
    # test_stoc_grad_ascent0()
    # test_stoc_grad_ascent1()
    multi_test()

```

## 完整代码logRegres.py

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from path_settings import machine_learning_PATH

data_set_path = os.path.join(machine_learning_PATH, '第五章/data-set')
testSet_path = os.path.join(data_set_path, 'testSet.txt')
horseColicTraining_path = os.path.join(data_set_path, 'horseColicTraining.txt')
horseColicTest_path = os.path.join(data_set_path, 'horseColicTest.txt')


def load_data_set():
    """导入数据集"""
    data_mat = []
    label_mat = []

    # 循环导入.txt文本数据构造成列表
    fr = open(testSet_path)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))

    return data_mat, label_mat


def sigmoid(in_x):
    """构造 sigmoid 函数"""
    return 1 / (1 + np.exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    """梯度上升算法"""
    # 生成特征矩阵
    data_matrix = np.mat(data_mat_in)
    # 生成标记矩阵并反置
    label_mat = np.mat(class_labels).transpose()

    # 计算data_matrix的行列
    m, n = np.shape(data_matrix)

    # 设置移动的步长为0.001
    alpha = 0.001
    # 设置最大递归次数500次
    max_cycles = 500

    # 初始化系数为1*3的元素全为1的矩阵
    weights = np.ones((n, 1))

    # 循环迭代梯度上升算法
    for k in range(max_cycles):
        # 计算真实类别与预测类别的差值
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)

        # 调整回归系数
        weights = weights + alpha * data_matrix.transpose() * error

    return weights


def test_grad_ascent():
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat)
    print(weights)
    """
    [[ 4.12414349]
     [ 0.48007329]
     [-0.6168482 ]]
    """


def plot_best_fit(wei):
    """画出被分割的数据集"""
    # getA==np.asarrayz(self)
    # 使用__class__.__name__为了判断是梯度上升和随机梯度上升
    if wei.__class__.__name__ == 'matrix':
        weights = wei.getA()
    elif wei.__class__.__name__ == 'ndarray':
        weights = wei
    else:
        weights = wei

    data_mat, label_mat = load_data_set()

    # 把特征集转换成数组
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]

    # 循环数据集分类
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 0.1是步长
    x = np.arange(-3, 3, 0.1)
    # 假设 sigmoid 函数为0，并且这里的 x，y 相当于上述的 x1和x2即可得出 y 的公式
    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def test_plot_best_fit():
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat)
    plot_best_fit(weights)


def stoc_grad_ascent0(data_matrix, class_labels):
    """随机梯度上升算法"""
    m, n = np.shape(data_matrix)

    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # 使用 sum 函数得出一个值，只用计算一次
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]

    return weights


def test_stoc_grad_ascent0():
    data_arr, label_mat = load_data_set()
    weights = stoc_grad_ascent0(np.array(data_arr), label_mat)
    plot_best_fit(weights)


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    """改进随机梯度上升算法，默认迭代150次"""
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            # 每次迭代减小 alpha 的值，但最小为0.01，确保新数据依然有影响。缓解系数波动的情况
            alpha = 4 / (1 + j + i) + 0.01

            # 随机选取值进行更新
            rand_index = int(np.random.uniform(0, len(data_index)))

            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]

            # 删除更新后的值
            del (data_index[rand_index])

    return weights


def test_stoc_grad_ascent1():
    data_arr, label_mat = load_data_set()
    weights = stoc_grad_ascent1(np.array(data_arr), label_mat)
    plot_best_fit(weights)


def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colic_test():
    """马疝病造成马死亡概率预测"""
    fr_train = open(horseColicTraining_path)
    fr_test = open(horseColicTest_path)

    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        # 切分所有特征并把特征加入 line_arr 列表中
        curr_line = line.strip().split('\t')  # type:list
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        # 分开处理特征和标记
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    print(train_weights)

    error_count = 0
    num_test_vec = 0
    for line in fr_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')  # type:list
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))

        # 通过比较样本标记与输入系数与特征相乘值 sigmoid 函数得到的标记判断是否预测失误
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1

    error_rate = (float(error_count) / num_test_vec)
    print('测试集的错误率: {}'.format(error_rate))
    # 测试集的错误率: 0.373134328358209

    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0
    for k in range(num_tests):
        error_sum += colic_test()
    print('迭代 {} 次后平均错误率为: {}'.format(num_tests, error_sum / float(num_tests)))
    # 迭代 10 次后平均错误率为: 0.3656716417910448


if __name__ == '__main__':
    # test_grad_ascent()
    # test_plot_best_fit()
    # test_stoc_grad_ascent0()
    # test_stoc_grad_ascent1()
    multi_test()

```

## 总结
* Logistic 回归：寻找一个非线性函数 Sigmoid 的最佳拟合参数。
* 求解过程：通过最优化算法（常用的梯度上升算法），通过简化梯度上升算法得到随机梯度上升算法
* 对缺失数据的处理：机器学习中最后只能更要的问题之一，主要还是取决于实际应用中的需求。


## 支持向量机 coding……



==尊重原创==
==可以伸出你的小手点个关注，谢谢！==

博客园地址：[https://www.cnblogs.com/chenyoude/](https://www.cnblogs.com/chenyoude/)
git 地址：[https://github.com/nickcyd/machine_learning](https://github.com/nickcyd/machine_learning)
微信：a1171958281