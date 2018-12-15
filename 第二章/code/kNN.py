from numpy import *
import operator


def create_data_set():
    """
    初始化数据，其中group 数组的函数应该和标记向量 labels 的元素数目相同。
    :return: 返回训练样本集和标记向量
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 创建数据集
    labels = ['A', 'A', 'B', 'B']  # 创建标记

    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    对上述 create_data_set 的数据使用 k-近邻算法分类。
    :param in_x: 用于分类的向量
    :param data_set: 训练样本集
    :param labels: 标记向量
    :param k: 选择最近的数据的数目
    :return:
    """
    data_set_size = data_set.shape[0]  # 计算训练集的大小
    # 4

    # 距离计算
    # tile(inX, (a, b)) tile函将 inX 重复 a 行，重复 b 列
    # … - data_set 每个对应的元素相减，相当于欧式距离开平房内的减法运算
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    '''
       [[-1.  -1.1]
        [-1.  -1. ]
        [ 0.   0. ]
        [ 0.  -0.1]]
    '''

    # 对 diff_mat 内部的每个元素平方
    sq_diff_mat = diff_mat ** 2
    '''
        [[1.   1.21]
        [1.   1.  ]
        [0.   0.  ]
        [0.   0.01]]
    '''

    # sum(axis=0) 每列元素相加，sum(axis=1) 每行元素相加
    sq_distances = sq_diff_mat.sum(axis=1)
    # [2.21 2.   0.   0.01]

    # 每个元素开平方求欧氏距离
    distances = sq_distances ** 0.5
    # [1.48660687 1.41421356 0.         0.1       ]

    # argsort函数返回的是数组值从小到大的索引值
    sorted_dist_indicies = distances.argsort()
    # [2 3 1 0]

    # 选择距离最小的 k 个点
    class_count = {}  # type:dict
    for i in range(k):
        # 取出前 k 个对应的标签
        vote_i_label = labels[sorted_dist_indicies[i]]
        # 计算每个类别的样本数
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # operator.itemgetter(0) 按照键 key 排序，operator.itemgetter(1) 按照值 value 排序
    # reverse 倒序取出频率最高的分类
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # [('B', 2), ('A', 1)]

    # 取出频率最高的分类结果
    classify_result = sorted_class_count[0][0]

    return classify_result


def file2matrix(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        # 获取文件的行数
        array_0_lines = fr.readlines()  # type:list
        number_of_lines = len(array_0_lines)

        # 创建以零填充的的 NumPy 矩阵，并将矩阵的另一维度设置为固定值3
        return_mat = zeros((number_of_lines, 3))  # 创建一个1000行3列的0零矩阵

        # 解析文件数据到列表
        class_label_vector = []  # 把结果存储成列向量
        index = 0

        # 书本内容(报错)
        # for line in fr.readlines():
        #     line = line.strip()
        #     list_from_line = line.split("\t")
        #     return_mat[index, :] = list_from_line[0:3]
        #     class_label_vector.append(int(list_from_line[-1]))
        #     index += 1

        # 自己编写
        for line in array_0_lines:
            line = line.strip()
            list_from_line = line.split("\t")
            # return_mat 存储每一行数据的特征值
            return_mat[index, :] = list_from_line[0:3]

            # 通过数据的标记做分类
            if list_from_line[-1] == "didntLike":
                class_label_vector.append(int(1))
            elif list_from_line[-1] == "smallDoses":
                class_label_vector.append(int(2))
            elif list_from_line[-1] == "largeDoses":
                class_label_vector.append(int(3))
            index += 1

    return return_mat, class_label_vector


def scatter_diagram(dating_data_mat, dating_labels, diagram_type=1):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # windows下配置 font 为中文字体
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    # mac下配置 font 为中文字体
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

    # 通过 dating_labels 的索引获取不同分类在矩阵内的行数
    index = 0
    index_1 = []
    index_2 = []
    index_3 = []
    for i in dating_labels:
        if i == 1:
            index_1.append(index)
        elif i == 2:
            index_2.append(index)
        elif i == 3:
            index_3.append(index)
        index += 1

    # 对不同分类在矩阵内不同的行数构造每个分类的矩阵
    type_1 = dating_data_mat[index_1, :]
    type_2 = dating_data_mat[index_2, :]
    type_3 = dating_data_mat[index_3, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)  # 就是1行一列一张画布一张图，

    if diagram_type == 1:
        # 通过对特征0、1比较的散点图
        type_1 = ax.scatter(type_1[:, 0], type_1[:, 1], c='red')
        type_2 = ax.scatter(type_2[:, 0], type_2[:, 1], c='blue')
        type_3 = ax.scatter(type_3[:, 0], type_3[:, 1], c='green')
        plt.xlabel('每年的飞行里程数', fontproperties=font)
        plt.ylabel('玩视频游戏所耗时间百分比', fontproperties=font)

    elif diagram_type == 2:
        # 通过对特征1、2比较的散点图
        type_1 = ax.scatter(type_1[:, 1], type_1[:, 2], c='red')
        type_2 = ax.scatter(type_2[:, 1], type_2[:, 2], c='blue')
        type_3 = ax.scatter(type_3[:, 1], type_3[:, 2], c='green')
        plt.xlabel('玩视频游戏所耗时间百分比', fontproperties=font)
        plt.ylabel('每周所消费的冰淇淋公升数', fontproperties=font)

    elif diagram_type == 3:
        # 通过对特征0、2比较的散点图
        type_1 = ax.scatter(type_1[:, 0], type_1[:, 2], c='red')
        type_2 = ax.scatter(type_2[:, 0], type_2[:, 2], c='blue')
        type_3 = ax.scatter(type_3[:, 0], type_3[:, 2], c='green')
        plt.xlabel('每年的飞行里程数', fontproperties=font)
        plt.ylabel('每周所消费的冰淇淋公升数', fontproperties=font)

    plt.legend((type_1, type_2, type_3), ('不喜欢的人', '魅力一般的人', '极具魅力的人'), loc=4, prop=font)
    plt.show()


def auto_norm(data_set):
    # min(0)使得函数从列中选取最小值，min(1)使得函数从行中选取最小值
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals

    # 获取 data_set 的总行数
    m = data_set.shape[0]

    # 特征值相除
    # 相当于公式里的old_value-min
    # tile函数相当于将 min_vals 重复 m 行，重复1列
    norm_data_set = data_set - tile(min_vals, (m, 1))
    # 相当于公式里的(old_value-min)/(max-min)
    norm_data_set = norm_data_set / tile(ranges, (m, 1))

    return norm_data_set, ranges, min_vals


def dating_class_test():
    import os

    # 测试样本比率
    ho_ratio = 0.20

    # 读取文本数据
    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dating_test_set.txt')
    dating_data_mat, dating_labels = file2matrix(filename)

    # 对数据归一化特征值处理
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)

    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0

    for i in range(num_test_vecs):
        # 因为你的数据本来就是随机的，所以直接选择前20%的数据作为测试数据
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)

        if classifier_result != dating_labels[i]: error_count += 1

    # print("the total error rate is: {}".format(error_count / float(num_test_vecs)))
    # the total error rate is: 0.08


def main():
    import os

    group, labels = create_data_set()
    classify0([0, 0], group, labels, 3)

    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dating_test_set.txt')
    dating_data_mat, dating_labels = file2matrix(filename)

    # 需要画图演示开启
    '''
    diagram_type = 1, 比较特征(0, 1);
    diagram_type = 2, 比较特征(1, 2);
    diagram_type = 3, 比较特征(0, 2)
    '''
    # scatter_diagram(dating_data_mat, dating_labels, diagram_type=1)

    auto_norm(dating_data_mat)


def classify_person():
    import os

    result_list = ['讨厌', '有点喜欢', '非常喜欢']

    ff_miles = float(input("每年的出行公里数(km)？例如：1000\n"))
    percent_tats = float(input("每年玩游戏的时间占比(.%)？例如：10\n"))
    ice_cream = float(input("每年消费多少零食(kg)？例如：1\n"))

    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dating_test_set.txt')
    dating_data_mat, dating_labels = file2matrix(filename)

    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)

    in_arr = array([ff_miles, percent_tats, ice_cream])

    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)

    print("你可能对他/她的印象：\n{}".format(result_list[classifier_result - 1]))


if __name__ == '__main__':
    main()
    dating_class_test()
    classify_person()
