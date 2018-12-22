from math import log


def calc_shannon_ent(data_set):
    # 计算实例总数
    num_entries = len(data_set)
    label_counts = {}

    # 统计所有类别出现的次数
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    # 通过熵的公式计算香农熵
    shannon_ent = 0
    for key in label_counts:
        # 统计所有类别出现的概率
        prob = float(label_counts[key] / num_entries)
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def create_data_set():
    """构造我们之前对鱼鉴定的数据集"""
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']

    return data_set, labels


def shannon_run():
    data_set, labels = create_data_set()
    shannon_ent = calc_shannon_ent(data_set)
    print(shannon_ent)
    # 0.9709505944546686


def split_data_set(data_set, axis, value):
    # 对符合规则的特征进行筛选
    ret_data_set = []
    for feat_vec in data_set:
        # 选取某个符合规则的特征
        if feat_vec[axis] == value:
            # 如果符合特征则删掉符合规则的特征
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])

            # 把符合规则的数据写入需要返回的列表中
            ret_data_set.append(reduced_feat_vec)

    return ret_data_set


def split_data_run():
    my_data, labels = create_data_set()
    ret_data_set = split_data_set(my_data, 0, 1)
    print(ret_data_set)
    # [[1, 'yes'], [1, 'yes'], [0, 'no']]


def choose_best_feature_to_split(data_set):
    # 计算特征数量
    num_features = len(data_set[0]) - 1
    # 计算原始数据集的熵值
    base_entropy = calc_shannon_ent(data_set)

    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]

        # 遍历某个特征下的所有特征值按照该特征的权重值计算某个特征的熵值
        unique_vals = set(feat_list)
        new_entropy = 0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)

        # 计算最好的信息增益
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def choose_best_feature_run():
    data_set, _ = create_data_set()
    best_feature = choose_best_feature_to_split(data_set)
    print(best_feature)
    # 0


def majority_cnt(class_list):
    """
    对 class_list 做分类处理，类似于 k-近邻算法的 classify0 方法
    返回出现最多的标记
    """
    import operator

    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    # 取出数据集中的标记信息
    class_list = [example[-1] for example in data_set]

    # 如果数据集中的标记信息完全相同则结束递归
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # TODO 补充解释
    # 如果最后使用了所有的特征无法将数据集划分成仅包含唯一类别的分组
    # 因此我们遍历完所有特征时返回出现次数最多的特征
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    # 通过计算熵值返回最适合划分的特征
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]

    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    # 取出最合适特征对应的值并去重即找到该特征对应的所有可能的值
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)

    # 遍历最适合特征对应的所有可能值，并且对这些可能值继续生成子树
    # 类似于 choose_best_feature_to_split 方法
    for value in unique_vals:
        sub_labels = labels[:]
        # 对符合该规则的特征继续生成子树
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)

    return my_tree


def create_tree_run():
    my_dat, labels = create_data_set()
    my_tree = create_tree(my_dat, labels)
    print(my_tree)
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]

    # 使用 index 方法查找当前列表中第一个匹配 first_str 变量的元素
    feat_index = feat_labels.index(first_str)

    # 递归到达叶子节点
    class_label = None
    for key in list(second_dict.keys()):
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]

    return class_label


def classify_run():
    from 第三章.code.tree_plotter import retrieve_tree
    my_dat, labels = create_data_set()
    my_tree = retrieve_tree(0)
    class_label = classify(my_tree, labels, [1, 0])
    print(class_label)
    # 'no'


def store_tree(input_tree, filename):
    import pickle

    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    import pickle

    with open(filename, 'rb') as fr:
        return pickle.load(fr)


def store_grab_tree_run():
    import os

    from 第三章.code.tree_plotter import retrieve_tree
    my_tree = retrieve_tree(0)

    classifier_storage_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'classifierStorage.txt')

    store_tree(my_tree, classifier_storage_filename)
    my_tree = grab_tree(classifier_storage_filename)
    print(my_tree)
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


def create_lenses_tree():
    import os

    lenses_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lenses.txt')
    with open(lenses_filename, 'rt', encoding='utf-8') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = create_tree(lenses, lenses_labels)

        return lenses_tree


def plot_lenses_tree():
    from 第三章.code import tree_plotter

    lenses_tree = create_lenses_tree()
    tree_plotter.create_plot(lenses_tree)


if __name__ == '__main__':
    # shannon_run()
    # split_data_run()
    # choose_best_feature_run()
    # create_tree_run()
    # classify_run()
    # store_grab_tree_run()
    plot_lenses_tree()
