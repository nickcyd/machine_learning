import matplotlib.pyplot as plt
from chinese_font import font

# 定义文本框和箭头格式
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """执行绘图功能，设置树节点的位置"""
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args, fontproperties=font)


def create_plot_before():
    fig = plt.figure(1, facecolor='white')
    # 清空绘图区
    fig.clf()

    # 绘制两个不同类型的树节点
    create_plot.ax1 = plt.subplot(111, frameon=False)

    # 绘制树节点，参数依次为节点名称、节点位置、上一个节点位置、文本框和箭头格式
    plot_node('决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)

    plt.show()


def get_num_leafs(my_tree):
    """获取叶子节点总数"""
    # 取出根节点
    num_leafs = 0
    first_str = list(my_tree.keys())[0]

    # 循环判断子节点
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 对于子节点为判断节点，继续递归寻找叶子节点
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1

    return num_leafs


def get_tree_depth(my_tree):
    """获取树的层数"""
    # 找到根节点
    max_depth = 0
    first_str = list(my_tree.keys())[0]

    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        # 如果子节点为判断节点，继续递归
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1

        # 调用中间变量获取树的层数
        if this_depth > max_depth: max_depth = this_depth

    return max_depth


def retrieve_tree(i):
    list_of_trees = [{
        'no surfacing': {
            0: 'no', 1: {
                'flippers': {
                    0: 'no', 1: 'yes'
                }
            }
        }
    },
        {'no surfacing': {
            0: 'no', 1: {
                'flippers': {
                    0: {
                        'head': {
                            0: 'no', 1: 'yes'
                        }
                    }
                }, 1: 'no',
            }
        }
        }
    ]

    return list_of_trees[i]


def get_num_and_get_depth():
    my_tree = retrieve_tree(0)
    print(my_tree)
    # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    num_leafs = get_num_leafs(my_tree)
    max_depth = get_tree_depth(my_tree)
    print(num_leafs)
    print(max_depth)


# TODO 补充解释
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """在父子节点填充文本信息"""
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    """"""
    # 计算宽与高
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.yOff)

    # 标记子节点属性值
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]

    # 减少 y 偏移
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.total_d
    for key in list(second_dict.keys()):
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))

    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.total_d


def create_plot(in_tree):
    """"""
    # 创建并清空画布
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    #
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)

    #
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))

    #
    plot_tree.xOff = -0.5 / plot_tree.total_w
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')

    plt.show()


def create_plot_run():
    """create_plot_2的运行函数"""
    my_tree = retrieve_tree(0)
    create_plot(my_tree)


if __name__ == '__main__':
    # create_plot()
    # get_num_and_get_depth()
    create_plot_run()
