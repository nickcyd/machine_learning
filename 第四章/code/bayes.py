def load_data_set():
    """创建实验样本"""
    # 实验样本
    posting_list = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 样本标记
    class_vec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论

    return posting_list, class_vec


def create_vocab_list(data_set):
    """创建词汇表"""
    vocab_set = set([])

    # 创建不含重复单词的词汇表
    for document in data_set:
        vocab_set = vocab_set | set(document)

    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """创建文档向量"""
    # 创建一个和词汇表等长且元素全为0的列表
    return_vec = [0] * len(vocab_list)

    # 遍历文档
    for word in input_set:
        # 如果单词出现在词汇表中，则将0替换成1
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('{} 不在词汇表中'.format(word))

    return return_vec


def set_of_words2vec_run():
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    print(my_vocab_list)
    # ['park', 'ate', 'licks', 'steak', 'please', 'love', 'is', 'him', 'maybe', 'how', 'dog', 'has', 'food',
    # 'dalmation', 'I', 'my', 'stop', 'worthless', 'help', 'garbage', 'stupid', 'quit', 'cute', 'mr', 'buying',
    # 'posting', 'not', 'flea', 'problem', 'so', 'to', 'take']
    return_vec = set_of_words2vec(my_vocab_list, list_of_posts[0])
    print(return_vec)
    # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1]


def train_nb_0(train_matrix, train_category):
    """计算每个类别中的文档数目"""
    import numpy

    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    # 计算属于侮辱性文档的概率，因为侮辱为1，非侮辱为0，使用 sum 函数可以计算
    p_abusive = sum(train_category) / float(num_train_docs)

    # 初始化概率(修改前)
    '''
    p0_num = numpy.zeros(num_words)
    p1_num = numpy.zeros(num_words)
    p0_denom = 0
    p1_denom = 0
    '''
    # (修改后)由于之前计算每个元素的向量积，因此只要有一个元素为0输出结果只为0
    p0_num = numpy.ones(num_words)
    p1_num = numpy.ones(num_words)
    p0_denom = 2
    p1_denom = 2

    # 遍历文档矩阵
    for i in range(num_train_docs):
        # 计算文档中侮辱性的句子总和以及单词在词汇表中的计数总和
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        # 计算文档中非侮辱性的句子总和以及单词在词汇表中的计数总和
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    # 计算每个元素除以该类别中的总词数，即矩阵中的每个元素除以int(p1_denom)
    # 修改前
    '''
    p1_vect = math.log(p1_num / p1_denom)
    p0_vect = math.log(p0_num / p0_denom)
    '''
    # 修改后，由于多个较小的数相乘四舍五入结果会输出0，使用 log 计算则不会
    p1_vect = numpy.log(p1_num / p1_denom)
    p0_vect = numpy.log(p0_num / p0_denom)

    return p0_vect, p1_vect, p_abusive


def train_nb_0_run():
    # 创建词汇表
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)

    # 把数据集中的所有单词转化成0，1的值，即不在词汇表中、在词汇表中
    train_mat = []
    for post_in_doc in list_of_posts:
        # 比较词汇表与句子，在词汇表中则词汇表中单词数+1
        train_mat.append(set_of_words2vec(my_vocab_list, post_in_doc))

    # 每个元素除以该类别总词数，返回每个类别的条件概率
    p0_v, p1_v, p_ab = train_nb_0(train_mat, list_classes)
    print(p0_v)
    print(p1_v)
    print(p_ab)


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class1):
    """构建完整的分类器"""
    import math
    p1 = sum(vec2_classify * p1_vec) + math.log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + math.log(1 - p_class1)

    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb_run():
    import numpy

    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)

    train_mat = []
    for post_in_doc in list_of_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, post_in_doc))

    p0_v, p1_v, p_ab = train_nb_0(train_mat, list_classes)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = numpy.array(set_of_words2vec(my_vocab_list, test_entry))
    print('{} 分类结果 {}'.format(test_entry, classify_nb(this_doc, p0_v, p1_v, p_ab)))

    # test_entry = ['stupid', 'garbage']
    # this_doc = numpy.array(set_of_words2vec(my_vocab_list, test_entry))
    # print('{} 分类结果 {}'.format(test_entry, classify_nb(this_doc, p0_v, p1_v, p_ab)))


def bag_of_words2vec(vocal_list, input_set):
    """词袋模型"""
    return_vec = [0] * len(vocal_list)

    # 遍历词汇表有对应的单词计数加1
    for word in input_set:
        if word in vocal_list:
            return_vec[vocal_list.index(word)] += 1

    return return_vec


def text_parse(my_text):
    import re

    # 使用正则切分句子，分隔符是除单词、数字外的任意字符串
    # 加上[]是避免报警告：不能用零长度字符串切分
    reg_ex = re.compile('[\W*]', )
    list_of_tokens = reg_ex.split(my_text)

    # 只返回长度大于0的单词并把切分的单词全小写
    tok_list = [tok.lower() for tok in list_of_tokens if len(tok) > 2]

    return tok_list


def text_parse_run():
    import os

    email_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data-set/email/ham/6.txt')
    with open(email_filename, 'r') as fr:
        email_text = fr.read()
        tok_list = text_parse(email_text)

        print(tok_list)


def span_test():
    import os
    import numpy

    # 导入并解析文件
    doc_list = []
    class_list = []
    full_text = []
    email_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data-set/email/{}/{}.txt')
    for i in range(1, 26):
        # 切割垃圾邮件文本并添加标记
        word_list = text_parse(open(email_filename.format('spam', i), errors='ignore').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        # 切割非垃圾邮件文本并添加标记
        word_list = text_parse(open(email_filename.format('ham', i), errors='ignore').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    # 构造词汇表
    vocab_list = create_vocab_list(doc_list)

    # 从50封邮件中随机取10封邮件当做测试集
    training_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(numpy.random.uniform(0, len(training_set)))
        print(rand_index)
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    # 创建词汇向量
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])

    # 通过词汇向量和向量标记得到垃圾邮件和非垃圾邮件概率
    p0_v, p1_v, p_spam = train_nb_0(numpy.array(train_mat), numpy.array(train_classes))

    # 测试集测试分类器
    error_count = 0
    for doc_index in test_set:
        word_vector = set_of_words2vec(vocab_list, doc_list[doc_index])
        if classify_nb(numpy.array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('错误率：{}'.format(float(error_count) / len(test_set)))


if __name__ == '__main__':
    # set_of_words2vec_run()
    # train_nb_0_run()
    # testing_nb_run()
    # text_parse_run()
    span_test()
