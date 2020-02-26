import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


# load_sentences函数用于加载数据
# 数据格式为三维[[[]]],形状构成是（语句，语句中的构成单元，构成单元的内容:字和标签(['大', 'I-SGN'])
# 返回值sentences的数据格式为：
# [[['无', 'O'], ['长', 'B-Dur'], ['期', 'I-Dur'], ['0', 'O'], ['0', 'O'], ['0', 'O'], ['年', 'O'], ['0', 'O'],
#  ['0', 'O'], ['月', 'O'], ['0', 'B-DRU'], ['0', 'I-DRU'], ['日', 'O'], ['出', 'O'], ['院', 'O'], ['记', 'O'],
#  ['录', 'O'], ['患', 'O'], ['者', 'O'], ['姓', 'O'], ['名', 'O'], ['：', 'O'], ['闫', 'O'], ['X', 'O'], ['X', 'O'],
#  ['性', 'O'], ['别', 'O'], ['：', 'O'], ['男', 'O'], ['年', 'O'], ['龄', 'O'], ['：', 'O'], ['0', 'O'], ['0', 'O'],
#  ['岁', 'O'], ['入', 'O'], ['院', 'O'], ['日', 'O'], ['期', 'O'], ['：', 'O'], ['0', 'O'], ['0', 'O'], ['0', 'O'],
#  ['0', 'O'], ['年', 'O'], ['0', 'O'], ['0', 'O'], ['月', 'O'], ['0', 'O'], ['0', 'O'], ['日', 'O'], ['0', 'O'],
#  ['0', 'O'], ['时', 'O'], ['0', 'O'], ['0', 'O'], ['分', 'O'], ['出', 'O'], ['院', 'O'], ['日', 'O'], ['期', 'O'],
#  ['：', 'O'], ['0', 'O'], ['0', 'O'], ['0', 'O'], ['0', 'O'], ['年', 'O'], ['0', 'O'], ['0', 'O'], ['月', 'O'],
#  ['0', 'B-DRU'], ['0', 'I-DRU'], ['日', 'O'], ['0', 'O'], ['0', 'O'], ['时', 'O'], ['0', 'B-DRU'], ['0', 'I-DRU'],
#  ['分', 'O'], ['共', 'O'], ['住', 'O'], ['院', 'O'], ['0', 'O'], ['0', 'O'], ['天', 'O'], ['。', 'O']], [['入', 'O'],
#  ['院', 'O'], ['情', 'B-DRU'], ['况', 'I-DRU'], ['：', 'O'], ['女', 'O'], ['，', 'O'], ['0', 'O'], ['0', 'O'], ['岁', 'O'],
#  ['，', 'O'], ['以', 'O'], ['突', 'B-SYM'], ['发', 'I-SYM'], ['言', 'B-SYM'], ['语', 'I-SYM'], ['不', 'I-SYM'],
#  ['清', 'I-SYM'], ['0', 'O'], ['天', 'O'], ['，', 'O'], ['加', 'O'], ['重', 'O'], ['0', 'O'], ['天', 'O'], ['入', 'O'],
#  ['院', 'O'], ['。', 'O']]]


def load_sentences(path, lower, zeros):
    sentences = []
    sentence = []
    num = 0
    for line in open(path, 'r', encoding='utf8'):
        num += 1
        # 在这里将line中的数字(正则表达式是\d)转换成0
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        # 从该处开始对数据进行校验
        # 如果文档内容为空返回一个空语句
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []

        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                # 在该处分割每个单词，为该函数的主要操作内容
                word = line.split()
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    # 该模型的输入是加载好的数据sentences是load_sentences的返回值和tag_scheme是'iob'或'iobes'。简而言之就是增加语句的end和
    # 该模型的输入数据是:
    # [[['入', 'O'], ['院', 'O'], ['情', 'B-DRU'], ['况', 'I-DRU'], ['：', 'O'], ['女', 'O'], ['，', 'O'], ['0', 'O'],
    #   ['0', 'O'], ['岁', 'O'], ['，', 'O'], ['以', 'O'], ['突', 'B-SYM'], ['发', 'I-SYM'], ['言', 'B-SYM'], ['语', 'I-SYM'],
    #   ['不', 'I-SYM'], ['清', 'I-SYM'], ['0', 'O'], ['天', 'O'], ['，', 'O'], ['加', 'O'], ['重', 'O'], ['0', 'O'],
    #   ['天', 'O'], ['入', 'O'], ['院', 'O'], ['。', 'O'], ['入', 'O'], ['院', 'O'], ['情', 'B-DRU'], ['况', 'I-DRU'],
    #   ['：', 'O'], ['患', 'O'], ['者', 'O'], ['以', 'O'], ['腰', 'O'], ['痛', 'O'], ['伴', 'O'], ['双', 'B-REG'],
    #   ['下', 'I-REG'], ['肢', 'I-REG'], ['疼', 'B-SYM'], ['痛', 'I-SYM'], ['半', 'O'], ['年', 'O'], ['，', 'O'], ['加', 'O'],
    #   ['重', 'O'], ['0', 'O'], ['0', 'O'], ['余', 'O'], ['天', 'O'], ['为', 'O'], ['主', 'O'], ['诉', 'O'], ['入', 'O'],
    #   ['院', 'O'], ['。', 'O']]]
    #
    # 该模型的返回值是经过修改的传入的sentences,该处牵扯python的语法，具体内容见https://blog.csdn.net/nathan_yo/article/details/98639051
    # 该模型更改后的数据为：
    # [[['入', 'O'], ['院', 'O'], ['情', 'B-DRU'], ['况', 'E-DRU'], ['：', 'O'], ['女', 'O'], ['，', 'O'], ['0', 'O'],
    #   ['0', 'O'], ['岁', 'O'], ['，', 'O'], ['以', 'O'], ['突', 'B-SYM'], ['发', 'E-SYM'], ['言', 'B-SYM'], ['语', 'I-SYM'],
    #   ['不', 'I-SYM'], ['清', 'E-SYM'], ['0', 'O'], ['天', 'O'], ['，', 'O'], ['加', 'O'], ['重', 'O'], ['0', 'O'],
    #   ['天', 'O'], ['入', 'O'], ['院', 'O'], ['。', 'O'], ['入', 'O'], ['院', 'O'], ['情', 'B-DRU'], ['况', 'E-DRU'],
    #   ['：', 'O'], ['患', 'O'], ['者', 'O'], ['以', 'O'], ['腰', 'O'], ['痛', 'O'], ['伴', 'O'], ['双', 'B-REG'],
    #   ['下', 'I-REG'], ['肢', 'E-REG'], ['疼', 'B-SYM'], ['痛', 'E-SYM'], ['半', 'O'], ['年', 'O'], ['，', 'O'], ['加', 'O'],
    #   ['重', 'O'], ['0', 'O'], ['0', 'O'], ['余', 'O'], ['天', 'O'], ['为', 'O'], ['主', 'O'], ['诉', 'O'], ['入', 'O'],
    #   ['院', 'O'], ['。', 'O']]]
    # 这里i 和 s分别取字符和标记
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            # 该处用于检查错误，是否全是IOB标签B-
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """

    # 按照频率构造字典chars取sentences中的最数组最内单元的第一个数组作为结果，本质上就是最开始输入的经过处理替换数字的文字字符串。
    # ['入', '院', '情', '况', '：', '女', '，', '0', '0', '岁', '，', '以', '突', '发', '言', '语', '不',
    # '清', '0', '天', '，', '加', '重', '0', '天', '入', '院', '。', '入', '院', '情', '况', '：', '患',
    # '者', '以', '腰', '痛', '伴', '双', '下', '肢', '疼', '痛', '半', '年', '，', '加', '重', '0', '0',
    #  '余', '天', '为', '主', '诉', '入', '院', '。']
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # 在此处获取字及对应出现的频率
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    # 这里将无重复的字符传入create_mapping构件字典，传入数据是{'无': 1754, '长': 68, '期': 179,...}
    # char_to_id
    # {'<PAD>': 0, '<UNK>': 1, '0': 2, '入': 3, '院': 4, '，': 5, '天': 6, '。': 7, '以': 8, '况': 9, '加': 10,
    #   '情': 11, '痛': 12, '重': 13, '：': 14, '下': 15, '不': 16, '为': 17, '主': 18, '伴': 19, '余': 20, '半': 21,
    #   '双': 22, '发': 23, '女': 24, '岁': 25, '年': 26, '患': 27, '清': 28, '疼': 29, '突': 30, '者': 31, '肢': 32,
    #   '腰': 33, '言': 34, '诉': 35, '语': 36}
    #  id_to_char
    # {0: '<PAD>', 1: '<UNK>', 2: '0', 3: '入', 4: '院', 5: '，', 6: '天', 7: '。', 8: '以', 9: '况', 10: '加',
    #  11: '情', 12: '痛', 13: '重', 14: '：', 15: '下', 16: '不', 17: '为', 18: '主', 19: '伴', 20: '余', 21: '半',
    #  22: '双', 23: '发', 24: '女', 25: '岁', 26: '年', 27: '患', 28: '清', 29: '疼', 30: '突', 31: '者', 32: '肢',
    #  33: '腰', 34: '言', 35: '诉', 36: '语'}
    char_to_id, id_to_char = create_mapping(dico)
    # print("Found %i unique words (%i in total)" % (
    #    len(dico), sum(len(x) for x in chars)
    # ))
    #    这里dico{'入': 4, '院': 4, '情': 2, '况': 2, '：': 2, '女': 1, '，': 4, '0': 6, '岁': 1, '以': 2, '突': 1,
    #    '发': 1, '言': 1, '语': 1, '不': 1, '清': 1, '天': 3, '加': 2, '重': 2, '。': 2, '患': 1, '者': 1, '腰': 1,
    #    '痛': 2, '伴': 1, '双': 1, '下': 1, '肢': 1, '疼': 1, '半': 1, '年': 1, '余': 1, '为': 1, '主': 1, '诉': 1,
    #    '<PAD>': 10000001, '<UNK>': 10000000}

    return dico, char_to_id, id_to_char


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    # 该模型用于统计字符出现的频率
    # 输出为:
    # {'入': 4, '院': 4, '情': 2, '况': 2, '：': 2, '女': 1, '，': 4, '0': 6, '岁': 1, '以': 2, '突': 1, '发': 1,
    # '言': 1, '语': 1, '不': 1, '清': 1, '天': 3, '加': 2, '重': 2, '。': 2, '患': 1, '者': 1, '腰': 1, '痛': 2,
    # '伴': 1, '双': 1, '下': 1, '肢': 1, '疼': 1, '半': 1, '年': 1, '余': 1, '为': 1, '主': 1, '诉': 1}
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    # 按照频率构造字典
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # dico输出为:
    # {'入': 4, '院': 4, '情': 2, '况': 2, '：': 2, '女': 1, '，': 4, '0': 6, '岁': 1, '以': 2, '突': 1, '发': 1,
    # '言': 1, '语': 1, '不': 1, '清': 1, '天': 3, '加': 2, '重': 2, '。': 2, '患': 1, '者': 1, '腰': 1, '痛': 2,
    # '伴': 1, '双': 1, '下': 1, '肢': 1, '疼': 1, '半': 1, '年': 1, '余': 1, '为': 1, '主': 1, '诉': 1}
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    # 这里将无重复的字符传入create_mapping构件字典，传入数据是{'无': 1754, '长': 68, '期': 179,...}
    #    char_to_id
    # {'<PAD>': 0, '<UNK>': 1, '0': 2, '入': 3, '院': 4, '，': 5, '天': 6, '。': 7, '以': 8, '况': 9, '加': 10,
    #   '情': 11, '痛': 12, '重': 13, '：': 14, '下': 15, '不': 16, '为': 17, '主': 18, '伴': 19, '余': 20, '半': 21,
    #   '双': 22, '发': 23, '女': 24, '岁': 25, '年': 26, '患': 27, '清': 28, '疼': 29, '突': 30, '者': 31, '肢': 32,
    #   '腰': 33, '言': 34, '诉': 35, '语': 36}
    char_to_id, id_to_char = create_mapping(dico)
    # print("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in chars)))
    # {0: '<PAD>', 1: '<UNK>', 2: '0', 3: '入', 4: '院', 5: '，', 6: '天', 7: '。', 8: '以', 9: '况', 10: '加',
    #  11: '情', 12: '痛', 13: '重', 14: '：', 15: '下', 16: '不', 17: '为', 18: '主', 19: '伴', 20: '余', 21: '半',
    #  22: '双', 23: '发', 24: '女', 25: '岁', 26: '年', 27: '患', 28: '清', 29: '疼', 30: '突', 31: '者', 32: '肢',
    #  33: '腰', 34: '言', 35: '诉', 36: '语'}

    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    # 该函数用于将tag做mapping，
    # 其输入是:
    # 输出是:

    f = open('tag_to_id.txt', 'w', encoding='utf8')
    f1 = open('id_to_tag.txt', 'w', encoding='utf8')
    tags = []
    for s in sentences:
        ts = []
        for char in s:
            tag = char[-1]
            ts.append(tag)
        tags.append(ts)
    # tags [['O', 'O', 'B-DRU', 'E-DRU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-SYM',
    #       'E-SYM', 'B-SYM', 'I-SYM', 'I-SYM', 'E-SYM', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    #       'O', 'O', 'O', 'O', 'O', 'B-DRU', 'E-DRU', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    #       'B-REG', 'I-REG', 'E-REG', 'B-SYM', 'E-SYM', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    #       'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
    # tags1 = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    # dico {'O': 44, 'B-DRU': 2, 'E-DRU': 2, 'B-SYM': 3, 'E-SYM': 3, 'I-SYM': 2, 'B-REG': 1,
    #       'I-REG': 1, 'E-REG': 1}
    tag_to_id, id_to_tag = create_mapping(dico)
    # tag_to_id:
    # {'O': 0, 'B-SYM': 1, 'E-SYM': 2, 'B-DRU': 3, 'E-DRU': 4, 'I-SYM': 5, 'B-REG': 6, 'E-REG': 7,
    #  'I-REG': 8}
    # id_to_tag
    # {0: 'O', 1: 'B-SYM', 2: 'E-SYM', 3: 'B-DRU', 4: 'E-DRU', 5: 'I-SYM', 6: 'B-REG', 7: 'E-REG',
    # 8: 'I-REG'}
    # print("Found %i unique named entity tags" % len(dico))
    # 写入文档中
    for k, v in tag_to_id.items():
        f.write(k + ":" + str(v) + "\n")
    for k, v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    # 返回数据格式是:[[[]]] 三个标记分别是字符串,字符串对应的ID，经过结巴分词后词语的长度，字对应的标签的ID
    # [[['入', '院', '情', '况', '：', '女', '，', '0', '0', '岁', '，', '以', '突', '发', '言',
    # '语', '不', '清', '0', '天', '，', '加', '重', '0', '天', '入', '院', '。', '入', '院', '情',
    # '况', '：', '患', '者', '以', '腰', '痛', '伴', '双', '下', '肢', '疼', '痛', '半', '年',
    # '，', '加', '重', '0', '0', '余', '天', '为', '主', '诉', '入', '院', '。'],
    #  [3, 4, 11, 9, 14, 24, 5, 2, 2, 25, 5, 8, 30, 23, 34, 36, 16, 28, 2, 6, 5,
    #  10, 13, 2, 6, 3, 4, 7, 3, 4, 11, 9, 14, 27, 31, 8, 33, 12, 19, 22, 15, 32, 29,
    #  12, 21, 26, 5, 10, 13, 2, 2, 20, 6, 17, 18, 35, 3, 4, 7],
    #  [1, 3, 1, 3, 0, 0, 0, 1, 3, 0, 0, 0, 1, 3, 1, 3, 1, 3, 0, 0, 0, 1, 3, 0, 0, 1,
    #   3, 0, 1, 3, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 3, 1, 3, 1, 3,
    #   1, 3, 0, 1, 3, 0],
    #  [0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 5, 5, 2, 0, 0, 0,
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 6, 8, 7, 1, 2, 0, 0, 0, 0, 0,
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    # 这里将数据进行拼接返回，分别包含了word index（在字典中的序列数）word char index（单独按照数字的排列） 字符tag对应的标签的ID
    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        # print(sentences)
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        # 这里获取单词长度的标签，将一个单词标记为122223的格式，其中1是开始字符，2是中间字，3是结束，如果是单个字则直接标记为0
        #
        # 用结巴分词对单词进行分词，将单词转换歘122223格式例如seg_feature[1, 3, 1, 3, 0, 0, 0]对应的就是两个单词长度为2的和三个字
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])
    # 这里data的数据是string chars segs tags的合集
    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    "ext_emb_path是vec.txt文件用于将字符转化为预训练的词向量"
    # print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    # pretrained是加载的预训练的单词对应的vetctor
    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    # chars是传入的每句话
    # 这里如果chars是空，则赋值为0，这里将训练出来的分别派出成1,2,3,4,5的格式，结果是{'<PAD>': 0, '<UNK>': 1, '0': 2, '，': 3, '：': 4, '。': 5,
    # ....}
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
