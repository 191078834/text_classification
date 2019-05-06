#!/usr/bin/python
# -*- coding: utf-8 -*- 
#Auther: WQM
#Time: 2019/3/26 9:13
import jieba
import jieba.analyse
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.externals import joblib
from utils.log import log
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
sys.path.append('../')


def jieba_cut(text, cut_all=False, HMM=False, userdict='dict.txt'):
    # 标准分词 是否为全模式
    if userdict:
        jieba.load_userdict(userdict)
    content = jieba.cut(text, cut_all=cut_all)
    #返回生成器 需要迭代
    if HMM:
        content = jieba_cut(text, HMM=True)
    return content

def jieba_cut_for_search(text, HMM=False):
    content = jieba.cut_for_search(text, HMM=HMM)
    # 返回生成器
    return content

def delete_word(text, stop_word_path='stop.txt'):
    word=''
    word_lists=[]
    if isinstance(text, str):
        lists = jieba_cut(text)
        for li in lists:
            word_lists.append(li)
        with open(stop_word_path, mode='r', encoding='utf-8') as e:
            stop_cons = e.readlines()
            for stop_word in stop_cons:
                stop_word = stop_word.strip('\n')
                if stop_word in word_lists:
                    word_lists.remove(stop_word)

        return  ''.join(word_lists).strip()
    else:
        return ''

def write_data(text, file_name):
    if text is None:
        pass
    else:
        with open(file_name, mode='a', encoding='utf-8') as e:
            e.write(text.strip() + '\t')
        log.info('写入成功')

def read_data(file_name, encoding='utf-8'):
    with open(file_name, mode='r', encoding=encoding) as e:
        content = e.read()
    return content

def get_counts(lists):

    counter = Counter(lists)
    return counter.most_common()

def get_TF(text):
    dicts={}
    cut_texts = jieba_cut(text)
    for cut_text in cut_texts:
        if cut_text in dicts.keys():
            dicts[cut_text] += 1
        else:
            dicts[cut_text] = 1
    return dicts

# 转换为向量矩阵
def toarray_vocabulary_and_dump_ocabulary(lists, vocabulary_path='vocabulary.word'):
    '''
    :param lists: ['for example word']
    :return: [[1 0 1]]
    '''
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(lists)
    vocabulary = vectorizer.vocabulary_
    if vocabulary_path:
        joblib.dump(vocabulary, vocabulary_path)
    tfidf_array = convert_vector_array(X)
    return tfidf_array

# 转换tf-idf向量矩阵
def convert_vector_array(x):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(x)  # 将词频矩阵X统计成TF-IDF值
    tfidf_array = tfidf.toarray()
    return tfidf_array

# 加载词典
def load_vocabulary(load_vocabulary_path):

    new_vectorizer = CountVectorizer(min_df=1, vocabulary=joblib.load(load_vocabulary_path))
    return new_vectorizer

# 单个矩阵转换向量矩阵
def text_toarray(new_vectorizer, text):
    '''
    :param new_vectorizer: 词袋库模型
    :param text: ['for example word']
    :return: [[0 1 2]]
    '''
    word_array = new_vectorizer.fit_transform(text)
    transformer = TfidfTransformer()
    b = transformer.fit_transform(word_array)  # 将词频矩阵X统计成TF-IDF值
    c = b.toarray()
    # 对获取到的返回值进行predict
    return c

# BernoulliNB 朴素贝叶斯分类器
def BernoulliNB_predict(tf_idf_array, labels, test_size=0.3, model_score=0.7, model_path=''):
    '''
    :param tf_idf_array: tf-idf矩阵
    :param labels: 标签列表
    :param test_size: 测试集百分比
    :param model_score: 模型准确率包粉笔
    :param model_path: 保存模型路径
    '''
    train_wids, test_wids, train_labels, test_labels = train_test_split(tf_idf_array, labels, test_size=test_size)
    classifier = BernoulliNB(alpha=0.01)
    classifier.fit(train_wids, train_labels)
    score = classifier.score(test_wids, test_labels)
    if score>model_score:
        if model_path:
            log.info(score)
            joblib.dump(classifier, model_score)
        else:
            log.info(score)























