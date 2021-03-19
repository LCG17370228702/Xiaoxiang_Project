from flask import Flask,render_template,url_for,request
import pickle
import re
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
import jieba
import warnings
warnings.filterwarnings("ignore")


class UserProfiling():
  def __init__(self, classifier=MultinomialNB()):
  # def __init__(self, classifier=LGBMClassifier()):
    self.classifier = classifier
    self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=2, max_features=40000, preprocessor=self._preprocess_text)
  
  # 读取停用词表
  def get_stopwords(self, stopwords_path='./cn_stopwords.txt'):
    return pd.read_table(stopwords_path, header=None)[0].values

  # 私有函数，数据清洗
  def _preprocess_text(self, document):
    stopwords = self.get_stopwords()
    noise_pattern = re.compile("|".join(["http\S+", ":http\S+", "ｗｗｗ.+?\t",'\d+\@\S+']))
    clean_text = re.sub(noise_pattern, "", document)
    segs = jieba.lcut(clean_text)
    segs = list(filter(lambda x:len(x)>1, segs)) #没有解析出来的新闻过滤掉
    segs = list(filter(lambda x:x not in stopwords, segs)) #把停用词过滤掉
    return ' '.join(segs)

  # 构建特征
  def features(self, X):
    return self.vectorizer.transform(X)
  
  # 拟合数据
  def fit(self, X, y, n_splits=5, shuffle=True, random_state=None):
  # def fit(self, X, y):  
    self.vectorizer.fit(X)
    # self.classifier.fit(self.features(X), y)
    self.stratified_kfold(self.features(X), y, n_splits=n_splits, shuffle=shuffle, random_state=random_state)

  # 交叉验证
  def stratified_kfold(self, X, y, n_splits, shuffle, random_state):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for idx, (train_index, valid_index) in enumerate(skf.split(X, y)):
      X_train, x_valid = X[train_index], X[valid_index]
      y_train, y_valid = y[train_index], y[valid_index]
      self.classifier.fit(X_train, y_train)
      print('第{}折：acc = {}'.format(idx+1, accuracy_score(y_valid, self.classifier.predict(x_valid))))

  # 预测
  def predict(self, x):
    return self.classifier.predict(self.features([x]))

  def score(self, X, y):
    return self.classifier.score(self.features(X), y)

  # 模型持久化存储
  def save_model(self, path):
    dump((self.classifier, self.vectorizer), path)
    
  # 模型加载
  def load_model(self, path):
    self.classifier, self.vectorizer = load(path)


app = Flask(__name__)

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
  age_path = './model/user_profiling_Age.model'
  edu_path = './model/user_profiling_Edu.model'
  gender_path = './model/user_profiling_Gender.model'

  user_profiling_age = UserProfiling()
  user_profiling_age.load_model(age_path)

  user_profiling_edu = UserProfiling()
  user_profiling_edu.load_model(edu_path)

  user_profiling_gender = UserProfiling()
  user_profiling_gender.load_model(gender_path)

  if request.method == 'POST':
    message = request.form['message']
    age = user_profiling_age.predict(message)
    edu = user_profiling_edu.predict(message)
    gender = user_profiling_gender.predict(message)
  return render_template('result.html', age = age[0],  edu = edu[0],  gender = gender[0])

if __name__ == '__main__':
  app.run(debug=True)
