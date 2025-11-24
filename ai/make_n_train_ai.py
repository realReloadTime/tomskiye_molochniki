# CUDA версия pytorch для 13.0.x СКАЧИВАТЬ С VPN
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import numpy as np
import re

from datasets import load_dataset

import nltk
from nltk.stem.snowball import SnowballStemmer

'''
If you did not install the data to one of the above central locations, you will need to set the NLTK_DATA 
environment variable to specify the location of the data. (On a Windows machine, right click on 
“My Computer” then select Properties > Advanced > Environment Variables > User Variables > New...)
'''

nltk.download('averaged_perceptron_tagger_rus', 'C:\\nltk_data')
nltk.download('stopwords', 'C:\\nltk_data')


def upper_case_rate(string):
    """Returns percentage of uppercase letters in the string"""
    return np.array(list(map(str.isupper, string))).mean()


def clean_text(string: str) -> str:
    string = string.lower()
    string = re.sub(r"http\S+", "", string)  # deletion urls
    string = string.replace('ё', 'е')

    # cyrillic + latin
    words = re.findall(r'[а-яa-z]+', string)

    # deletion "и", "а", "на", "в", etc.
    stopwords = set(nltk.corpus.stopwords.words('russian'))
    words = [w for w in words if w not in stopwords]

    functionalPos = {'CONJ', 'PRCL'}
    words = [w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functionalPos]

    # "Я ходил в магазин и купил молоко, а потом бегал в парке! http://example.com" -> "ход магазин куп молок бега парк"
    stemmer = SnowballStemmer('russian')
    stemmed_words = [stemmer.stem(word) for word in words]

    return ' '.join(stemmed_words)


def train_nn(data: pd.DataFrame):
    data['upcase_rate'] = list(map(upper_case_rate, data.comment.values))
    print(data.head(20))

    text = np.array(data.comment.values)
    target = data.toxic.astype(int).values

    text = list(map(clean_text, text))
    X_train, X_test, y_train, y_test = train_test_split(text, target, test_size=.3, stratify=target, shuffle=True,
                                                        random_state=0)
    print('Dim of train:', len(X_train), '\tTarget rate: {:.2f}%'.format(y_train.mean()))
    print("Dim of test:", len(X_test), '\tTarget rate: {:.2f}%'.format(y_test.mean()))

    clf_pipeline = Pipeline(
        [("vectorizer", TfidfVectorizer()),
         ("classifier", LinearSVC())]
    )

    clf_pipeline.fit(X_train, y_train)
    print(metrics.classification_report(y_test, clf_pipeline.predict(X_test)))
    f1_base = metrics.f1_score(y_test, clf_pipeline.predict(X_test))
    print(f1_base)


def test_gpu_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


if __name__ == "__main__":
    # print(clean_text("Я ходил в магазин и купил молоко, а потом бегал в парке! http://example.com"))
    df1 = pd.read_csv(r'dataset/labeled_2ch_pikabu.csv')
    df2 = pd.concat([pd.read_json('dataset/okru_part1.jsonl', lines=True),
               pd.read_json('dataset/okru_part2.jsonl', lines=True)], axis=0, ignore_index=True)
    df2['comment'] = df2['text']
    df2['toxic'] = df2['label'].apply(lambda x: float(x))
    df2.drop('text', axis=1, inplace=True)
    df2.drop('label', axis=1, inplace=True)

    total_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    train_nn(total_df)
