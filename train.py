import argparse

import re
import pickle
import random
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

from xgboost import XGBClassifier


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def tokenize(text):
        return [symbol for symbol in re.split(r' |,|\n|\.|!|\?', text.lower())
                if symbol.isalpha()]

    @staticmethod
    def tokenize_sentences(text):
        tokenized = list()
        for sentence in re.split(r'\.|!|\?', text.lower()):
            tokenized_sentence = [symbol for symbol in
                                  re.split(r' |,|\n', sentence) if
                                  symbol.isalpha()]
            if len(tokenized_sentence) > 1:
                tokenized.append(tokenized_sentence)
        return tokenized


def create_bigrams(tokens):
    dictionary = dict()
    for i in range(1, len(tokens)):
        if tokens[i - 1] not in dictionary:
            dictionary[tokens[i - 1]] = [tokens[i]]
        else:
            dictionary[tokens[i - 1]].append(tokens[i])
        dictionary[tokens[i - 1]] = list(set(dictionary[tokens[i - 1]]))
    return dictionary


class DataBuilder:
    def __init__(self, data_path=None):
        if data_path is None:
            self.data = input()
        else:
            with open(data_path, encoding="utf8") as file:
                self.data = file.read()

        self.tokenizer = Tokenizer()
        self.tokens = self.tokenizer.tokenize(self.data)
        self.sentences = self.tokenizer.tokenize_sentences(self.data)

        self.dictionary = create_bigrams(self.tokens)

        self.X = list()
        self.y = list()

        self.X_vectorized = list()
        self.y_vectorized = list()

        self.X_train = list()
        self.X_test = list()
        self.y_train = list()
        self.y_test = list()

    def create_dataset(self):
        for sentence in tqdm(self.sentences, desc="Creating dataset"):
            for i in range(1, len(sentence)):
                np.random.seed(42)
                possible_words = list(
                    np.random.choice(
                        self.dictionary[sentence[i - 1]], size=4)
                ) + [sentence[i]]
                possible_words = list(set(possible_words))
                random.seed(42)
                random.shuffle(possible_words)
                y_subset = [
                    int(word == sentence[i])
                    for word in possible_words
                ]
                X_subset = [
                    [sentence[i - 1], word, len(sentence[i - 1]), len(word), i]
                    for word in possible_words
                ]
                self.X.extend(X_subset)
                self.y.extend(y_subset)

    def create_train_dataset(self, w2v):
        self.X_vectorized = list()
        self.y_vectorized = self.y.copy()
        for i in tqdm(range(len(self.X)),
                      desc="Creating train dataset"):
            self.X_vectorized.append(list())
            self.X_vectorized[i].extend(w2v(self.X[i][0]).tolist())
            self.X_vectorized[i].extend(w2v(self.X[i][1]).tolist())
            self.X_vectorized[i].extend(
                [self.X[i][2], self.X[i][3], self.X[i][4]]
            )

    def train_test_split(self, train_size=0.9):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_vectorized, self.y_vectorized, train_size=train_size,
            shuffle=True, random_state=42)

    def save(self, path):
        with open(path + "/X", "wb") as fp:
            pickle.dump(self.X, fp)
        with open(path + "/y", "wb") as fp:
            pickle.dump(self.y, fp)

    def load(self, X_path, y_path):
        with open(X_path + '/X', "rb") as fp:
            self.X = pickle.load(fp)
        with open(y_path + '/y', "rb") as fp:
            self.y = pickle.load(fp)


class Word2VecWrapper:
    def __init__(self):
        self.w2v = None
        self.sentences = list()
        self.emb_size = 64

    def train(self, sentences, emb_size=64, window=7, min_count=1, sg=1,
              workers=4, epochs=5):
        self.sentences = sentences
        self.emb_size = emb_size

        self.w2v = Word2Vec(sentences=self.sentences,
                            vector_size=self.emb_size, window=window,
                            min_count=min_count, sg=sg, workers=workers)

        self.w2v.train(self.sentences, total_examples=len(self.sentences),
                       epochs=epochs)

    def save(self, path):
        self.w2v.save(path + '/word2vec.model')

    def load(self, path):
        self.w2v = Word2Vec.load(path + '/word2vec.model')

    def __call__(self, word):
        if word in self.w2v.wv:
            return self.w2v.wv[word]
        return np.array([0 for _ in range(self.emb_size)])


class XGBClassifierWrapper:
    def __init__(self, n_estimators=30, max_depth=6, eval_metric='auc',
                 learning_rate=None, early_stopping_rounds=10, n_jobs=4):
        self.model = XGBClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   eval_metric=eval_metric,
                                   learning_rate=learning_rate, n_jobs=n_jobs,
                                   early_stopping_rounds=early_stopping_rounds,
                                   random_state=42)

    def fit(self, X_train, y_train, eval_set):
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=10)

    def save(self, path):
        self.model.save_model(path + '/xgb.json')

    def load(self, path):
        self.model.load_model(path + '/xgb.json')

    def __call__(self, model_input):
        return self.model.predict_proba(model_input)[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='the path to the file that contains the training text'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='models',
        help='the path to the folder where the models will be saved'
    )
    namespace = parser.parse_args()

    data_path = namespace.input_file
    builder = DataBuilder(data_path=data_path)
    builder.create_dataset()

    model_path = namespace.model
    builder.save(model_path)

    emb_size = 64
    w2v = Word2VecWrapper()
    w2v.train(sentences=builder.sentences, emb_size=emb_size, epochs=5)

    builder.create_train_dataset(w2v=w2v)
    builder.train_test_split()

    xgb = XGBClassifierWrapper()
    xgb.fit(builder.X_train, builder.y_train,
            eval_set=[(builder.X_test, builder.y_test)])

    xgb.save(model_path)
    w2v.save(model_path)
