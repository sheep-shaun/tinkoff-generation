import argparse

import numpy as np

from train import Tokenizer, DataBuilder, Word2VecWrapper, XGBClassifierWrapper


class Generator:
    def __init__(self, builder, w2v, model):
        self.builder = builder
        self.w2v = w2v
        self.model = model
        self.tokenizer = Tokenizer()

    def prepare_text(self, text):
        text = self.tokenizer.tokenize(text)
        model_input = list()
        np.random.seed(42)
        if text[-1] in self.builder.dictionary:
            temp_dict = self.builder.dictionary[text[-1]]
            next_words = np.random.choice(temp_dict, min(5, len(temp_dict)))
        else:
            next_words = np.random.choice(self.builder.tokens, 15)
        for word in next_words:
            temp = list()
            temp.extend(self.w2v(text[-1]).tolist())
            temp.extend(self.w2v(word).tolist())
            temp.extend([len(text[-1]), len(word), len(text), word])
            model_input.append(temp.copy())
        return model_input

    def generate(self, text='', words_number=3):
        if text == '':
            np.random.seed(42)
            text = np.random.choice(self.builder.tokens, 1)[0]
        for _ in range(words_number):
            model_input = self.prepare_text(text)
            best_score = 0
            next_word = ''
            for i in model_input:
                score = self.model([i[:-1]])
                if score > best_score and i[-1] != text.split()[-1]:
                    best_score = score
                    next_word = i[-1]
            text = text + ' ' + next_word
        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text')
    parser.add_argument(
        '--models',
        type=str,
        default='models',
        help='the path to the folder where the models are uploaded from'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='beginning of a sentence (optional argument)'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=5,
        help='length of the generated sequence'
    )
    namespace = parser.parse_args()

    builder = DataBuilder('data/Vlastelin-Kolec.txt')
    builder.load(X_path=namespace.models, y_path=namespace.models)

    w2v = Word2VecWrapper()
    w2v.load(namespace.models)

    xgb = XGBClassifierWrapper()
    xgb.load(namespace.models)

    generator = Generator(builder=builder, w2v=w2v, model=xgb)
    print(generator.generate(namespace.prefix, namespace.length))
