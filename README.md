# tinkoff-generation

### Мой способ решения задачи:
1) Берется 5 случайных слов, полученных с помощью Bi-Gram;
2) С помощью Word2Vec беру эмбеддинг полученного слова и эмбеддинг последнего слова в префиксе, добавляю в массив длину этих слов и длину предложения;
3) Подаю эти данные в XGBoost, он возвращает вероятность того, что полученное слово следующее;
4) Выбираю слово, которое имело наибольшую вероятность и добавляю его в конец предожения;
5) Повторяю процесс length раз, где length - длина генерируемой последовательности;

Пример использования agrparse в train.py:
```
python train.py --input_file="data/Vlastelin-Kolec.txt" --models="models"
```

Пример использования argparse в generate.py:
```
python generate.py --models="models" --prefix="Он купил" --length=3
```
