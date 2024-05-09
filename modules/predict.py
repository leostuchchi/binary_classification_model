import json
import dill
import pandas as pd
from pydantic import BaseModel
import logging
import os
import glob
import sys
import csv
import zipfile
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'sber_auto'))

model_file = glob.glob(f'{path}/data/models/*.pkl')

with open(*model_file, 'rb') as file:
    model = dill.load(file)
print(model['metadata'])



def el_zippo(): # архивируем предыдущее предсказание
    directory = (f'{path}/data/predictions')
    files = os.listdir(directory)
    csv_file = [file for file in files if file.endswith('.csv')]
    if csv_file:  # Создание имени архива
        nov = datetime.datetime.now()
        timestamp = str(int(datetime.datetime.timestamp(nov)))
        archive_name = f"archive_{timestamp}.zip"
        with zipfile.ZipFile(archive_name, 'w') as zipf:
            zipf.write(os.path.join(directory, csv_file[0]))


def remover(): # очистка директории
    directory = (f'{path}/data/predictions')
    for root, dirs, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(root, file))


def fileout(): # создадим имя файла для предсказания
    directory = (f'{path}/data/models')  # известная директория
    extension = '.pkl'  # известное расширение

    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_name = filename.replace(extension, '')

            return file_name


def predict():
    remover()
    file_out = (f'{path}/data/predictions/pred_mod_{fileout()}.csv')

    # Находим все файлы с расширением .json в текущей директории
    predict_file = glob.glob(f'{path}/data/test/*.json')

    # Если найден хотя бы один файл
    if predict_file:
        with open(file_out, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            for i in predict_file:
                # Выбираем первый найденный файл
                with open(i, 'rb') as fin:
                    form = json.load(fin)

                df = pd.DataFrame.from_dict([form.dict()])
                y = model.predict(df)
                pred_str = (i + ',', y[0])

                csvwriter.writerow(pred_str)




def true_positive_rate():
    df = pd.read_csv(f'{path}/data/train/to_pipeline/df_train.csv')
    X = df.drop('event_value', axis=1)
    y = df['event_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    y_pred = model['model'].predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    TPR = tp / (tp + fn)
    print("True Positive Rate (TPR):", TPR)
    print(confusion_matrix(y_test, y_pred).ravel())


if __name__ == '__main__':
    true_positive_rate()
