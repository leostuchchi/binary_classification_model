import warnings
warnings.filterwarnings('ignore')
import logging
import os

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import dill


# Укажем путь к файлам корневая директория: os.path.path.join(os.path.expanduser('~')
path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'sber_auto'))


def missing_out(df: pd.DataFrame) -> pd.DataFrame: # удалим колонки с пропусками в признаках более 45%
    missing_values = ((df.isna().sum() / len(df)) * 100)
    col_drop = missing_values[missing_values.values > 45].index.tolist()
    df = df.drop(col_drop, axis=1)
    return df



def nan_out(df: pd.DataFrame) -> pd.DataFrame: # удаляем пропуски
    categorical = df.select_dtypes(include=['object', 'category']).columns
    numerical = df.select_dtypes(include=['int64', 'float64']).columns
    # В категориальных фичах заменяем пропуски значением 'other'
    for feat in categorical:
        df[feat] = df[feat].fillna('other')
        #print(df[feat].nunique())
    # В численных фичах заменяем пропуски медианой
    for feat in numerical:
        #df[feat].fillna(df[feat].median(), inplace=True) такой код плохо влияет на предсказания
        df[feat] = df[feat].fillna(df[feat].median())
    print('NAN:', sum(df.isnull().sum()))  # Убедимся, что пропущенных значений больше нет
    return df




def referral(df: pd.DataFrame) -> pd.DataFrame:
    df['utm_medium'] = df['utm_medium'].str.lower()
    df['utm_organic'] = df['utm_medium'].apply(lambda x: 1 if x in ['organic', 'referral', '(none)'] else 0)
    df = df.drop(['utm_medium'], axis=1)
    return df



def visit_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df['visit_number'] = df['visit_number'].astype('int')
    df['visit_number'] = np.where(df['visit_number'] > 1, 2, df['visit_number'])
    return df



def short_utm(df: pd.DataFrame) -> pd.DataFrame:
    df['w_source'] = df['utm_source'].str[0:5]
    df.drop(['utm_source'], axis=1, inplace=True)
    df['w_campaign'] = df['utm_campaign'].str[0:4]
    df.drop(['utm_campaign'], axis=1, inplace=True)
    df['w_adcontent'] = df['utm_adcontent'].str[0:3]
    df.drop(['utm_adcontent'], axis=1, inplace=True)
    return df



def date_time(df: pd.DataFrame) -> pd.DataFrame:
    df['date_time'] = pd.to_datetime(df['visit_date'] + ' ' + df['visit_time'])
    col_to_drop = ['visit_date', 'visit_time']  # сразу дропнем date & time
    df = df.drop(col_to_drop, axis=1)
    return df



def month_day_hour(df: pd.DataFrame) -> pd.DataFrame:
    df['date_time'] = pd.to_datetime(df['visit_date'] + ' ' + df['visit_time'])
    col_to_drop = ['visit_date', 'visit_time']  # сразу дропнем date & time
    df = df.drop(col_to_drop, axis=1)
    df['month_day_hour'] = df['date_time'].dt.month.astype(str) + ':' + df['date_time'].dt.dayofweek.astype(str) + ':' + df['date_time'].dt.hour.astype(str)
    df.drop(['date_time'], axis=1, inplace=True)
    return df



def screen_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df['device_screen_resolution'] = df['device_screen_resolution'].apply(
        lambda x: float(x.split('x')[0]) * float(x.split('x')[1]))
    return df


def geo_to_int(df):
    df['geo_country'] = df['geo_country'].str.lower()
    df['geo_city'] = df['geo_city'].str.lower()

    df['geo_city'] = df['geo_city'].apply(lambda x: 3 if x in ['moscow'] else 2 if x in ['saint petersburg'] else 1)
    df['geo_country'] = df['geo_country'].apply(lambda x: 1 if x in ['russia'] else 0)
    df['geo'] = df['geo_country'] * df['geo_city']
    df.drop(['geo_country'], axis=1, inplace=True)
    df.drop(['geo_city'], axis=1, inplace=True)
    return df



def utm_path(df: pd.DataFrame) -> pd.DataFrame:
    df['utm_path'] = df['w_source'].str.lower() + '_' + df['w_campaign'].str.lower() + '_' + df['w_adcontent'].str.lower()
    return df



def drop_utm(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(['w_source'], axis=1, inplace=True)
    df.drop(['w_campaign'], axis=1, inplace=True)
    df.drop(['w_adcontent'], axis=1, inplace=True)
    return df



def drop_ids(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(['session_id', 'client_id'], axis=1, inplace=True)
    return df



def global_drop(df: pd.DataFrame) -> pd.DataFrame:
    global_drop = ['visit_time', 'device_browser', 'device_category', 'device_brand']
    df.drop(global_drop, axis=1, inplace=True)
    print(df.columns)
    return df



def to_category(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype('category')
    return df



def remove_outliers(df: pd.DataFrame) -> pd.DataFrame: # удалим выбросы
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    numerical_columns.remove('visit_number')
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def remover(): # очистка директории
    directory = (f'{path}/data/models')
    for root, dirs, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(root, file))


def pipeline(): # -> None:
    df = pd.read_csv(f'{path}/data/train/to_pipeline/df_train.csv')
    # дропнем не Россию
    df['geo_country'] = df['geo_country'].str.lower()
    df = df[df['geo_country'] == 'russia']

    df['event_value'] = df['event_value'].astype('category')

    X = df.drop('event_value', axis=1)
    y = df['event_value']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object', 'category'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('without_data_remover', FunctionTransformer(missing_out)),
        ('without_NAN', FunctionTransformer(nan_out)),
        ('free_or_not_free', FunctionTransformer(referral)),
        ('outliers_visit', FunctionTransformer(visit_outliers)),
        ('short_utm', FunctionTransformer(short_utm)),
        #('date_and_time', FunctionTransformer(date_time)),
        #('feature_month_day_hour', FunctionTransformer(month_day_hour)),
        ('screen_to_float', FunctionTransformer(screen_to_float)),
        ('feature_location', FunctionTransformer(geo_to_int)),
        ('feature_utm', FunctionTransformer(utm_path)),
        ('?drop_w_utm', FunctionTransformer(drop_utm)),
        ('drop_ids', FunctionTransformer(drop_ids)),
        ('global_drop', FunctionTransformer(global_drop)),
        ('everybody_to_category', FunctionTransformer(to_category)),
        ('outliers_was_here_!without_VISIT', FunctionTransformer(remove_outliers)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='liblinear'),
        LogisticRegression(class_weight='balanced'),
        RandomForestClassifier(n_estimators=75, max_depth=10, max_features='sqrt', random_state=42)
        #KNeighborsClassifier(),
        #MLPClassifier(),
        #SVC()
    ]

    best_score = .0

    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Создаем объект StratifiedKFold с random_state=42
        kf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        score = cross_val_score(pipe, X, y, cv=kf, scoring='roc_auc')

        logging.info(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
            print(best_score)
    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    best_pipe.fit(X, y)
    model_filename = f'{path}/data/models/sber_{round(best_score, 4)}_roc_auc.pkl'
    metadata = {
        "name": "SberAutoAction prediction model",
        "author": "leostuchchi",
        "version": 1,
        "date": datetime.now(),
        "roc_auc": best_score
    }

    remover() # удаляем предыдущую модель

    print(best_score)

    with open(model_filename, 'wb') as file:
        dill.dump({'model': best_pipe, 'metadata': metadata}, file, recurse=True)

    logging.info(f'Model is saved as {model_filename}')

if __name__ == '__main__':
    pipeline()










