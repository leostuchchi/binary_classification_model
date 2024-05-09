
import datetime as dt
import datetime as datetime
import os
import sys
import airflow
import pandas
import pandas as pd
import glob
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import dill

path = os.path.expanduser(os.path.join(os.path.expanduser('~'), 'sber_auto'))

os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

from modules.take_data import take_data
from modules.pipeline import pipeline
from modules.predict import predict
from modules.result_out import result_out


args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}


def result():
    directory = (f'{path}/data/predictions')  # известная директория
    extension = '.csv'  # известное расширение

    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_name = filename.replace(extension, '')
            df = pd.read_csv(f'{path}/data/predictions/{filename}')
            new_filename = 'predict_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
            file = pd.DataFrame(df)
            df.to_csv(os.path.join(os.path.expanduser('~'), new_filename), encoding='utf-8',index=False)



with DAG(
        dag_id='sber_auto_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:

    # BashOperator, выполняющий указанную bash-команду
    first_task = BashOperator(
        task_id='first_task',
        bash_command='echo "Here we start!"',
        dag=dag,
    )

    # получаем целевую переменную из ga_hits и inner к ga_sessions
    take_data = PythonOperator(
        task_id='take_data',
        python_callable=take_data,
    )


    # запуск конвейера
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    # запуск predict
    predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )

    # запись результата предсказания
    result_out = PythonOperator(
        task_id='write_file',
        python_callable=result_out,
    )


    first_task >> take_data >> pipeline >> predict >> result_out
