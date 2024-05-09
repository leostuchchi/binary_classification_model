import pandas as pd
import os
import datetime

path = os.path.expanduser(os.path.join(os.path.expanduser('~'), 'sber_auto'))

def result_out():
    directory = (f'{path}/data/predictions')  # известная директория
    extension = '.csv'  # известное расширение
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_name = filename.replace(extension, '')
            df = pd.read_csv(f'{path}/data/predictions/{filename}')
            new_filename = 'predict_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
            file = pd.DataFrame(df)
            df.to_csv(os.path.join(os.path.expanduser('~'), new_filename), encoding='utf-8', index=False)
        else:
            continue



if __name__ == '__main__':
    result_out()