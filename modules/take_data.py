import os
import pandas as pd

# Укажем путь к файлам корневая директория: os.path.path.join(os.path.expanduser('~')
path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'sber_auto'))


def remover():
    directory = (f'{path}/data/train/to_pipeline')
    for root, dirs, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(root, file))


def take_cell():
    df = pd.read_csv(f'{path}/data/train/ga_hits.csv')
    df_lite = df[['session_id', 'event_action', 'event_value']]
    del df
    event = ['sub_car_claim_click',
             'sub_car_claim_submit_click',
             'sub_open_dialog_click',
             'sub_custom_question_submit_click',
             'sub_call_number_click',
             'sub_callback_submit_click',
             'sub_submit_success',
             'sub_car_request_submit_click']
    df_lite['event_action'] = df_lite['event_action'].str.lower()
    df_lite['event_value'] = df_lite['event_action'].apply(lambda x: 1 if x in event else 0)
    df = df_lite[['session_id', 'event_value']]
    del df_lite
    print(df.shape)
    return df


def merge_data():
    df_event = take_cell()
    df_sess = pd.read_csv(f'{path}/data/train/ga_sessions.csv', dtype={'client_id': str}, low_memory=False)
    df = pd.merge(df_sess, df_event, on='session_id', how='inner')
    del df_sess
    del df_event
    print(df.shape)
    return df



def take_data():
    (df) = merge_data()
    df.drop_duplicates(subset=df.columns, inplace=True, keep='first')
    #df['count_duplicates'] = df.groupby(df.columns.tolist()).cumcount() + 1
    #df['count_duplicates'] = df['count_duplicates'].fillna(0)

    print(df.shape)
    # remover()
    df.to_csv(f'{path}/data/train/to_pipeline/df_train.csv', index=False)


if __name__ == '__main__':
    take_data()