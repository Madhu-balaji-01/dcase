import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------
# PART I
# changing path to audio files & encoding scene labels

train_file = './dataset/dcase/evaluation_setup/fold1_train.csv'
test_file = './dataset/dcase/evaluation_setup/fold1_test.csv'
evaluate_file = './dataset/dcase/evaluation_setup/fold1_evaluate.csv'

train_df = pd.read_csv(train_file, sep='\t')
test_df = pd.read_csv(test_file, sep='\t')
evaluate_df = pd.read_csv(evaluate_file, sep='\t')

mapper = {
    'airport': 0,
    'shopping_mall': 1,
    'metro_station': 2,
    'street_pedestrian': 3,
    'public_square': 4,
    'street_traffic': 5,
    'tram': 6,
    'bus': 7,
    'metro': 8,
    'park': 9,
}

def encode(df, mapper):
    df.replace({'scene_label':mapper}, inplace=True)
    
def edit(df):
    df['filename_audio'] = './dataset/dcase/' + df['filename_audio']
    df['filename_video'] = './dataset/dcase/' + df['filename_video']
    
encode(train_df, mapper)
encode(evaluate_df, mapper)
edit(train_df)
edit(test_df)
edit(evaluate_df)

# train_df.to_csv('./dataset/dcase/evaluation_setup/modify_train.csv', index=False)
# test_df.to_csv('./dataset/dcase/evaluation_setup/modify_test.csv', index=False)
# evaluate_df.to_csv('./dataset/dcase/evaluation_setup/modify_evaluate.csv', index=False)

# -----------------------------------------------------------------------------------------
# PART II
# changing csv for 1 sec files

# train_file = './dataset/dcase/evaluation_setup/modify_train.csv'
# test_file = './dataset/dcase/evaluation_setup/modify_test.csv'
# evaluate_file = './dataset/dcase/evaluation_setup/modify_evaluate.csv'

# train_df = pd.read_csv(train_file)
# test_df = pd.read_csv(test_file)
# evaluate_df = pd.read_csv(evaluate_file)

def clips(df):
    n = len(df)
    df_copy = df.copy()
    df_copy = df_copy.reindex(df_copy.index.repeat(10))
    
    df_copy['num'] = np.tile(np.arange(0, 10), n)
    df_copy['num'] = df_copy['num'].astype(str)
    df_copy['filename_audio'] = df_copy['filename_audio'].apply(lambda x: x[:-4]).astype(str)
    df_copy['filename_video'] = df_copy['filename_video'].apply(lambda x: x[:-4]).astype(str)
    df_copy['filename_audio'] = df_copy[['filename_audio', 'num']].apply(lambda x: '-'.join(x.astype(str)), axis=1)
    df_copy['filename_video'] = df_copy[['filename_video', 'num']].apply(lambda x: '-'.join(x.astype(str)), axis=1)
    df_copy['filename_audio'] += '.wav'
    df_copy['filename_video'] += '.mp4'
    
    del df_copy['num']
    
    return df_copy    

train_df_2 = clips(train_df)
test_df_2 = clips(test_df)
evaluate_df_2 = clips(evaluate_df)

train_df_2.to_csv('./dataset/dcase/evaluation_setup/modify_train.csv', index=False)
test_df_2.to_csv('./dataset/dcase/evaluation_setup/modify_test.csv', index=False)
evaluate_df_2.to_csv('./dataset/dcase/evaluation_setup/modify_evaluate.csv', index=False)