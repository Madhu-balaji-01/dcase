import numpy as np 
import pandas as pd

infile1 = './dataset/dcase/evaluation_setup/fold1_train.csv'
infile2 = './dataset/dcase/evaluation_setup/fold1_test.csv'
infile3 = './dataset/dcase/evaluation_setup/fold1_evaluate.csv'

df1 = pd.read_csv(infile1, sep='\t')
df2 = pd.read_csv(infile2, sep='\t')
df3 = pd.read_csv(infile3, sep='\t')

mapper = {
    'airport' : 0,
    'shopping_mall' : 1,
    'metro_station' : 2,
    'street_pedestrian' : 3,
    'public_square' : 4,
    'street_traffic' : 5,
    'tram' : 6,
    'bus' : 7,
    'metro' : 8,
    'park' : 9
}

df1['filename_audio'] = '/home/audio_server1/intern/dataset/dcase/' + df1['filename_audio']
df1['filename_video'] = '/home/audio_server1/intern/dataset/dcase/' + df1['filename_video']
df2['filename_audio'] = '/home/audio_server1/intern/dataset/dcase/' + df2['filename_audio']
df2['filename_video'] = '/home/audio_server1/intern/dataset/dcase/' + df2['filename_video']
df3['filename_audio'] = '/home/audio_server1/intern/dataset/dcase/' + df3['filename_audio']
df3['filename_video'] = '/home/audio_server1/intern/dataset/dcase/' + df3['filename_video']

df1.replace(mapper, inplace=True)
df2.replace(mapper, inplace=True)
df3.replace(mapper, inplace=True)

# print(df1.head())
# print(df2.head())
# print(df3.head())

df1.to_csv('./dataset/dcase/evaluation_setup/modify_train2.csv', index=False)
df2.to_csv('./dataset/dcase/evaluation_setup/modify_test2.csv', index=False)
df3.to_csv('./dataset/dcase/evaluation_setup/modify_evaluate2.csv', index=False)