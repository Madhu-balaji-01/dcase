import os
import random
import csv
import math

def write_csv(final_list, name):
    random.shuffle(final_list)
    with open(name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(final_list)
    csvFile.close()

class ClassNode:

    def __init__(self, name,data_list, address):
        self.name = name
        self.data_list = data_list
        self.address = address

    def count(self):
        return len(self.data_list)

    def data_shuffle(self):
        random.shuffle(self.data_list)

def read_folders(main_address):
    list_folders = os.listdir(main_address)
    raw_data_dict = {}
    for folder in list_folders:
        file_in_folder = os.listdir('{}{}/'.format(main_address,folder))
        raw_data_dict[folder] = ClassNode(folder,file_in_folder,'{}{}/'.format(main_address,folder))
    return raw_data_dict

def copy_files(file_list, target_list, address,y_tag):
    for file in file_list:
        target_list.append([address+file, y_tag])
    return target_list

def gen_csv_tvt(raw_data_dict, data_percent, classes_inused):

    train_list = []
    valid_list = []
    test_list = []

    for key in classes_inused.keys():
        raw_data_dict[key].data_shuffle()
        if classes_inused[key][1] != 'all':
            temp_train_end = math.floor(classes_inused[key][1]*data_percent[0])
            temp_valid_end = math.floor(classes_inused[key][1]*data_percent[1]) + temp_train_end
            temp_test_end = math.floor(classes_inused[key][1]*data_percent[2]) + temp_valid_end
            files_list_train = raw_data_dict[key].data_list[:temp_train_end]
            files_list_valid = raw_data_dict[key].data_list[temp_train_end:temp_valid_end]
            files_list_test = raw_data_dict[key].data_list[temp_valid_end:temp_test_end]
        else:
            temp_train_end = math.floor(raw_data_dict[key].count() * data_percent[0])
            temp_valid_end = math.floor(raw_data_dict[key].count() * data_percent[1]) + temp_train_end
            temp_test_end = math.floor(raw_data_dict[key].count() * data_percent[2]) + temp_valid_end
            files_list_train = raw_data_dict[key].data_list[:temp_train_end]
            files_list_valid = raw_data_dict[key].data_list[temp_train_end:temp_valid_end]
            files_list_test = raw_data_dict[key].data_list[temp_valid_end:temp_test_end]

        train_list = copy_files(files_list_train, train_list, raw_data_dict[key].address, classes_inused[key][0])
        valid_list = copy_files(files_list_valid, valid_list, raw_data_dict[key].address, classes_inused[key][0])
        test_list = copy_files(files_list_test, test_list, raw_data_dict[key].address, classes_inused[key][0])

    write_csv(train_list, './dataset/train_4.csv')
    write_csv(valid_list, './dataset/valid_4.csv')
    write_csv(test_list, './dataset/test_4.csv')


if __name__ == '__main__':

    classes_inused = { 'silence_all':[0, 4000],
                    'clapping':[1,'all'],
                    'laughing':[2, 'all'],
                    'screaming':[3, 'all'],
                    'shout':[3, 'all'],
                    'conversation':[4,'all'],
                    'speech_male':[4, 1500],
                    'speech_female':[4, 1500]
                    # 'happy':[5, 'all'],
                    # 'angry':[6, 'all']
                    }

    data_percent = (.8, .1, .1)  # (train, valid, test)

    main_address = './dataset/audioset/'
    raw_data_dict = read_folders(main_address)

    gen_csv_tvt(raw_data_dict, data_percent, classes_inused)

# classes = {'silence_all':0,
#            'conversation':1,
#            'laughing':2,
#            'clapping':3,
#            'shout':4,
#            'screaming':4}
#
#
#
# class_dict = {}
# for i, key in enumerate(classes):
#     temp_address = '{}{}/'.format(address,key)
#     temp_name = key
#     temp_aufile_list = os.listdir(temp_address)
#     class_dict[str(i)] = ClassNode(temp_name, temp_aufile_list,temp_address, classes[key])
#
# data_list = []
# for key in class_dict:
#     class_dict[key].data_shuffle()
#     for audio_file in class_dict[key].data_list:
#         au_file_address = '{}{}'.format(class_dict[key].address,audio_file)
#         temp = [au_file_address, class_dict[key].y_tag]
#         data_list.append(temp)
#
# random.shuffle(data_list)
#
# train_list = data_list[0:12000]
# valid_list = data_list[12000:]
#
# write_csv(train_list, './dataset/train.csv')
# write_csv(valid_list, './dataset/valid.csv')