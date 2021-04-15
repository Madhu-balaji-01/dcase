import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.metrics import confusion_matrix
import itertools


matrix = np.array([[88,   1,   1,   1, 3, 2, 3],
       [  5,  63,  11,   13,  5, 2, 1],
       [  1,   6, 76,  11,  2,1,1],
       [ 2,   3,  8, 79,  2,4,2],
       [ 2,   1,  1,  1, 83, 8,6],
           [2,4,2,12,8,70,2],
           [ 2,1,14,3,10,3,72],])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation', 'angry', 'happy']
# matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)
acc= 75.57
# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=classes, normalize=True,
                      title= ' Confusion Matrix (Accuracy: %0.3f%%)' %acc)



#
# matrix = ([[88,   1,   1,   1, 3, 2, 3],
#        [  5,  63,  11,   13,  5, 2, 1],
#        [  1,   6, 76,  11,  2,1,1],
#        [ 2,   3,  8, 79,  2,4,2],
#        [ 2,   1,  1,  1, 83, 8,6],
#            [2,4,2,12,8,70,2],
#            [ 2,1,14,3,10,3,72],])






# def add_noise(data, noise_factor):
#     noise = np.random.randn(len(data))
#     augmented_data = data + noise_factor * noise
#     # Cast back to same data type
#     augmented_data = augmented_data.astype(type(data[0]))
#     return augmented_data
#
# def time_shift(data, sampling_rate, shift_max, shift_direction):
#     shift = np.random.randint(sampling_rate * shift_max)
#     if shift_direction == 'right':
#         shift = -shift
#     elif shift_direction == 'both':
#         direction = np.random.randint(0, 2)
#         if direction == 1:
#             shift = -shift
#     augmented_data = np.roll(data, shift)
#     # Set to silence for heading/ tailing
#     if shift > 0:
#         augmented_data[:shift] = 0
#     else:
#         augmented_data[shift:] = 0
#     return augmented_data
#
# def pitch_shift(data, sampling_rate, pitch_factor):
#     return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
#
# def time_streach(data, speed_factor):
#     return librosa.effects.time_stretch(data, speed_factor)
#
#
# ip, _ = librosa.load('./report_mat/test.wav', sr=16000)
# ip_norm = librosa.util.normalize(ip)
#
# ip_new = time_streach(ip_norm, 1.5)
#
# plt.figure()
# plt.plot(ip_norm)
#
# plt.figure()
# plt.plot(ip_new)
#
# sf.write('./report_mat/time_streach.wav',ip_new,16000)