# DCASE Challenge 2021 - Task 1b

## System

OS: Ubuntu 16.04/18.04
Language: Python 3.6
Library: Pytorch 1.8
CUDA: v10.1/v10.2
GPU: Nvidia GeForce RTX 2080

## Requirements
*the system was tested in conda environment*

Packages required:
pytorch=1.8 torchaudio torchvision
cudatoolkit=10.1/10.2 cudnn
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
pyaudio
librosa
pyroomacoustics
webrtcvad

### File Structure

|_dataset
    |_evaluation_setup
        train.csv
        test.csv
        evaluate.csv
    |_audio
        *audio.wav
    |_video
        *video.mp4
|_model_zoo
|_models
    vgg_m.py
    dcase_past.py
main_train.py
data_engin.py
fit_model.py
au_transform.py

## Description

*main_train.py* is the main file to run for training.
*fit_model.py* contains the training and validation processes.
*data_engin.py* generates data batches.
*au_transform.py* is used to extract features and transforms them into tensors.
*vgg_m.py* and *dcase_past.py* contains the CNN architecture.

## Usage

Run the script from terminal
    python main_train.py