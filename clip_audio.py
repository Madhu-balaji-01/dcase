import os
from glob import glob
from tqdm import tqdm

from pydub import AudioSegment
from pydub.utils import make_chunks

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_path', default='./dataset/audio2/', help='audio files input path')
parser.add_argument('-o', '--output_path', default='./dataset/audio/', help='audio clip output path')
parser.add_argument('-f', '--file_type', default='wav', help='audio file type')
parser.add_argument('-l', '--clips_length', default=1, help='length of clips (in seconds)')

args = parser.parse_args()

if __name__ == "__main__":
    # enables iteration of audio files in a folder
    if args.file_type == ('wav' or 'mp3'):
        filetype = '*.' + args.file_type
        audio_files = glob(args.input_path + filetype)
    else:
        exit(f'FileTypeError: {args.file_type} format not recognized.')

    for file in tqdm(audio_files, ascii=True, desc='audio files:', ncols=100, dynamic_ncols=True):
        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0] # obtain file name without file extension

        if args.file_type == 'wav':
            myaudio = AudioSegment.from_wav(file)
        elif args.file_type == 'mp3':
            myaudio = AudioSegment.from_mp3(file)

        chunk_length_ms = int(args.clips_length) * 1000 # 5 seconds by default (pydub works in ms)
        chunks = make_chunks(myaudio, chunk_length_ms) # make chunks from audio

        # export all of the individual chunks as wav files
        for i, chunk in enumerate(chunks):
            chunk_name = f"{file_name}-{i}.wav" # clips stored in filename_clip_i format
            chunk_name = os.path.join(args.output_path, chunk_name)
            # print("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")