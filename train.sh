#!/bin/bash
set -v
python main_train.py --save_model_address ./model_zoo/dcase_14/
wait
python ensemble_train.py --save_model_address ./model_zoo/dcase_ensemble_5/ --n_mels 500
wait 
python main_train.py --save_model_address ./model_zoo/dcase_15/
wait
python ensemble_train2.py --save_model_address ./model_zoo/dcase_ensemble_5/ --n_mels 500