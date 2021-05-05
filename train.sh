#!/bin/bash
set -v

python main_train.py --save_model_address ./model_zoo/dcase_7/ --n_mels 500 --method pre
wait
python ensemble_train.py --save_model_address ./model_zoo/dcase_ensemble_3/ --n_mels 500