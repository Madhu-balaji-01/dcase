#!/bin/bash
set -v
python main_train.py --save_model_address ./model_zoo/dcase_16/ --method post --mono mean --win_len 1024 --hop_len 102
wait
python main_train.py --save_model_address ./model_zoo/dcase_17/ --method post --mono mean --win_len 1024 --hop_len 512
wait
python main_train.py --save_model_address ./model_zoo/dcase_18/ --method post --mono mean --win_len 2048 --hop_len 204
wait
python main_train.py --save_model_address ./model_zoo/dcase_19/ --method post --mono mean --win_len 2048 --hop_len 1024
wait
python main_train.py --save_model_address ./model_zoo/dcase_20/ --method post --mono diff --win_len 1024 --hop_len 102
wait
python main_train.py --save_model_address ./model_zoo/dcase_21/ --method post --mono diff --win_len 1024 --hop_len 512
wait
python main_train.py --save_model_address ./model_zoo/dcase_22/ --method pre --mono mean --win_len 1024 --hop_len 102
wait
python main_train.py --save_model_address ./model_zoo/dcase_23/ --method pre --mono diff --win_len 1024 --hop_len 102