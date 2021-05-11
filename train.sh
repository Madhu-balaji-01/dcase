#!/bin/bash
set -v
python main_train.py --save_model_address ./model_zoo/dcase_24/ --method post --mono mean
wait
python main_train.py --save_model_address ./model_zoo/dcase_25/ --method post --mono diff
wait
python main_train.py --save_model_address ./model_zoo/dcase_26/ --method pre --mono mean
wait
python main_train.py --save_model_address ./model_zoo/dcase_27/ --method pre --mono diff