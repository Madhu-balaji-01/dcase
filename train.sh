#!/bin/bash
python ensemble_train.py --epoch 1 --save_model_address ./model_zoo/dcase_tests/ --n_mels 500
# wait
# python ensemble_train2.py --epoch 1 --save_model_address ./model_zoo/dcase_tests/ --n_mels 500
wait
python main_train.py --epoch 1 --save_model_address ./model_zoo/dcase_tests/ --network dcase1 --method post --mono diff --n_mels 500
# wait
# python main_train.py --epoch 1 --save_model_address ./model_zoo/dcase_tests/ --network dcase2 --method post --mono diff --n_mels 500