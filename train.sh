#!/bin/bash
set -v

# python ensemble_train.py --save_model_address ./model_zoo/dcase_ensemble_3/ --method post --mono mean --win_len 1024 --hop_len 102
# wait
# python ensemble_train2.py --save_model_address ./model_zoo/dcase_ensemble_4/ --method post --mono mean --win_len 1024 --hop_len 102
# wait
# python ensemble_train.py --save_model_address ./model_zoo/dcase_ensemble_5/ --method post --mono diff --win_len 1024 --hop_len 102
# wait
# python ensemble_train2.py --save_model_address ./model_zoo/dcase_ensemble_6/ --method post --mono diff --win_len 1024 --hop_len 102
# wait
# python ensemble_train3.py --save_model_address ./model_zoo/dcase_ensemble_7/ --method post --win_len 1024 --hop_len 102
# wait
# python ensemble_train4.py --save_model_address ./model_zoo/dcase_ensemble_8/ --method post --win_len 1024 --hop_len 102
# wait
# python ensemble_train5.py --save_model_address ./model_zoo/dcase_ensemble_9/ --method post --win_len 1024 --hop_len 102
# wait
# python ensemble_train3.py --save_model_address ./model_zoo/dcase_ensemble_10/ --method pre --win_len 1024 --hop_len 102
# wait
# python ensemble_train4.py --save_model_address ./model_zoo/dcase_ensemble_11/ --method pre --win_len 1024 --hop_len 102
# wait
# python ensemble_train5.py --save_model_address ./model_zoo/dcase_ensemble_12/ --method pre --win_len 1024 --hop_len 102
pip install SpecAugment
wait
python main_train.py --save_model_address ./model_zoo/dcase_7_spec_aug/ --method post --mono mean --network vgg_m2 --spec_aug True
wait
python main_train.py --save_model_address ./model_zoo/dcase_7_mixup0.3/ --method post --mono mean --network vgg_m2 --alpha=03
wait
python main_train.py --save_model_address ./model_zoo/dcase_7_mixup/ --method post --mono mean --network vgg_m2 --alpha=0.2