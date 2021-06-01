#!/bin/bash
set -v
python ensemble_train.py --save_model_address ./model_zoo/resnet34/ 
# python feature_extract.py --input_path ./dataset/dcase/evaluation_setup/modify_evaluate.csv --output_path  ./audio_features/
wait
exit
# python main_train.py --save_model_address ./model_zoo/dcase_24/ --method post --mono mean
# wait
# python main_train.py --save_model_address ./model_zoo/dcase_25/ --method post --mono diff
# wait
# python main_train.py --save_model_address ./model_zoo/dcase_26/ --method pre --mono mean
# wait
# python main_train.py --save_model_address ./model_zoo/dcase_27/ --method pre --mono diff


# python main_train.py --save_model_address ./model_zoo/dcase1_spec_aug/ --method post --mono mean --network dcase1 --spec_aug True --spectra spectrum --epoch 40 --batch_size 14
# wait
# python main_train.py --save_model_address ./model_zoo/dcase1_manipulation/ --method post --mono mean --network dcase1 --manipulate True --n_mels 500 --batch_size 14
# wait

# python ensemble_train.py --save_model_address ./model_zoo/dcase_ensemble_14/ --method post --mono mean --win_len 1024 --hop_len 102 --alpha 0.2 --epoch 30 --n_mels 500 --batch_size 14
# wait
# wait
# python main_train.py --save_model_address ./model_zoo/baseline_mixup/ --network baseline --n_mels 500
# python ensemble_train.py --save_model_address ./model_zoo/dcase_ensemble_13/ --method post --mono mean --win_len 1024 --hop_len 102 --alpha 0.2 --epoch 50
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
# wait
# python main_train.py --save_model_address ./model_zoo/dcase_7_spec_aug/ --method post --mono mean --network vgg_m2 --spec_aug True
# wait
# python main_train.py --save_model_address ./model_zoo/dcase_7_mixup0.3/ --method post --mono mean --network vgg_m2 --alpha=0.3 --epoch 50
# wait
# python main_train.py --save_model_address ./model_zoo/dcase_7_mixup0.2/ --method post --mono mean --network vgg_m2 --alpha=0.2 --epoch 50
# wait
# python ensemble_train5.py --save_model_address ./model_zoo/dcase_ensemble_9/ --method post --win_len 1024 --hop_len 102
# wait

# wait

# python main_train.py --save_model_address ./model_zoo/dcase_7_mixup0.1/ --method post --mono mean --network vgg_m2 --alpha=0.1 --epoch 30
# wait
