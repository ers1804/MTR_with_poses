source /home/erik/venv/mtr_pose/bin/activate

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=40 --extra_tag=Base_Full_40 --eval_tag=Base_Full_40_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Base_Full_40/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=50 --extra_tag=Base_Full_30 --eval_tag=Base_Full_30_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Base_Full_30/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_512 --eval_tag=Full_Training_1_1_0001_512_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_512/ckpt --save_to_file --workers=8 --eval_all --set MODEL.CONTEXT_ENCODER.D_MODEL 512

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_100_50_Epochs --eval_tag=Full_Training_1_1_0001_100_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_100_50_Epochs/ckpt --save_to_file --workers=8 --eval_all

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_20_1_1_0001_100 --eval_tag=Full_Training_20_1_1_0001_100_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_20_1_1_0001_100/ckpt --save_to_file --workers=8 --eval_all