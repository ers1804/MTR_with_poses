source /home/erik/venv/mtr_pose/bin/activate

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=40 --extra_tag=Base_Full_40 --eval_tag=Base_Full_40_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Base_Full_40/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=50 --extra_tag=Base_Full_30 --eval_tag=Base_Full_30_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Base_Full_30/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_512 --eval_tag=Full_Training_1_1_0001_512_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_512/ckpt --save_to_file --workers=8 --eval_all --set MODEL.CONTEXT_ENCODER.D_MODEL 512

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_100_Ori_LR --eval_tag=Full_Training_1_1_0001_100_Ori_LR_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_100_Ori_LR/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_150_40_Epochs --eval_tag=Full_Training_1_1_0001_150_40_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_150_40_Epochs/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=30 --extra_tag=Base_Full_40_512 --eval_tag=Base_Full_40_512_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Base_Full_40_512/ckpt --save_to_file --workers=8 --eval_all --set MODEL.CONTEXT_ENCODER.D_MODEL 512

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=30 --extra_tag=Base_20 --eval_tag=Base_20_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Base_20/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_20_1_1_0001_80_50_Epochs --eval_tag=Full_Training_20_1_1_0001_80_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_20_1_1_0001_80_50_Epochs/ckpt --save_to_file --workers=8 --eval_all

# torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_20_1_1_0001_100_50_Epochs --eval_tag=Full_Training_20_1_1_0001_100_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_20_1_1_0001_100_50_Epochs/ckpt --save_to_file --workers=8 --eval_all

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_20_1_1_0001_512_150_50_Epochs --eval_tag=Full_Training_20_1_1_0001_512_150_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_20_1_1_0001_512_150_50_Epochs/ckpt --save_to_file --workers=8 --eval_all --set MODEL.CONTEXT_ENCODER.D_MODEL 512

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_512_C_LR --eval_tag=Full_Training_1_1_0001_512_C_LR_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_512_C_LR/ckpt --save_to_file --workers=8 --eval_all --set MODEL.CONTEXT_ENCODER.D_MODEL 512

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_20_1_1_0001_512_80_50_Epochs --eval_tag=Full_Training_20_1_1_0001_512_80_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_20_1_1_0001_512_80_50_Epochs/ckpt --save_to_file --workers=8 --eval_all --set MODEL.CONTEXT_ENCODER.D_MODEL 512

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_50_Epochs --eval_tag=Full_Training_1_1_0001_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_50_Epochs/ckpt --save_to_file --workers=8 --eval_all

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --extra_tag=Full_Training_1_1_0001_Attn_50_Epochs --eval_tag=Full_Training_1_1_0001_Attn_50_Epochs_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Full_Training_1_1_0001_Attn_50_Epochs/ckpt --save_to_file --workers=8 --eval_all