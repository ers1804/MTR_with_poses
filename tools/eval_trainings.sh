source /home/erik/venv/mtr_pose/bin/activate

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+20_percent_data_jepa_with_decoder.yaml --batch_size=40 --extra_tag=Normal_Loss_80 --eval_tag=Normal_Loss_80_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Normal_Loss_80/ckpt --save_to_file --workers=8 --eval_all

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+20_percent_data_jepa_with_decoder.yaml --batch_size=40 --extra_tag=Normal_Loss_40 --eval_tag=Normal_Loss_40_eval --ckpt_dir=/home/erik/NAS/personal/jepa_eval/complete_experiments/Normal_Loss_40/ckpt --save_to_file --workers=8 --eval_all