import shutil
import glob
import os
import tqdm

path_to_womd = '/home/erik/raid/datasets/womd'
scenario_lists = list(os.listdir(os.path.join(path_to_womd, 'lidar_snippets')))
destination_dir = os.path.join(path_to_womd, 'poses')
for scenario in tqdm.tqdm(scenario_lists):
    file_names = ['pose_data.npy', 'pose_data_mask.npy']
    for file in file_names:
        os.remove(os.path.join(path_to_womd, 'lidar_snippets', scenario, file))
        #shutil.copy(os.path.join(path_to_womd, 'lidar_snippets', scenario, file), os.path.join(destination_dir, scenario, file))