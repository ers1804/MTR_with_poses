# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time
import uuid

import numpy as np
import torch
import tqdm
import glob
import os

from mtr.utils import common_utils

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image


def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
        all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
        all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
        center_y: float. y coordinate for center of data.
        center_x: float. x coordinate for center of data.
        width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def vis_all_agents_smooth(batch_dict, pred_future_states, scenario_id):
    # Extract data for corresponding scenario
    curr_pos = batch_dict['center_objects_world'][:, :2]
    # Do we have to load the trajectories from before the preparsing??
    # obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
    # center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    # center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
    # map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
    past_traj = batch_dict['obj_trajs_pos'][0, :, :, :2] + curr_pos[0, :] # [num_agents, num_past_steps, 2]
    gt_traj = batch_dict['center_gt_trajs'][0, :, :2] + curr_pos[0, :] # [num_agents, num_future_steps, 2]
    fut_traj = pred_future_states[0]['pred_trajs'] + curr_pos # [num_agents, num_modes, num_future_steps, 2]
    map_data = batch_dict['map_polylines'][0, :, :, :2] + curr_pos[0, :]
    mask_past = None

    # Get paths to point clouds and poses
    # obj_types: (num_objects)
    # obj_ids: (num_objects)
    pc_sequence = list()
    pose_indices = list()
    for i, (id, type) in enumerate(zip(batch_dict['obj_ids'], batch_dict['obj_types'])):
        if type == 'TYPE_PEDESTRIAN' or type == 'TYPE_CYCLIST':
            pc_list = sorted(glob.glob(os.path.join('/home/erik/raid/datasets/womd/lidar_snippets', 'scenario_id', str(id) + '_*.npy')))
            pc_sequence.append(pc_list)
            pose_indices.append(i)

    curr_timestep = 10

    # Get infos from data
    num_agents, num_past_steps, _ = past_traj.shape
    _, num_modes, num_future_steps, _ = fut_traj.shape

    color_map = get_colormap(num_agents)

    # Get complete trajectories (past and future modes)
    past_traj_w_modes = np.repeat(past_traj[:, np.newaxis, :, :], num_modes, axis=1)
    all_traj = np.concatenate([past_traj_w_modes, fut_traj], axis=2)

    # Get viewport (size of map), assume every timestep is valid
    center_y, center_x, width = get_viewport(all_traj, np.ones_like(all_traj[..., 0]))

    # Plot the scene incl. past and future trajectories
    fig, ax = create_figure_and_axes(size_pixels=1000)
    rg_plts = map_data[:, :2].T
    ax.plot(rg_plts[0], rg_plts[1], 'k.', alpha=1, ms=2)

    # Plot current position
    ax.scatter(curr_pos[:, 0], curr_pos[:, 1], c=color_map, s=10)

    # Plot past trajectories
    ax.plot(past_traj[:, :, 0].T, past_traj[:, :, 1].T, c=color_map, alpha=0.5)

    # Plot future trajectories
    ax.plot(fut_traj[:, :, :, 0].T, fut_traj[:, :, :, 1].T, c=color_map, alpha=0.5)

    # Plot ground truth trajectories
    ax.plot(gt_traj[:, :, 0].T, gt_traj[:, :, 1].T, 'k--', alpha=0.5)

    # Set Title
    ax.set_title('Scenario: {}'.format(scenario_id))

    size = max(10, width * 1.0)
    ax.axis([
      -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
      size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)

    return image, pc_sequence, pose_indices


def vis_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
    )

    # Visualize result
    img, pc_sequence, pose_indices = vis_all_agents_smooth(batch_dict, pred_dicts, batch_dict['scenario_id'])
    pil_img = Image.fromarray(img)
    pil_img.save(result_dir / str(batch_dict['scenario_id']) + '_vis.png')
    with open(result_dir / str(batch_dict['scenario_id']) + '_pc_sequence.txt', 'w') as f:
        for pc_list in pc_sequence:
            f.write(pc_list)
            f.write('\n')
    with open(result_dir / str(batch_dict['scenario_id']) + '_pose_indices.txt', 'w') as f:
        for pose_idx in pose_indices:
            f.write(pose_idx)
            f.write('\n')


    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
