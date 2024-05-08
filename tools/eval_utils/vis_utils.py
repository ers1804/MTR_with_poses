# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time
import uuid

import numpy as np
np.random.seed(0)
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


def vis_all_agents_smooth(batch_dict, pred_future_states, scenario_id, timestamp=None, agent_id=None):
    if agent_id is None:
        # Extract data for corresponding scenario
        curr_pos = batch_dict['center_objects_world'][:, :2]
        angles = batch_dict['center_objects_world'][:, 6]
        past_trajs_obj = batch_dict['obj_trajs_pos'][0, :, :, :2] + curr_pos[0, :]
        future_trajs_obj = batch_dict['obj_trajs_future_state'][0, :, :, :2] + curr_pos[0, :]
        future_trajs_obj_mask = batch_dict['obj_trajs_future_mask'][0, :, :]
        # Do we have to load the trajectories from before the preparsing??
        # obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
        # center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        # center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
        # map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        #past_traj = batch_dict['obj_trajs_pos'][0, :, :, :2] + curr_pos[0, :] # [num_agents, num_past_steps, 2]
        #gt_traj = batch_dict['center_gt_trajs'][0, :, :2] + curr_pos[0, :] # [num_agents, num_future_steps, 2]
        gt_traj = np.concatenate([pred['gt_trajs'][np.newaxis, :, :2] for pred in pred_future_states[0]], axis=0) # num_center_objects, 91, 2
        gt_trajs_src = batch_dict['center_gt_trajs_src']
        past_traj = gt_traj[:, :11, :] # num_center_objects, 11, 2
        fut_traj = np.concatenate([pred['pred_trajs'][np.newaxis, :, :, :] for pred in pred_future_states[0]], axis=0) # num_center_objects, num_modes, 80, 2
        pred_scores = np.array([pred['pred_scores'] for pred in pred_future_states[0]]) # num_center_objects, num_modes
        obj_ids = [pred['object_id'] for pred in pred_future_states[0]]
        #fut_traj = pred_future_states[0]['pred_trajs'] + curr_pos # [num_agents, num_modes, num_future_steps, 2]
        map_data = batch_dict['map_polylines'][0, :, :, :2]
        num_polylines, num_points, _ = map_data.shape
        map_data = common_utils.rotate_points_along_z(map_data[None, :, :, :].view(1, -1, 2), angle=angles[0].view(1,)).view(num_polylines, num_points, -1)
        map_centers = batch_dict['map_polylines_center'][0, :, :2][:, None, :].repeat(1, map_data.shape[1], 1)
        map_data = map_data + curr_pos[0, :]
        map_mask = batch_dict['map_polylines_mask'][0, :, :]
        

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
        mask = gt_trajs_src[:, :, -1].bool()
        center_y, center_x, width = get_viewport(gt_trajs_src[:, :, :2].numpy(), mask.numpy())
        # width += 50
        # center_x = 2400
        # center_y = 750
        # width = 500
        #map_data = map_data + np.array([center_x, center_y])

        # Plot the scene incl. past and future trajectories
        fig, ax = create_figure_and_axes(size_pixels=1000)
        #rg_plts = map_data[:, :2].reshape(-1, 2).T
        for i in range(map_data.shape[0]):
            ax.plot(map_data[i, map_mask[i], 0], map_data[i, map_mask[i], 1], 'k.', alpha=1, ms=2, c='grey')

        # Plot current position
        if timestamp is None:
            ax.scatter(curr_pos[:, 0], curr_pos[:, 1], c=color_map, s=40)
            for i, pos in enumerate(curr_pos):
                plt.annotate(str(obj_ids[i]), (pos[0], pos[1]), textcoords="offset points", xytext=(0, 10), ha='center')
        else:
            plot_pos = gt_traj[:, 11:, :]
            ax.scatter(plot_pos[:, timestamp, 0], plot_pos[:, timestamp, 1], c=color_map, s=30)
            for i in range(plot_pos.shape[0]):
                plt.annotate(str(obj_ids[i]), (plot_pos[i, timestamp, 0], plot_pos[i, timestamp, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

        # Plot current position of objects
        #ax.scatter(past_trajs_obj[:, 10, 0], past_trajs_obj[:, 10, 1], c='red', s=5)

        # Plot past trajectories
        # for i in range(num_agents):
        #     ax.plot(past_traj[i, :, 0].T, past_traj[i, :, 1].T, c=color_map[i], alpha=0.5, linewidth=3.0)
        mask = gt_trajs_src[:, :11, -1].bool()
        for i in range(num_agents):
            ax.plot(gt_trajs_src[:, :11, :][i, mask[i], 0].numpy().T, gt_trajs_src[:, :11, :][i, mask[i], 1].numpy().T, 'k--', c='green', alpha=0.5)

        # Plot past trajectories of objects
        # for i in range(past_trajs_obj.shape[0]):
        #     ax.plot(past_trajs_obj[i, :, 0].T, past_trajs_obj[i, :, 1].T, 'r--', c='red', alpha=0.5)

        # Plot future trajectories
        indices_ml_mode = np.argmax(pred_scores, axis=1)
        pred_weights = pred_scores / np.max(pred_scores, axis=1)[:, np.newaxis]
        for i in range(num_agents):
            for t in range(num_modes):
                ax.plot(fut_traj[i, t, :, 0].T, fut_traj[i, t, :, 1].T, c=color_map[i], alpha=pred_weights[i, t], linewidth=3.0)
        # for i in range(num_agents):
        #     ax.plot(fut_traj[i, :, :, 0].T, fut_traj[i, :, :, 1].T, c=color_map[i], alpha=0.5, linewidth=3.0)
        
        # Plot future trajectories of objects
        # for i in range(future_trajs_obj.shape[0]):
        #     ax.plot(future_trajs_obj[i, future_trajs_obj_mask[i]==1, 0].T, future_trajs_obj[i, future_trajs_obj_mask[i]==1, 1].T, 'r--', c='red', alpha=0.5)

        # Plot ground truth trajectories
        gt_mask = batch_dict['center_gt_trajs_mask']
        gt_traj = gt_traj[:, 11:, :]
        for i in range(num_agents):
            ax.plot(gt_traj[i, gt_mask[i]==1, 0].T, gt_traj[i, gt_mask[i]==1, 1].T, 'k--',c='green', alpha=0.5)

        # Set Title
        ax.set_title('Scenario: {}'.format(scenario_id[0]))

        size = max(10, width * 1.0)
        ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
        ])
        ax.set_aspect('equal')

        image = fig_canvas_image(fig)
        plt.close(fig)
        return image, pc_sequence, pose_indices
    else:
        # Extract data for corresponding scenario
        obj_ids = [pred['object_id'] for pred in pred_future_states[0]]
        index = obj_ids.index(int(agent_id))
        curr_pos = batch_dict['center_objects_world'][index, :2]
        angles = batch_dict['center_objects_world'][index, 6]
        #past_trajs_obj = batch_dict['obj_trajs_pos'][0, :, :, :2] + curr_pos[0, :]
        #future_trajs_obj = batch_dict['obj_trajs_future_state'][0, :, :, :2] + curr_pos[0, :]
        #future_trajs_obj_mask = batch_dict['obj_trajs_future_mask'][0, :, :]

        gt_traj = np.concatenate([pred['gt_trajs'][np.newaxis, :, :2] for pred in pred_future_states[0]], axis=0)[index, :, :] # num_center_objects, 91, 2
        gt_trajs_src = batch_dict['center_gt_trajs_src'][index, :, :]
        past_traj = gt_traj[:11, :] # num_center_objects, 11, 2
        fut_traj = np.concatenate([pred['pred_trajs'][np.newaxis, :, :, :] for pred in pred_future_states[0]], axis=0)[index, :, :, :] # num_center_objects, num_modes, 80, 2
        pred_scores = np.array([pred['pred_scores'] for pred in pred_future_states[0]])[index, :] # num_center_objects, num_modes
        #fut_traj = pred_future_states[0]['pred_trajs'] + curr_pos # [num_agents, num_modes, num_future_steps, 2]
        map_data = batch_dict['map_polylines'][index, :, :, :2]
        num_polylines, num_points, _ = map_data.shape
        map_data = common_utils.rotate_points_along_z(map_data[None, :, :, :].view(1, -1, 2), angle=angles.view(1,)).view(num_polylines, num_points, -1)
        map_centers = batch_dict['map_polylines_center'][index, :, :2][:, None, :].repeat(1, map_data.shape[1], 1)
        map_data = map_data + curr_pos
        map_mask = batch_dict['map_polylines_mask'][index, :, :]
        

        # Get paths to point clouds and poses
        # obj_types: (num_objects)
        # obj_ids: (num_objects)
        pc_sequence = list()
        pose_indices = list()
        id = agent_id
        type = batch_dict['obj_types'][index]
        if type == 'TYPE_PEDESTRIAN' or type == 'TYPE_CYCLIST':
            pc_list = sorted(glob.glob(os.path.join('/home/erik/raid/datasets/womd/lidar_snippets', 'scenario_id', str(id) + '_*.npy')))
            pc_sequence.append(pc_list)
            pose_indices.append(index)

        # Get infos from data
        num_past_steps, _ = past_traj.shape
        num_modes, num_future_steps, _ = fut_traj.shape

        #color_map = get_colormap(num_agents)

        # Get complete trajectories (past and future modes)
        past_traj_w_modes = np.repeat(past_traj[np.newaxis, :, :], num_modes, axis=0)
        all_traj = np.concatenate([past_traj_w_modes, fut_traj], axis=1)

        # Get viewport (size of map), assume every timestep is valid
        mask = gt_trajs_src[:, -1].bool()
        center_y, center_x, width = get_viewport(gt_trajs_src[:, :2].numpy(), mask.numpy())
        width += 25
        # width += 50
        # center_x = 2400
        # center_y = 750
        # width = 500
        #map_data = map_data + np.array([center_x, center_y])

        # Plot the scene incl. past and future trajectories
        fig, ax = create_figure_and_axes(size_pixels=1000)
        #rg_plts = map_data[:, :2].reshape(-1, 2).T
        ax.plot(map_data[:, :, 0][map_mask], map_data[:, :, 1][map_mask], 'k.', alpha=1, ms=2, c='grey')

        # Plot current position
        if timestamp is None:
            ax.scatter(curr_pos[0], curr_pos[1], c='blue', s=40)
            plt.annotate(str(obj_ids[index]), (curr_pos[0], curr_pos[1]), textcoords="offset points", xytext=(0, 10), ha='center')
        else:
            plot_pos = gt_traj[11:, :]
            ax.scatter(plot_pos[timestamp, 0], plot_pos[timestamp, 1], c='blue', s=30)

            plt.annotate(str(obj_ids[index]), (plot_pos[timestamp, 0], plot_pos[timestamp, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

        # Plot current position of objects
        #ax.scatter(past_trajs_obj[:, 10, 0], past_trajs_obj[:, 10, 1], c='red', s=5)

        # Plot past trajectories
        # for i in range(num_agents):
        #     ax.plot(past_traj[i, :, 0].T, past_traj[i, :, 1].T, c=color_map[i], alpha=0.5, linewidth=3.0)
        mask = gt_trajs_src[:11, -1].bool()
        ax.plot(gt_trajs_src[:11, :][mask, 0].numpy().T, gt_trajs_src[:11, :][mask, 1].numpy().T, 'k--', c='green', alpha=0.5)

        # Plot past trajectories of objects
        # for i in range(past_trajs_obj.shape[0]):
        #     ax.plot(past_trajs_obj[i, :, 0].T, past_trajs_obj[i, :, 1].T, 'r--', c='red', alpha=0.5)

        # Plot future trajectories
        #indices_ml_mode = np.argmax(pred_scores, axis=1)
        pred_weights = pred_scores / np.max(pred_scores)
        for t in range(num_modes):
            ax.plot(fut_traj[t, :, 0].T, fut_traj[t, :, 1].T, c='blue', alpha=pred_weights[t], linewidth=3.0)
        # for i in range(num_agents):
        #     ax.plot(fut_traj[i, :, :, 0].T, fut_traj[i, :, :, 1].T, c=color_map[i], alpha=0.5, linewidth=3.0)
        
        # Plot future trajectories of objects
        # for i in range(future_trajs_obj.shape[0]):
        #     ax.plot(future_trajs_obj[i, future_trajs_obj_mask[i]==1, 0].T, future_trajs_obj[i, future_trajs_obj_mask[i]==1, 1].T, 'r--', c='red', alpha=0.5)

        # Plot ground truth trajectories
        gt_mask = batch_dict['center_gt_trajs_mask'][index]
        gt_traj = gt_traj[11:, :]
        # for i in range(num_agents):
        ax.plot(gt_traj[gt_mask==1, 0].T, gt_traj[gt_mask==1, 1].T, 'k--', c='green', alpha=0.5)

        # Set Title
        ax.set_title('Scenario: {}'.format(scenario_id[0]))

        size = max(10, width * 1.0)
        ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
        ])
        ax.set_aspect('equal')

        image = fig_canvas_image(fig)
        plt.close(fig)
        return image, pc_sequence, pose_indices


def vis_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50, agent_ids=None):
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

    # Visualize results
    if agent_ids is None:
        img, pc_sequence, pose_indices = vis_all_agents_smooth(batch_dict['input_dict'], pred_dicts, batch_dict['input_dict']['scenario_id'], timestamp=None, agent_id=None)
        pil_img = Image.fromarray(img)
        pil_img.save(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_vis_0.png'))
        with open(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_pc_sequence.txt'), 'w') as f:
                for pc_list in pc_sequence:
                    for pc in pc_list:
                        f.write(pc)
                        f.write('\n')
        with open(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_pose_indices.txt'), 'w') as f:
            for pose_idx in pose_indices:
                f.write(str(pose_idx))
                f.write('\n')
        timestamps = [29, 59, 79]
        for timestamp in timestamps:
            img, pc_sequence, pose_indices = vis_all_agents_smooth(batch_dict['input_dict'], pred_dicts, batch_dict['input_dict']['scenario_id'], timestamp=timestamp)
            pil_img = Image.fromarray(img)
            pil_img.save(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_vis_' + str(timestamp) + '.png'))
    else:
        for agent_id in agent_ids:
            img, pc_sequence, pose_indices = vis_all_agents_smooth(batch_dict['input_dict'], pred_dicts, batch_dict['input_dict']['scenario_id'], timestamp=None, agent_id=agent_id)
            pil_img = Image.fromarray(img)
            pil_img.save(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_' + str(agent_id) + '_vis_0.png'))
            with open(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_' + str(agent_id) + '_pc_sequence.txt'), 'w') as f:
                    for pc_list in pc_sequence:
                        for pc in pc_list:
                            f.write(pc)
                            f.write('\n')
            with open(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_' + str(agent_id) + '_pose_indices.txt'), 'w') as f:
                for pose_idx in pose_indices:
                    f.write(str(pose_idx))
                    f.write('\n')
            timestamps = [29, 59, 79]
            for timestamp in timestamps:
                img, pc_sequence, pose_indices = vis_all_agents_smooth(batch_dict['input_dict'], pred_dicts, batch_dict['input_dict']['scenario_id'], timestamp=timestamp, agent_id=agent_id)
                pil_img = Image.fromarray(img)
                pil_img.save(result_dir / (str(batch_dict['input_dict']['scenario_id'][0]) + '_' + str(agent_id) + '_vis_' + str(timestamp) + '.png'))


    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
