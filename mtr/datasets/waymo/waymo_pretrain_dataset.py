"""
WaymoPretrainDataset: full-Waymo pre-training using the NAS agent-batch processed
scenarios (sample_{sid}_N.pkl format, ~2.17M pedestrian+vehicle+cyclist agent
entries across 487k train scenes).

The only difference from WaymoDataset is get_all_infos, which normalises the
scenario_id field so that self.data_path / f'sample_{scene_id}.pkl' resolves to
the correct file:

  Old info format:  'sample_4b60f9400a30ceaf_0.pkl'  →  '4b60f9400a30ceaf_0'
  New info format:  '4b60f9400a30ceaf_0'              →  '4b60f9400a30ceaf_0'  (no-op)
"""

import pickle
from mtr.datasets.waymo.waymo_dataset import WaymoDataset


class WaymoPretrainDataset(WaymoDataset):

    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        # Normalise scenario_id to bare '{hex}_{N}' (the stem of the PKL filename).
        # Old format: 'sample_4b60f9400a30ceaf_0.pkl' → '4b60f9400a30ceaf_0'
        # New format: '4b60f9400a30ceaf_0'             → unchanged
        for info in src_infos:
            sid = info['scenario_id']
            if sid.startswith('sample_'):
                sid = sid[len('sample_'):]
            if sid.endswith('.pkl'):
                sid = sid[:-4]
            info['scenario_id'] = sid

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        self.logger.info(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos
