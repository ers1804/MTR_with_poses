import os,sys,pickle,numpy as np
REPO='/home/erik/ssd2/gitprojects/MTR_with_poses'
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,'tools'))
sys.path.insert(0,os.path.join(REPO,'tools','scripts'))
from analyze_multiseed import CELLS,OUT_ROOT,SEEDS,hierarchical_bootstrap,load_agent_metrics,run_summary
CACHE='/tmp/pose_quality_feats.pkl'
if os.path.exists(CACHE):
    feats=pickle.load(open(CACHE,'rb'))
else:
    import logging
    from mtr.config import cfg,cfg_from_yaml_file
    from mtr.datasets.waymo.waymo_pose_dataset import WaymoPoseDataset
    log=logging.getLogger('q'); log.addHandler(logging.NullHandler())
    cfg_from_yaml_file(os.path.join(REPO,'tools/cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map.yaml'),cfg)
    ds=WaymoPoseDataset(dataset_cfg=cfg.DATA_CONFIG,training=False,logger=log)
    feats={}
    for i in range(len(ds)):
        try: d=ds[i]
        except Exception: continue
        tix=d['track_index_to_predict']
        for c in range(len(d['center_objects_id'])):
            key=(str(d['scenario_id'][c]),str(d['center_objects_id'][c]))
            m=d['obj_poses_mask'][c,tix[c]]           # (11,) own pose validity
            p=d['obj_poses'][c,tix[c]]                # (11,144)
            n=int(m.sum())
            jit=0.0
            if n>=2:
                v=p[m]; jit=float(np.linalg.norm(np.diff(v,axis=0),axis=-1).mean())
            feats[key]={'n_pose':n,'jitter':jit}
        if (i+1)%1000==0: print(f'  {i+1} scenes',flush=True)
    pickle.dump(feats,open(CACHE,'wb'))

def ps(cell):
    o={}
    for s in SEEDS:
        rd=os.path.join(OUT_ROOT,CELLS[cell],f'MS_{cell}_s{s}')
        r=run_summary(rd)
        if r is None: continue
        m=load_agent_metrics(rd,r['best_epoch'])
        if m: o[s]={k:v[0] for k,v in m.items()}
    return o
cells={c:ps(c) for c in ['map_nopose','map_xattn_pe','map_shufflepose','map_constpose']}
keys=sorted(set.intersection(*[set(d) for c in cells.values() for d in c.values()]) & set(feats))
print(f'\n{len(keys)} agents')
npos=np.array([feats[k]['n_pose'] for k in keys])
print('valid past-pose frames per pedestrian (of 11):')
for v in range(12):
    c=(npos==v).sum()
    if c: print(f'   {v:2d} frames: {c:5d} ({100*c/len(keys):.1f}%)')
print(f'  mean={npos.mean():.2f}  median={np.median(npos):.0f}')

def cmp(a,b,sub,label):
    r=hierarchical_bootstrap({s:{k:v for k,v in d.items() if k in sub} for s,d in cells[a].items()},
                             {s:{k:v for k,v in d.items() if k in sub} for s,d in cells[b].items()},n_boot=4000)
    if r: print(f'   {label:38} n={len(sub):5d}  {r["rel_diff_pct"]:+6.2f}%  p={r["p_boot_two_sided"]:.4f}')

print('\n=== does CORRECT beat WRONG where pose is well observed? ===')
for lo in [0,6,8,10,11]:
    sub={k for k in keys if feats[k]['n_pose']>=lo}
    if len(sub)<200: continue
    print(f'\n pedestrians with >= {lo} valid pose frames:')
    cmp('map_nopose','map_xattn_pe',sub,'no-pose -> correct pose')
    cmp('map_shufflepose','map_xattn_pe',sub,'WRONG   -> correct pose  (content!)')
    cmp('map_constpose','map_xattn_pe',sub,'constant-> correct pose  (content!)')
