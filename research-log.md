# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-04-01 | bootstrap | Explored codebase. MTR extended with SMPL pose: GRU encoder, pose heads, MPJPE+geodesic+NLL losses. Data: 579 training scenes at final_10fps/training/. Found critical timestamp bug (µs vs s). Formed 5 hypotheses. Fix config DATA_ROOT path. |
| 2 | 2026-04-01 | bootstrap | Fixed timestamp unit bug in waymo_pose_dataset.py (divide by 1e6). Updated mtr+pose_data.yaml DATA_ROOT to actual data path. Set up autoresearch workspace. Set up /loop 20m for agent continuity. |
| 3 | 2026-04-01 | bugfix | Fixed SMPL betas mismatch: SMPL_NEUTRAL.pkl has 300-dim shapedirs but decoder passes 10-dim betas. Fix: truncate th_shapedirs[:,:,:10] and th_betas[:,:10] in mtr_decoder.py after SMPL_Layer init. |
| 4 | 2026-04-01 | bugfix | Fixed dataset IndexError: get_interested_agents fallback selected future-only agents (no valid past), causing valid_past_mask=False → track_index=-1. Fix: fallback now requires past-valid agents. |
| 5 | 2026-04-01 | bugfix | Fixed BatchNorm error on batch-size-1 (last batch). Fix: DATALOADER_DROP_LAST=True in config. Added __getitem__ retry on ValueError/IndexError for robustness. |
| 6 | 2026-04-01 | experiment | Launched H1 training run (H1_pose_all_losses): 30 epochs, batch=10, full pose model. Pipeline verified end-to-end. ~80s/epoch → ~40min total. |
