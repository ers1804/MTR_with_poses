import pickle
import tqdm

def main():
    with open('/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_val_infos_agent.pkl', 'rb') as f:
        data = pickle.load(f)
    for scenario in tqdm.tqdm(data):
        scenario['scenario_id'] = scenario['scenario_id'].split('_')[1] + "_" + scenario['scenario_id'].split('_')[2].split('.')[0]
        #print(scenario)
    with open('/home/erik/NAS/personal/waymo_agent_batch/processed_scenarios_val_infos_agent_new.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()