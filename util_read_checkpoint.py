# Controller has an environment and tests it against a dynamic solver program
import argparse
import os
import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='checkpoint.pkl', help="Path of checkpoint")
    parser.add_argument("--strategy", type=str, default='instance_dqn', help="Baseline strategy used to decide whether to dispatch routes")
    parser.add_argument("--instance_path", help="Instance to solve")
    parser.add_argument("--instance_seed", type=str, default="1,2,3,4,5", help="Seed to use for the dynamic instance (or multiple comma seperated), default 1,2,3,4,5")
    parser.add_argument("--num_instance", type=int, default=250, help="Number of instances that would be evaluted")
    parser.add_argument("--num_epoch_postpone", type=int, default=1, help="Number of epochs that would be postponed")
    parser.add_argument("--postpone_seed", type=int, default=1, help="Index of customer that would be postponed")
    parser.add_argument("--solver_seed", type=int, default=1, help="Seed to use for the solver")
    parser.add_argument("--epoch_tlim", type=int, default=120, help="Time limit per epoch")
    parser.add_argument("--oracle_tlim", type=int, default=120, help="Time limit for oracle")
    parser.add_argument("--tmp_dir", type=str, default=None, help="Provide a specific directory to use as tmp directory (useful for debugging)")
    parser.add_argument("--verbose", action='store_true', help="Show verbose output")
    parser.add_argument("--data_dir", default='baselines/supervised/data')
    args = parser.parse_args()

    tmp = pickle.load(open(args.path,'rb'))
    list_features = tmp[0]
    list_cost = tmp[1]

    if args.strategy == 'instance_dqn':
        features = np.vstack(list_features)
        Q_list = np.vstack(list_cost)

        training_set = pd.DataFrame(np.hstack((features, Q_list)))
        training_set.to_csv("baselines/instance_dqn/pretrained/dqn_training_set.csv", index=False)

    else:
        if args.strategy == 'collectpostone':
            columns = ['avg_dur_with_must_dispatch', 'min_dur_with_must_dispatch',
                        *[f'rank_duration_{i}' for i in range(1,101)],
                        'demand_left', 'avg_demand', 'total_demand_must_dispatch', 'capacity', 'rem_epoch']
            
            store_features = []
            for feat in list_features:
                if feat is None:
                    feat = []
                store_features.append(feat)
            df = pd.DataFrame.from_records(store_features,columns=columns)
        else:
            df = pd.DataFrame()
        
        num_records = df.shape[0]
        list_instances = tmp[2]
        instance_seed = args.instance_seed
        num_instances = args.num_instance
        num_epoch = args.num_epoch_postpone
        postpone_seed = args.postpone_seed
        df['cost'] = list_cost
        df['instance_seed'] = np.repeat(instance_seed,num_instances*num_epoch*postpone_seed)[0:num_records]
        df['instance'] = np.repeat(list_instances,num_epoch*postpone_seed)[0:num_records]
        df['postpone_epoch'] = np.tile(np.repeat(np.arange(1,num_epoch+1),postpone_seed),num_instances)[0:num_records]
        df['postpone_seed'] = np.tile(np.arange(1,postpone_seed+1),num_instances*num_epoch)[0:num_records]
        
        print(df[~df.instance.str.startswith('ORTEC')]['instance'])
        df.to_csv(f'misc/data_{args.strategy}_{instance_seed}.csv',index=False)

# python util_read_checkpoint.py --strategy collectpostone --path misc/checkpoint.pkl --instance_seed 1 --num_instance 1 --num_epoch_postpone 1 --postpone_seed 1