import os
import torch
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load_model(path, device='cpu'):
    from baselines.dqn.net import Network
    with open(os.path.join(path, 'config.json')) as f:
        config = json.load(f)
    net = Network(**config)
    net.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=device))
    net.eval()
    return net


def calc_percentile(a, method='min'):
    if isinstance(a, list):
        a = np.asarray(a)
    return rankdata(a, method=method) / float(len(a))

def get_postone_features(instance_observation, instance_static_info):
    BIGINT = 1e9
    
    # Features compared with must_dispatch
    epoch_instance = instance_observation['epoch_instance']
    must_dispatch = epoch_instance['must_dispatch']
    customer_idx = epoch_instance['customer_idx']
    idx_dispatchs = np.where(must_dispatch==True)[0]
    idx_lefts = np.where((must_dispatch==False)&(customer_idx!=0))[0]
    

    duration_matrix = epoch_instance['duration_matrix']
    avg_dur_with_must_dispatch = np.ones(len(idx_lefts))*BIGINT
    min_dur_with_must_dispatch = np.ones(len(idx_lefts))*BIGINT
    if len(idx_dispatchs) > 0:
        for idx, idx_left in enumerate(idx_lefts):
            dur_to_must_dispatch = duration_matrix[idx_left,:][idx_dispatchs]
            dur_from_must_dispatch = duration_matrix[:,idx_left][idx_dispatchs]
            dur_with_must_dispatch = np.concatenate((dur_to_must_dispatch,dur_to_must_dispatch))
            
            avg_dur_with_must_dispatch[idx] = np.mean(dur_with_must_dispatch)
            min_dur_with_must_dispatch[idx] = np.min(dur_with_must_dispatch)


    # Features compared with instances
    # customer_must_dispatch_idx = customer_idx[np.array(must_dispatch)]
    # customer_left_idx = customer_idx[~np.array(must_dispatch)]

    instance_duration_matrix = instance_static_info['dynamic_context']['duration_matrix']
    num_customer = instance_duration_matrix.shape[0]
    instance_duration_arr = instance_duration_matrix.reshape(-1,1)
    rank_duration_matrix = calc_percentile(instance_duration_arr).reshape(num_customer,num_customer)

    rank_duration_in_request = np.ones((len(idx_lefts),100))*BIGINT
    for idx,customer in enumerate(idx_lefts):
        col_rank_duration = rank_duration_matrix[:,customer]
        row_rank_duration = rank_duration_matrix[customer,:]
        min_rank_duration = np.amin([col_rank_duration,row_rank_duration],axis=0)
        
        other_cust_index = [cust for cust in customer_idx if cust not in [customer]]
        rank_duration = min_rank_duration[np.array(other_cust_index)]
        rank_duration = sorted(rank_duration)[0:100]
        
        rank_duration_in_request[idx,0:len(rank_duration)] = rank_duration


    # Capacity and Demand
    demands = epoch_instance['demands']
    demand_left = (demands[idx_lefts])
    avg_demand = np.mean(demands)
    total_demand_must_dispatch = np.sum(demands[idx_dispatchs])
    capacity = epoch_instance['capacity']
    

    # Epoch left
    current_epoch = instance_observation['current_epoch']
    end_epoch = instance_static_info['end_epoch']
    rem_epoch = end_epoch - current_epoch

    
    features = np.concatenate([
        avg_dur_with_must_dispatch[:,None], min_dur_with_must_dispatch[:,None],
        rank_duration_in_request,
        demand_left[:,None],
        np.repeat(avg_demand,len(idx_lefts))[:,None],
        np.repeat(total_demand_must_dispatch,len(idx_lefts))[:,None],
        np.repeat(capacity,len(idx_lefts))[:,None],
        np.repeat(rem_epoch,len(idx_lefts))[:,None]
        ],-1)
    
    return(features)

def get_df_features(instance_observation, instance_static_info):
    features = get_postone_features(instance_observation, instance_static_info)

    columns = ['avg_dur_with_must_dispatch', 'min_dur_with_must_dispatch',
                        *[f'rank_duration_{i}' for i in range(1,101)],
                        'demand_left', 'avg_demand', 'total_demand_must_dispatch', 'capacity', 'rem_epoch']
    
    df_features = pd.DataFrame(features,columns=columns)
    return(df_features)