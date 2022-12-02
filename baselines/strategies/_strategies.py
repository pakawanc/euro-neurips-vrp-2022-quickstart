import numpy as np
from environment import State


def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key in ('observation', 'static_info'):
            continue

        if key in ('capacity', 'top_must_dispatch_node', 'top_node', 'features', 'is_postpone'):
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res


def _greedy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[:] = True
    return _filter_instance(observation, mask)


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


def _supervised(observation: State, rng: np.random.Generator, net):
    from baselines.supervised.transform import transform_one
    mask = np.copy(observation['must_dispatch'])
    mask = mask | net(transform_one(observation)).argmax(-1).bool().numpy()
    mask[0] = True
    return _filter_instance(observation, mask)


def _dqn(observation: State, rng: np.random.Generator, net):
    import torch
    from baselines.dqn.utils import get_request_features
    actions = []
    epoch_instance = observation
    observation, static_info = epoch_instance.pop('observation'), epoch_instance.pop('static_info')
    request_features, global_features = get_request_features(observation, static_info, net.k_nearest)
    all_features = torch.cat((request_features, global_features[None, :].repeat(request_features.shape[0], 1)), -1)
    actions = net(all_features).argmax(-1).detach().cpu().tolist()
    mask = epoch_instance['must_dispatch'] | (np.array(actions) == 0)
    mask[0] = True  # Depot always included in scheduling
    return _filter_instance(epoch_instance, mask)


def _cluster(observation: State, rng: np.random.Generator):
    clusters,counts = np.unique(observation['cluster'], return_counts=True)
    # most_cluster = clusters[np.argmax(counts)]
    least_cluster = clusters[np.argmin(counts)]
    mask = np.copy(observation['must_dispatch'])
    # mask = (mask | (observation['cluster']==most_cluster))
    mask = (mask | (observation['cluster']!=least_cluster))
    mask[0] = True
    return _filter_instance(observation, mask)


def _topnode(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    is_in_topnode = [(item in observation['top_must_dispatch_node']) for item in observation['customer_idx']]
    is_in_topnode = np.array(is_in_topnode)
    mask = (mask | (is_in_topnode))
    mask[0] = True
    return _filter_instance(observation, mask)


def _topcluster(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])

    clusters,counts = np.unique(observation['cluster'], return_counts=True)
    # most_cluster = clusters[np.argmax(counts)]
    least_cluster = clusters[np.argmin(counts)]
    is_in_cluster = observation['cluster']==least_cluster

    is_in_topnode = [(item in observation['top_must_dispatch_node']) for item in observation['customer_idx']]
    is_in_topnode = np.array(is_in_topnode)

    # mask = (mask | (observation['cluster']==most_cluster))
    mask = (mask | (~is_in_cluster & is_in_topnode))
    mask[0] = True
    return _filter_instance(observation, mask)


def _connectdot(observation: State, rng: np.random.Generator):
    cust_in_line = []
    # cust_in_queue = [0]
    cust_in_queue = []
    customer_idx = observation['customer_idx']
    top_node = {int(k):v for k,v in observation['top_node'].items()}

    cust_must_dispatch = customer_idx[observation['must_dispatch']]
    cust_in_queue = cust_in_queue + cust_must_dispatch.tolist()

    while(len(cust_in_queue) > 0):    
        check_cust_idx = cust_in_queue[0]
        idx_cust_in_arr = np.where(customer_idx == check_cust_idx)[0][0]
        
        top_custs = top_node[check_cust_idx]
        is_in_topcust = [(item in top_custs) for item in customer_idx]
        is_in_topcust = np.array(is_in_topcust)

        is_same_cluster = [(item == observation['cluster'][idx_cust_in_arr]) for item in observation['cluster']]
        is_same_cluster = np.array(is_same_cluster)

        cust_in_line.append(check_cust_idx)
        # cust_to_check = customer_idx[is_in_topcust & is_same_cluster].tolist()
        cust_to_check = customer_idx[is_in_topcust].tolist()
        cust_in_queue += cust_to_check
        cust_in_queue = list(set(cust_in_queue)-set(cust_in_line))
    
    is_in_line = np.array([(item in cust_in_line) for item in customer_idx])
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | is_in_line)
    mask[0] = True
    return _filter_instance(observation, mask)


def _modcutoff(observation: State, rng: np.random.Generator):
  
    mask = np.copy(observation['must_dispatch'])
    obs_duration = observation['duration_matrix']
    isDispatch = np.full(len(mask), False)
    current_epoch = observation['observation']['current_epoch']
    distance_half = np.triu(obs_duration)
    distance_half = np.delete(distance_half.reshape(-1,),0)
    quantile = 0.7+current_epoch*0.08
    if  quantile > 1:
        quantile = 1
    cutoff=np.quantile(distance_half, quantile)

    for i in [idx for idx, m in enumerate(mask) if m==True]:
        for j in range(len(mask)):
            dum = max(obs_duration[i,j], obs_duration[j,i]) < cutoff
            isDispatch[j] = (isDispatch[j] | dum)
    mask = (mask | isDispatch)

    mask[0] = True
    return _filter_instance(observation, mask)


def _collectpostone(observation: State, rng: np.random.Generator, epoch_postpone=1, postpone_seed=1):
    mask = np.copy(observation['must_dispatch'])
    mask[:] = True

    epoch_instance = observation
    req_observation, static_info = epoch_instance.pop('observation'), epoch_instance.pop('static_info')
    cust_left = np.where((observation['must_dispatch']==False) & (observation['customer_idx']!=0))[0]
    count_epoch = req_observation['current_epoch'] - static_info['start_epoch'] + 1
    if (len(cust_left) > 0) & (count_epoch == epoch_postpone):
        # Get index of non must_dispatch to be postponed
        idx_postponed = (min(postpone_seed,len(cust_left)) - 1)
        mask[cust_left[idx_postponed]] = False

        # Get features
        from baselines.postone.utils import get_postone_features
        features = get_postone_features(req_observation, static_info)
        observation['features'] = features[idx_postponed].tolist()
    
    return _filter_instance(observation, mask)


def _postall(observation: State, rng: np.random.Generator, net):
    mask = np.copy(observation['must_dispatch'])
    mask[:] = True

    epoch_instance = observation
    req_observation, static_info = epoch_instance.pop('observation'), epoch_instance.pop('static_info')
    cust_left = np.where((observation['must_dispatch']==False) & (observation['customer_idx']!=0))[0]
    if (len(cust_left) > 0) & (req_observation['current_epoch'] < static_info['end_epoch']):
        # Predict prob of only customers which must not dispatched
        from baselines.postone.utils import get_df_features
        features = get_df_features(req_observation, static_info)
        score_cust_left = net.predict_proba(features)[:1]
        
        idx_cust_left = np.argmax(score_cust_left)
        cust_left = np.where((observation['must_dispatch']==False) & (observation['customer_idx']!=0))[0]
        idx_delay = cust_left[idx_cust_left]
        
        mask[idx_delay] = False
    return _filter_instance(observation, mask)

def _postone(observation: State, rng: np.random.Generator, is_postpone, net):
    mask = np.copy(observation['must_dispatch'])
    mask[:] = True

    epoch_instance = observation
    req_observation, static_info = epoch_instance.pop('observation'), epoch_instance.pop('static_info')
    
    cust_left = np.where((observation['must_dispatch']==False) & (observation['customer_idx']!=0))[0]
    not_end_epoch = (req_observation['current_epoch'] < static_info['end_epoch'])

    observation['is_postpone'] = is_postpone
    if (len(cust_left) > 0) & not_end_epoch & (not is_postpone):
        # Predict prob of only customers which must not dispatched
        from baselines.postone.utils import get_df_features
        features = get_df_features(req_observation, static_info)
        score_cust_left = net.predict_proba(features)[:1]
        
        # Check wheter should delay or not
        if np.max(score_cust_left) > 0.5:
            idx_cust_left = np.argmax(score_cust_left)
            cust_left = np.where((observation['must_dispatch']==False) & (observation['customer_idx']!=0))[0]
            idx_delay = cust_left[idx_cust_left]
            
            mask[idx_delay] = False
            observation['is_postpone'] = True
    return _filter_instance(observation, mask)


STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    supervised=_supervised,
    dqn=_dqn,
    # cluster=_cluster,
    # topnode=_topnode,
    # topcluster=_topcluster,
    connectdot=_connectdot,
    modcutoff=_modcutoff,
    collectpostone=_collectpostone,
    postone=_postone,
    # postall=_postall,
)
