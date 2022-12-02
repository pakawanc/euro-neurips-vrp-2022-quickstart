# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import argparse
import os
import uuid
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import functools
import random
import pickle

if __name__ == "__main__":
    # Add current working directory to path so we can import
    sys.path.insert(0, os.getcwd())

import tools
from environment import VRPEnvironment
from baselines.strategies import STRATEGIES
from solver import solve_static_vrptw, run_baseline


def run_collect_postone(args, env, oracle_solution=None, strategy=None, seed=None, epoch_postpone=1, postpone_seed=1):
    strategy = strategy or args.strategy
    strategy = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    seed = seed or args.solver_seed

    rng = np.random.default_rng(seed)

    features = None
    total_reward = 0
    done = False
    # Note: info contains additional info that can be used by your solver
    observation, static_info = env.reset()
    
    num_epoch = static_info['num_epochs']
    is_postpone_before_end_epoch = (epoch_postpone <= (num_epoch - 1))
    if not is_postpone_before_end_epoch:
        return(np.zeros((0,107)),[])

    epoch_tlim = static_info['epoch_tlim']
    num_requests_postponed = 0
    while not done:
        epoch_instance = observation['epoch_instance']

        if args.verbose:
            log(f"Epoch {static_info['start_epoch']} <= {observation['current_epoch']} <= {static_info['end_epoch']}", newline=False)
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_new_requests = num_requests_open - num_requests_postponed
            log(f" | Requests: +{num_new_requests:3d} = {num_requests_open:3d}, {epoch_instance['must_dispatch'].sum():3d}/{num_requests_open:3d} must-go...", newline=False, flush=True)

        if oracle_solution is not None:
            request_idx = set(epoch_instance['request_idx'])
            epoch_solution = [route for route in oracle_solution if len(request_idx.intersection(route)) == len(route)]
            cost = tools.validate_dynamic_epoch_solution(epoch_instance, epoch_solution)
        else:
            # Select the requests to dispatch using the strategy
            # Note: DQN strategy requires more than just epoch instance, bit hacky for compatibility with other strategies
            epoch_instance_dispatch = strategy({**epoch_instance, 'observation': observation, 'static_info': static_info}, rng)

            # Run HGS with time limit and get last solution (= best solution found)
            # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
            # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
            solutions = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim, tmp_dir=args.tmp_dir, seed=args.solver_seed))
            assert len(solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
            epoch_solution, cost = solutions[-1]

            # Map HGS solution to indices of corresponding requests
            epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]

        if args.verbose:
            num_requests_dispatched = sum([len(route) for route in epoch_solution])
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_requests_postponed = num_requests_open - num_requests_dispatched
            log(f" {num_requests_dispatched:3d}/{num_requests_open:3d} dispatched and {num_requests_postponed:3d}/{num_requests_open:3d} postponed | Routes: {len(epoch_solution):2d} with cost {cost:6d}")

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        assert cost is None or reward == -cost, "Reward should be negative cost of solution"
        assert not info['error'], f"Environment error: {info['error']}"

        features = epoch_instance_dispatch.get('features',None) if epoch_instance_dispatch.get('features',None) is not None else features
        total_reward += reward
        total_cost = -total_reward

    if args.verbose:
        log(f"Cost of solution: {-total_reward}")

    return total_cost, features

def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default='greedy', help="Baseline strategy used to decide whether to dispatch routes")
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

    if args.tmp_dir is None:
        # Generate random tmp directory
        args.tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
        cleanup_tmp_dir = True
    else:
        # If tmp dir is manually provided, don't clean it up (for debugging)
        cleanup_tmp_dir = False

    try:
        # List ans Sample instances
        instance_seed = int(args.instance_seed)
        random.seed(instance_seed)
        list_instances = os.listdir(args.instance_path)
        list_instances = [instance for instance in list_instances if 'ORTEC' in instance]

        list_instances = random.sample(list_instances,args.num_instance)
        num_instances = len(list_instances)

        list_features, list_cost, store_instance = [],[],[]
        for i,instance in enumerate(list_instances):
            for epoch_postpone in range(1,args.num_epoch_postpone+1):
                for postpone_seed in range(1,args.postpone_seed+1):
                    if (postpone_seed%5==0):
                        log(f'Instance {i+1}/{num_instances} postpone at {epoch_postpone} at customer index {postpone_seed}')
                    
                    if args.strategy == 'collectpostone':
                        strategy = functools.partial(STRATEGIES[args.strategy], epoch_postpone=epoch_postpone, postpone_seed=postpone_seed)
                    else:
                        strategy = STRATEGIES[args.strategy]

                    instance_path = os.path.join(args.instance_path,instance)
                    env = VRPEnvironment(seed=instance_seed, instance=tools.read_vrplib(instance_path), epoch_tlim=args.epoch_tlim, is_static=False)
                    total_cost, features = run_collect_postone(args, env, strategy=strategy,
                                                                epoch_postpone=epoch_postpone, postpone_seed=postpone_seed)

                    list_features.append(features)
                    list_cost.append(total_cost)
                    store_instance.append(instance)
                pickle.dump([list_features,list_cost,store_instance],open('misc/checkpoint.pkl','wb'))
                
        
        assert len(list_features) == len(list_cost), f'Number X {len(list_features)} != y {len(list_cost)}'
        # Get output
        if args.strategy == 'collectpostone':
            columns = ['avg_dur_with_must_dispatch', 'min_dur_with_must_dispatch',
                        *[f'rank_duration_{i}' for i in range(1,101)],
                        'demand_left', 'avg_demand', 'total_demand_must_dispatch', 'capacity', 'rem_epoch']
            df = pd.DataFrame.from_records(list_features,columns=columns)
        else:
            df = pd.DataFrame()
        df['cost'] = list_cost
        df['instance_seed'] = np.repeat(instance_seed,num_instances*args.num_epoch_postpone*args.postpone_seed)
        df['instance'] = np.repeat(list_instances,args.num_epoch_postpone*args.postpone_seed)
        df['postpone_epoch'] = np.tile(np.repeat(np.arange(1,args.num_epoch_postpone+1),args.postpone_seed),num_instances)
        df['postpone_seed'] = np.tile(np.arange(1,args.postpone_seed+1),num_instances*args.num_epoch_postpone)
        df.to_csv(f'misc/data_postone_{args.strategy}_{args.instance_seed}.csv',index=False)

    finally:
        if cleanup_tmp_dir:
            tools.cleanup_tmp_dir(args.tmp_dir)