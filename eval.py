import os
import multiprocessing as mp
import subprocess
import argparse
from itertools import repeat
import numpy as np
import pandas as pd
import random
import time

def eval_instance(instances,args):
    cmd_controller = f'''
        python controller.py --instance {args.instance_path}/{instances} --epoch_tlim 5
            -- python solver.py --strategy {args.strategy}
                --model_path {args.model_path} --instance_seed {args.instance_seed}
                        '''
    p = subprocess.run(cmd_controller.split(), capture_output=True, text=True)
    
    # Extract cost from output
    cost = [int(line.split(': ')[1]) for line in p.stdout.split('\n') if 'Cost of solution' in line]

    with cnt.get_lock():
        cnt.value += 1
        if len(cost) > 0:
            print(f'Finished instances {cnt.value} : {instances} -- Cost : {cost[0]}')
            return(cost[0])
        else:
            print(f'Finished instances {cnt.value} : {instances} -- Cost : {cost}')
            return(cost)

def init_globals(counter):
    global cnt
    cnt = counter

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default='greedy', help="Baseline strategy used to decide whether to dispatch routes")
    parser.add_argument("--model_path", type=str, default=None, help="Provide the path of the machine learning model to be used as strategy (Path must not contain `model.pth`)")
    parser.add_argument("--instance_path", type=str, default='instances', help="Provide the path of instances")
    parser.add_argument("--instance_seed", type=int, default=0, help="Seed used for sampling instances")
    parser.add_argument("--num_instance", type=int, default=250, help="Number of instances that would be evaluted")
    parser.add_argument("--processor", type=int, default=1, help="Numver of processers would be used")
    args = parser.parse_args()

    # List ans Sample instances
    random.seed(args.instance_seed)
    list_instances = os.listdir(args.instance_path)
    list_instances = [instance for instance in list_instances if 'ORTEC' in instance]

    list_instances = random.sample(list_instances,args.num_instance)
    num_instances = len(list_instances)
    cnt = mp.Value('i', 0)

    # Evaluate cost
    num_processors = min(mp.cpu_count(),int(args.processor))
    pool = mp.Pool(processes=num_processors,initializer=init_globals,initargs=(cnt,))
    arr_cost = pool.starmap(eval_instance,
                            zip(list_instances, repeat(args)))
    
    res = pd.DataFrame({'instance':list_instances, 'cost':arr_cost})
    res.to_csv(f'misc/cost_{args.strategy}.csv',index=False)

    # Check whether iterate over all files
    assert len(arr_cost)==num_instances, 'Not complete checking over all files'
    print('='*60)
    print(f'Avg Cost of solution : {int(np.mean(arr_cost))}')
    print(f'Med Cost of solution : {int(np.median(arr_cost))}')
    print(f'----- Time : {int(time.time()-start_time)} seconds -----')
    print('='*60)
