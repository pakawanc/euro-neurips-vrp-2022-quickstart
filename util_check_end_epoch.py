# Controller has an environment and tests it against a dynamic solver program
import os
import pandas as pd
import tools

if __name__ == "__main__":

    MAX_REQUESTS_PER_EPOCH = 100  # Every epoch, we will sample at most 100 new requests
    MARGIN_DISPATCH = 3600  # Assume it takes one hour to dispatch the vehicle
    EPOCH_DURATION = 3600  # We will dispatch vehicles once an hour

    list_instances = os.listdir('./instances')
    list_instances = [instance for instance in list_instances if 'ORTEC' in instance]

    list_start, list_end, list_num = [],[],[]
    for instance in list_instances:
        if instance.startswith('ORTEC'):
            instance_path = os.path.join('./instances',instance)
            # Load instance
            static_instance = tools.read_vrplib(instance_path)

            timewi = static_instance['time_windows']
            start_epoch = int(max((timewi[1:, 0].min() - MARGIN_DISPATCH) // EPOCH_DURATION, 0))
            end_epoch = int(max((timewi[1:, 0].max() - MARGIN_DISPATCH) // EPOCH_DURATION, 0))
            num_epoch = end_epoch-start_epoch+1

            list_start.append(start_epoch)
            list_end.append(end_epoch)
            list_num.append(num_epoch)

    df = pd.DataFrame({'instance':list_instances,'num':list_num,
                        'start':list_start,'end':list_end})
    df.to_csv('misc/num_epoch.csv',index=False)