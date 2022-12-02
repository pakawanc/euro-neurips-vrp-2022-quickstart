gcloud compute scp --project sharp-ring-366018 --zone "asia-southeast1-b" vip:/home/mymail_forgcp/temp/my_dir/misc/checkpoint_3.pkl misc/checkpoint_3.pkl
gcloud compute scp --project sharp-ring-366018 --zone "asia-southeast1-b" vip:/home/mymail_forgcp/temp/my_dir/misc/checkpoint_4.pkl misc/checkpoint_4.pkl
gcloud compute scp --project sharp-ring-366018 --zone "asia-southeast1-b" vip:/home/mymail_forgcp/temp/my_dir/misc/checkpoint_5.pkl misc/checkpoint_5.pkl
python util_read_checkpoint.py --path misc/checkpoint_3.pkl --strategy collectpostone --instance_seed 3 --num_instance 250 --num_epoch_postpone 7 --postpone_seed 10
python util_read_checkpoint.py --path misc/checkpoint_4.pkl --strategy collectpostone --instance_seed 4 --num_instance 250 --num_epoch_postpone 7 --postpone_seed 10
python util_read_checkpoint.py --path misc/checkpoint_5.pkl --strategy collectpostone --instance_seed 5 --num_instance 250 --num_epoch_postpone 7 --postpone_seed 10
#cat misc/data_collect* > misc/temp.csv
#sed -i '' '/seed/d' misc/temp.csv
#cat misc/head.csv misc/temp.csv > misc/data_collectpostone.csv
#rm misc/data_collectpostone_* misc/temp.csv
