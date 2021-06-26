### for pwm run script
wandb login 24706ca326383e9462ee49fd7e91f4dde87aa402
### 17-2
#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 17-2 --step 0 --lr 0.01 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 17-2 --step 1 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=1 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 80 --dataset city --name PLOP --task 17-2 --step 1 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/ --test;

### 13-6
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6 --step 0 --lr 0.01 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6 --step 1 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=1 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 80 --dataset city --name PLOP --task 13-6 --step 1 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/ --test;

### 13-6s
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 0 --lr 0.01 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 1 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 2 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 3 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 4 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 5 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 12 --dataset city --name PLOP --task 13-6s --step 6 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/;
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port 29501 --nproc_per_node=1 run.py  --data_root /home/AIL-net/dataset --overlap --batch_size 80 --dataset city --name PLOP --task 13-6s --step 6 --lr 0.001 --epochs 30 --method PLOP --opt_level O1 --checkpoint checkpoints/step/ --test;

