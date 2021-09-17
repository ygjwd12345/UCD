### for pwm run script
wandb login 24706ca326383e9462ee49fd7e91f4dde87aa402
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 19-1 --step 0 --lr 0.01 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 19-1 --step 1 --lr 0.001 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5 --step 0 --lr 0.01 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5 --step 1 --lr 0.001 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 0 --lr 0.01 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 1 --lr 0.001 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 2 --lr 0.001 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 3 --lr 0.001 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 4 --lr 0.001 --method att;
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 run.py --data_root ./dataset --batch_size 6 --dataset voc --name att --task 15-5s --step 5 --lr 0.001 --method att;

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10 --step 0 --lr 0.01 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10 --step 1 --lr 0.001 --method att;

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 0 --lr 0.01 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 1 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 2 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 3 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 4 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 5 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 6 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 7 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 8 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 9 --lr 0.001 --method att;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root ./dataset --batch_size 12 --dataset voc --name att --task 10-10s --step 10 --lr 0.001 --method att;


