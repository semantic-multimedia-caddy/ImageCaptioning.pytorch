python distributed_run.py --cfg configs/fc.yml \
                          --id fc \
                          --world_size 2 \
                          --rank 0 \
                          --master 127.0.0.1 \
                          --port 25555 \
                          --batch_size 128

# mpirun -np 2 \
#     -H 3.37.4.242:1,3.37.163.245:1 \
#     -bind-to none -map-by slot \
#     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#     -mca pml ob1 -mca btl ^openib \
#     -mca plm_rsh_args "-i ~/.ssh/bdml.pem" \
#     python train_distributed.py --cfg configs/fc.yml
