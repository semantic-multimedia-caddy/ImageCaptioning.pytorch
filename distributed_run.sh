python distributed_run.py --cfg configs/fc.yml \
                          --id fc \
                          --world_size 2 \
                          --rank 0 \
                          --master ec2-3-37-4-242.ap-northeast-2.compute.amazonaws.com \
                          --port 25555
