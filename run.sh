#!/bin/bash
#SBATCH  --gres=gpu:1 --constraint=[v100|p100] -p compsci-gpu --mem 65536

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

for phi in $(seq 90.0 -7.5 -90.0)
    do 
    for theta in {-180..180..15}
        do  
            echo "/usr/xtmp/ct214/daml/vr_sickness/perspectives_skyhouse/left_eye/theta_${theta}_phi_${phi}/"
            srun -u python main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder \
                        --inference_dataset_root "/usr/xtmp/ct214/daml/vr_sickness/perspectives_skyhouse/left_eye/theta_${theta}_phi_${phi}/" \
                        --resume FlowNet2_checkpoint.pth.tar --save "/usr/xtmp/ct214/daml/vr_sickness/test" --name "left_eye_theta_${theta}_phi_${phi}" --save_dir '/usr/xtmp/ct214/daml/vr_sickness/perspective_skyhouse_of_results/'
            srun -u python main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder \
                        --inference_dataset_root "/usr/xtmp/ct214/daml/vr_sickness/perspectives_skyhouse/right_eye/theta_${theta}_phi_${phi}/" \
                        --resume FlowNet2_checkpoint.pth.tar --save "/usr/xtmp/ct214/daml/vr_sickness/test" --name "right_eye_theta_${theta}_phi_${phi}" --save_dir '/usr/xtmp/ct214/daml/vr_sickness/perspective_skyhouse_of_results/'
        done
    done


# rm -r skyhouse_perspective_png_output

# python -m flowiz /usr/xtmp/ct214/daml/vr_sickness/flownet2-pytorch/skyhouse_perpective_output/inference/run.epoch-0-flow-field/*.flo --outdir skyhouse_perspective_png_output

# srun -u python script.py
