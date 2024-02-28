WORKSPACE='/home/yangzheng/layout-diffusion-mindspore'
cd ${WORKSPACE}

conda activate LayoutDiffusion

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
mpirun python scripts/run_train.py \
  --config_file configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml \
  -n 8
