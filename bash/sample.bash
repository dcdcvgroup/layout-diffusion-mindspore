WORKSPACE='/home/yangzheng/layout-diffusion-mindspore'
cd ${WORKSPACE}

WORKSPACE='/data0/yangzheng/samples/'

VERSION='LayoutDiffusion_large'
MODEL_ITERATION='1150000'

DATASET='COCO-stuff' # ['COCO-stuff', 'VG']
IMAGE_SIZE=256

SAMPLE_METHOD='dpm_solver' # ['dpm_solver', 'ddpm', 'ddim']
STEPS=25
SAMPLE_TIMES=5

CLASSIFIER_SCALE=1.0

# sample 3097 x 5
NUM_IMG=3097 # 2048, 3097, 5096
CUDA_VISIBLE_DEVICES=0 \
mpirun python scripts/run_sample.py \
      --config_file ./configs/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}.yaml \
      sample.pretrained_model_path=./pretrained_models/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}_${VERSION}_ema_${MODEL_ITERATION}.ckpt \
      sample.log_root=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION} \
      sample.sample_method=${SAMPLE_METHOD} sample.timestep_respacing=[${STEPS}] sample.classifier_free_scale=${CLASSIFIER_SCALE} sample.sample_times=${SAMPLE_TIMES} \
      sample.sample_suffix=model${MODEL_ITERATION}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD} \
      sample.save_cropped_images=False sample.fix_seed=False \
      data.parameters.test.max_num_samples=${NUM_IMG}



# sample 2048 x 1, 2048 image ids can be found in layout_diffusion/dataset/image_ids_layout2i_2048.txt, should copy it into annotations/deprecated-challenge2017
CUDA_VISIBLE_DEVICES=0 \
mpirun python scripts/run_sample.py \
      --config_file ./configs/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}.yaml \
      sample.pretrained_model_path=./pretrained_models/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}_${VERSION}_ema_${MODEL_ITERATION}.ckpt \
      sample.log_root=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION} \
      sample.sample_method=${SAMPLE_METHOD} sample.timestep_respacing=[${STEPS}] sample.classifier_free_scale=${CLASSIFIER_SCALE} sample.sample_times=${SAMPLE_TIMES} \
      sample.sample_suffix=model${MODEL_ITERATION}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD} \
      sample.save_cropped_images=False sample.fix_seed=False \
      data.parameters.test.deprecated_stuff_ids_txt='./annotations/deprecated-challenge2017/image_ids_layout2i_2048.txt' \
      data.parameters.filter_mode=SG2Im


# evaluate
pip install pytorch-fid
python -m pytorch_fid \
/data0/yangzheng/samples/VG_128x128/LayoutDiffusion_large/conditional_[25]/sample128x1/model0400000_0.5_32/generated_imgs \
/data0/yangzheng/samples/VG_128x128/LayoutDiffusion_large/conditional_[25]/sample128x1/model0400000_0.5_32/real_imgs