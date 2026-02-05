export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
btz=128
num_iterations=200_000
exp_id=AC_${btz}_numgpus${NUM_GPUS}_niter${num_iterations}_T5_CLAP_fluxaudio_s

text_encoder_name=t5_clap

text_c_dim=512
model=fluxaudio_s


OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train.py \
    --config-name train_config_jamendo.yaml \
    exp_id=$exp_id \
    compile=False \
    model=$model \
    batch_size=${btz} \
    eval_batch_size=32 \
    num_iterations=$num_iterations \
    text_encoder_name=$text_encoder_name \
    data_dim.text_c_dim=$text_c_dim \
    pin_memory=False \
    num_workers=10 \
    ac_oversample_rate=5 \
    use_meanflow=False \
    cfg_strength=4.5 \
    ++use_rope=True \
    ++use_wandb=True \
    ++debug=False