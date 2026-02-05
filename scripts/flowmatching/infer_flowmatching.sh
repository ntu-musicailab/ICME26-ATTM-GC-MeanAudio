export CUDA_VISIBLE_DEVICES=1

output_path=./exps/AC_128_numgpus1_niter200_000_T5_CLAP_fluxaudio_s/outputs

prompt="A beautiful and moving classical piano piece showcasing intricate melodies and harmonies, with elements of dissonance and tension creating a sense of release."
ckpt_path=./AC_128_numgpus1_niter200_000_T5_CLAP_fluxaudio_s/AC_128_numgpus1_niter200_000_T5_CLAP_fluxaudio_s_ema_final.pth

python infer.py \
    --variant "fluxaudio_s" \
    --prompt "$prompt" \
    --model_path "$ckpt_path" \
    --output $output_path \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --duration 10 \
    --use_rope \
    --text_c_dim 512
