## split audio clips
PATH_TO_AUDIO_DIR=     # dir to audio clips e.g.: ./data/jamendo_meanaudio_ready/train/audios
OUTPUT_PARTITION_FILE=     # ouput csv path, e.g.: ./data/jamendo_meanaudio_ready/train/partitions.tsv

python training/partition_clips.py \
    --data_dir $PATH_TO_AUDIO_DIR \
    --output_dir $OUTPUT_PARTITION_FILE


## extract audio latents
export CUDA_VISIBLE_DEVICES=0

CAPTIONS_TSV=    # captions tsv path, e.g.: ./data/jamendo_meanaudio_ready/train/jamendo_train.tsv
OUTPUT_LATENT_DIR=     # output latent dir, e.g.: ./data/jamendo_meanaudio_ready/train/latents
OUTPUT_NPZ_DIR=     # output npz dir, e.g.: ./data/jamendo_meanaudio_ready/train/npz

torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \
    --captions_tsv $CAPTIONS_TSV \
    --data_dir $PATH_TO_AUDIO_DIR \
    --clips_tsv $OUTPUT_PARTITION_FILE \
    --latent_dir $OUTPUT_LATENT_DIR \
    --output_dir $OUTPUT_NPZ_DIR \
    --text_encoder='t5_clap'  # ['clip', 't5', 't5_clap']