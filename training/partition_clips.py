import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import torchaudio
from tqdm import tqdm

min_length_sec = 10
max_segments_per_clip = 10

parser = argparse.ArgumentParser(description="Process audio clips.")
parser.add_argument(
    "--data_dir",
    type=Path,
    help="Path to the directory containing audio files",
    default="./training/example_audios",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    help="Path to the output tsv file",
    default="./training/example_output/clips.tsv",
)
parser.add_argument(
    "--start", type=int, help="Start index for processing files", default=0
)
parser.add_argument(
    "--end", type=int, help="End index for processing files", default=-1
)
parser.add_argument(
    "--num_workers", type=int, help="Number of parallel workers", default=cpu_count()
)
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
start = args.start
end = args.end
num_workers = args.num_workers


def process_audio_file(audio_file_path):
    """Process a single audio file and return segment data."""
    audio_name = audio_file_path.stem  # file name without extension
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        return None, "load_error"

    # waveform: (1/2) * length
    if waveform.shape[1] < 1 / 2 * sample_rate * min_length_sec:
        return None, "too_short"

    # try to partition the audio into segments, each with length of min_length_sec
    segment_length = int(sample_rate * min_length_sec)
    total_length = waveform.shape[1]
    num_segments = min(
        max_segments_per_clip, max(total_length // segment_length, 1)
    )  # at least select one segment
    if num_segments > 1:
        segment_interval = (total_length - segment_length) // (num_segments - 1)
    else:
        segment_interval = 0

    segments = []
    for i in range(num_segments):
        start_sample = i * segment_interval
        end_sample = start_sample + segment_length  # num of points before resampling
        audio_id = f"{audio_name}_{i}"
        segments.append((audio_id, audio_name, start_sample, end_sample))

    return segments, "success"


output_data = []

blacklisted = 0
if end == -1:
    end = len(os.listdir(data_dir))
audio_files = sorted(os.listdir(data_dir))[start:end]
print(
    f"Processing {len(audio_files)} files from {start} to {end} with {num_workers} workers"
)

jump = 0

# Prepare file paths
audio_file_paths = [data_dir / audio_file for audio_file in audio_files]

# Process files in parallel
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Submit all tasks
    futures = {
        executor.submit(process_audio_file, path): path for path in audio_file_paths
    }

    # Process results as they complete with progress bar
    for future in tqdm(as_completed(futures), total=len(audio_file_paths)):
        segments, status = future.result()
        if status == "success" and segments:
            output_data.extend(segments)
        else:
            jump += 1

output_dir.parent.mkdir(parents=True, exist_ok=True)
print(len(output_data))
output_df = pd.DataFrame(
    output_data, columns=["id", "name", "start_sample", "end_sample"]
)
output_df.to_csv(output_dir, index=False, sep="\t")

print(f" Jumping {jump} audio files .. ")
