import librosa
import keyboard
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

SR = 22050
DURATION = 1.0
TARGET_SAMPLES = int(DURATION * SR)

BASE_DIR = Path("../../data/audio")
PROCESSED_DIR = BASE_DIR / "processed"
RAW_DIR = BASE_DIR / "raw"
MIXED_DIR = BASE_DIR / "mixed"


def process_audio(output_subdir: str):
    """
    Cut audio from directory to one-second fragments
    
    :param input_dir: Path to directory with initial audio (drone or noise)
    :type input_dir: Path
    :param output_subdir: subdirrectory 
    :type output_subdir: str
    """
    
    input_dir = RAW_DIR / output_subdir
    print(f"Input directory: {input_dir}")
    output_dir = PROCESSED_DIR / output_subdir
    print(f"Output directory: {output_dir}")

    audio_files = []

    for item in input_dir.iterdir():
        if item.is_file():
            audio_files.append(item)
        
    if not audio_files:
        print(f"Input directory is empty!")
        return
    
    print(f"{len(audio_files)} audio files were found to process.")

    total_segments = 0
    skipped_files = 0
    
    for file_idx, file_path in enumerate(tqdm(audio_files, desc="Processig files"), 1):
        try:
            audio, _ = librosa.load(str(file_path), sr=SR)
            
            total_samples = len(audio)
            num_segments = total_samples // TARGET_SAMPLES
            
            if num_segments == 0:
                print(f"⚠️  [{file_idx}/{len(audio_files)}] {file_path.name} "
                      f"is too short ({total_samples/SR:.2f} sec) — skipped")
                skipped_files += 1
                continue
            
            base_name = f"{output_subdir}__{file_path.stem}"
            
            for seg_idx in range(num_segments):
                start = seg_idx * TARGET_SAMPLES
                end = start + TARGET_SAMPLES
                segment = audio[start:end]
                
                seg_filename = f"{base_name}_{seg_idx:03d}.wav"
                seg_path = output_dir / seg_filename
                
                sf.write(str(seg_path), segment, SR, subtype='PCM_16')
            
            total_segments += num_segments
            
        except Exception as e:
            print(f"Error while processing {file_path.name}: {str(e)}")
            continue

    print("\n" + "="*50)
    print("Processed!")
    print(f"   Input directory: {input_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Processed {len(audio_files) - skipped_files} files")
    print(f"   Skipped {skipped_files} short files")
    print(f"   Created {total_segments} {DURATION}-sec segments")
    print("="*50)


def main():
    print('\n' + '='*50)
    print("Choose type of audio files:")
    print("  Press 'd' - drone audio")
    print("  Press 'b' - background audio")
    print("  Press 'm' - mixed audio")
    print("  Press 'q' - quit program")
    print("="*50)

    choice = None
    while choice is None:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'd':
                print("Selected DRONE audio.")
                choice = "drone"
            elif event.name == 'b':
                print("Selected BACKGROUND audio.")
                choice = "background"
            elif event.name == 'm':
                print("Selected MIXED audio.")
                choice = "mixed"
            elif event.name == 'q':
                print("\nExiting program...")
                exit(0)
    
    process_audio(choice)

if __name__ == "__main__":
    main()