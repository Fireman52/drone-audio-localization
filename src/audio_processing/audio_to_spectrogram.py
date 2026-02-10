import numpy as np
import librosa
import cv2
import keyboard
from tqdm import tqdm
from pathlib import Path

# Configuration
SAMPLE_RATE = 22050          # Target sample rate (Hz)
DURATION = 1.0               # Expected duration of input clips (seconds)
N_MELS = 128                 # Mel frequency bands (height of output)
N_FFT = 1024                 # FFT window size
HOP_LENGTH = 345             # Precisely yields 64 time frames for 1s @ 22050 Hz
FMAX = SAMPLE_RATE // 2      # Maximum frequency for Mel scale (Nyquist)

# Input/output paths
INPUT_DIR = Path("../../data/audio/processed")
OUTPUT_ARRAYS = Path("../../data/arrays")
OUTPUT_IMAGES = Path("../../data/images")

OUTPUT_ARRAYS.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)


def audio_to_mel_spectrogram(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert audio to Mel spectrogram with precise 128×64 dimensions.
    
    Returns:
        mel_db: Log-scaled spectrogram (128×64) for model training
        mel_norm: Normalized spectrogram (128×64) for image visualization
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )
    
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    return mel_db, mel_norm


def process_audio_file(file_path: Path, choice: str) -> bool:
    """
    Process a single 1-second audio clip into spectrogram representations.
    
    Args:
        file_path: Path to input audio file (.wav/.mp3)
        label: Class label (parent directory name)
    
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        audio, sr = librosa.load(str(file_path), sr=SAMPLE_RATE)
        
        target_samples = int(SAMPLE_RATE * DURATION)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
        
        mel_db, mel_norm = audio_to_mel_spectrogram(audio, sr)
        
        if mel_db.shape != (N_MELS, 64):
            print(f"Unexpected spectrogram shape {mel_db.shape} for {file_path.name}")
            return False
        
        array_dir = OUTPUT_ARRAYS / choice
        image_dir = OUTPUT_IMAGES / choice
        array_dir.mkdir(exist_ok=True)
        image_dir.mkdir(exist_ok=True)
        
        array_path = array_dir / f"{file_path.stem}.npy"
        np.save(array_path, mel_db.astype(np.float32))
        
        img = (mel_norm * 255).astype(np.uint8)
        img = cv2.flip(img, 0) 
        
        image_path = image_dir / f"{file_path.stem}.jpg"
        cv2.imwrite(str(image_path), img)
        
        return True
        
    except Exception as e:
        print(f"Processing failed for {file_path}: {str(e)}")
        return False


def main():
    
    print('\n' + '='*50)
    print("Choose type of audio files:")
    print("  Press 'd' - drone audio")
    print("  Press 'b' - background audio")
    print("  Press 'm' - mixed audio")
    print('='*50)

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

    audio_files = [file for file in Path(INPUT_DIR / choice).iterdir()]
    
    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(audio_files)} audio clips for processing")
    print(f"Target representation: {N_MELS}×64 Mel spectrograms")
    print(f"Sample rate: {SAMPLE_RATE} Hz | Duration: {DURATION}s per clip")
    print(f"Outputs: Arrays → {OUTPUT_ARRAYS} | Images → {OUTPUT_IMAGES}\n")
    
    successful = 0
    for file_path in tqdm(audio_files, desc="Converting audio to spectrograms"): 
        if process_audio_file(file_path, choice):
            successful += 1
    
    print("\nProcessing complete!")
    print(f"   Successfully converted: {successful}/{len(audio_files)} files")
    print(f"   Output arrays: {OUTPUT_ARRAYS / choice} (subdirectories per class)")
    print(f"   Output images: {OUTPUT_IMAGES / choice} (subdirectories per class)")

if __name__ == "__main__":
    main()