import sounddevice as sd
import soundfile as sf
import keyboard
from pathlib import Path

# Settings
DURATION_TOTAL = 60.0     # total duration of record (sec)
SAMPLE_RATE = 22050       # Gz
CHANNELS = 1              # Mono

# Path to save audio
BASE_DIR = Path("../../data/audio/raw")

def record_audio(duration, sr=SAMPLE_RATE, channels=CHANNELS):

    """Records audio from microphone"""
    
    print("Recording...")
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=channels,
        dtype='float32'
    )
    sd.wait()
    print("Record finished.")
    
    if channels == 1:
        audio = audio.flatten()
    return audio


if __name__ == "__main__":
    try:
        while True:
            print("\n" + "="*50)
            print("Select type of record:")
            print("  Press 'd' - drone record")
            print("  Press 'b' - background record")
            print("  Press 'm' - mixed record")
            print("  Press 'q' - quit program")
            print("="*50)
            
            choice = None
            while choice is None:
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    if event.name == 'd':
                        choice = "drone"
                    elif event.name == 'b':
                        choice = "background"
                    elif event.name == 'm':
                        choice = "mixed"
                    elif event.name == 'q':
                        print("\nExiting program...")
                        exit(0)

            print(f"\nRecording {choice} audio")
            
            print("\nPress Enter to start recording... (or 'q' to exit program)")
            
            while True:
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    if event.name == 'enter':
                        
                        audio = record_audio(DURATION_TOTAL, sr=SAMPLE_RATE)

                        output_dir = BASE_DIR / choice

                        audio_path = output_dir / f"audio_record_{sum(1 for i in output_dir.iterdir())}.wav"
          
                        sf.write(audio_path, audio, SAMPLE_RATE)
                        print(f"\nRecord saved to: {audio_path}")
                        break
                    
                    elif event.name == 'q':
                        print("\nExiting program...")
                        exit(0)
            
            print("\n" + "-"*50)
            print("What would you like to do next?")
            print("  Press 'c' - continue recording (go to directory selection)")
            print("  Press 'q' - quit program")
            
            while True:
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    if event.name == 'c':
                        break 
                    elif event.name == 'q':
                        print("\nExiting program...")
                        exit(0)
        
    except KeyboardInterrupt:
        print("\nRecord was canceled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
