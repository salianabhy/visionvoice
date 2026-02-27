# tts_generator.py
# Handles converting a text description into an MP3 audio file
# using Google Text-to-Speech (gTTS).

from gtts import gTTS
import os
import uuid


def generate_audio(text: str, output_dir: str) -> str:
    """
    Convert a text string to spoken audio and save as an MP3 file.

    Args:
        text:       The description text to convert to speech
        output_dir: Directory path where the MP3 file should be saved

    Returns:
        The filename (not full path) of the saved MP3 file
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename to avoid overwriting previous audio files
    # This also allows multiple users to use the app without conflicts
    filename = f"description_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(output_dir, filename)

    # Create a gTTS object
    # lang='en' — English language
    # slow=False — normal speaking speed (True makes it slower, good for accessibility)
    tts = gTTS(text=text, lang="en", slow=False)

    # Save the audio to file
    tts.save(filepath)

    print(f"Audio saved: {filepath}")
    return filename


def cleanup_old_audio(output_dir: str, keep_latest: int = 10):
    """
    Remove old audio files, keeping only the most recent ones.
    Prevents the static/audio folder from filling up during a demo session.

    Args:
        output_dir:  Directory containing audio files
        keep_latest: Number of recent files to keep (default: 10)
    """
    try:
        files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".mp3")
        ]
        # Sort by modification time (oldest first)
        files.sort(key=os.path.getmtime)

        # Delete files beyond the keep limit
        files_to_delete = files[:-keep_latest] if len(files) > keep_latest else []
        for f in files_to_delete:
            os.remove(f)
            print(f"Cleaned up old audio file: {f}")

    except Exception as e:
        print(f"Cleanup warning (non-critical): {e}")
