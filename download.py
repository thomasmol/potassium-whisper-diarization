# This file runs during container build time to get model weights built into the container

import whisper
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

def download_model():
    # tiny, small, medium, large-v1, large-v2
    model_name = "large-v2"
    whisper.load_model(model_name)
    PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

if __name__ == "__main__":
    download_model()