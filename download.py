# This file runs during container build time to get model weights built into the container

from faster_whisper import WhisperModel
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

def download_model() -> tuple:
    # tiny, small, medium, large-v1, large-v2
    model_name = "large-v2"
    model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16")
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))
    return model, embedding_model

if __name__ == "__main__":
    download_model()