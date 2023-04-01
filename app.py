from potassium import Potassium, Request, Response

import os
import time
import json
import wave
import torch
import base64
import whisper
import datetime
import contextlib
import numpy as np
import pandas as pd
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

app = Potassium("whisper_diarization")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "large-v2"
    model = whisper.load_model(model_name, device=device)
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", device=device
    )
    context = {"model": model}
    context = {"embedding_model": embedding_model}

    return context


# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    filename = request.json.get("filename")
    file = request.json.get("file")
    num_speakers = request.json.get("num_speakers")
    model = context.get("model")
    embedding_model = context.get("embedding_model")

    if file == None or file == "" or file == "":
        return Response(json={"message": "No correct input provided"}, status=400)

    # TODO: check if file is right format, can also be done in frontend
    base64file = file.split(",")[1] if "," in file else file

    file_data = base64.b64decode(base64file)
    file_start, file_ending = os.path.splitext(f"{filename}")

    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{ts}-{file_start}{file_ending}"
    with open(filename, "wb") as f:
        f.write(file_data)

    def convert_time(secs):
        return datetime.timedelta(seconds=round(secs))

    def speech_to_text(filepath, whisper_model, num_speakers, prompt):
        model = whisper.load_model(whisper_model)
        time_start = time.time()

        try:
            _, file_ending = os.path.splitext(f"{filepath}")
            print(f"file enging is {file_ending}")
            audio_file_wav = filepath.replace(file_ending, ".wav")
            print("-----starting conversion to wav-----")
            os.system(
                f'ffmpeg -i "{filepath}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file_wav}"'
            )
        except Exception as e:
            raise RuntimeError("Error converting audio")

        # Get duration
        with contextlib.closing(wave.open(audio_file_wav, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        print("starting whisper")
        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(
            audio_file_wav, **transcribe_options, initial_prompt=prompt
        )
        segments = result["segments"]
        print("done with whisper")

        try:
            # Create embedding
            def segment_embedding(segment):
                audio = Audio()
                start = segment["start"]
                # Whisper overshoots the end timestamp in the last segment
                end = min(duration, segment["end"])
                clip = Segment(start, end)
                waveform, sample_rate = audio.crop(audio_file_wav, clip)
                return embedding_model(waveform[None])

            print("starting embedding")
            embeddings = np.zeros(shape=(len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = segment_embedding(segment)
            embeddings = np.nan_to_num(embeddings)
            print(f"Embedding shape: {embeddings.shape}")

            # Assign speaker label
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

            # Make output
            output = []  # Initialize an empty list for the output
            for segment in segments:
                # Append the segment to the output list
                output.append(
                    {
                        "start": str(convert_time(segment["start"])),
                        "end": str(convert_time(segment["end"])),
                        "speaker": segment["speaker"],
                        "text": segment["text"],
                    }
                )

            print("done with embedding")
            time_end = time.time()
            time_diff = time_end - time_start

            system_info = f"""-----Processing time: {time_diff:.5} seconds-----"""
            print(system_info)
            os.remove(audio_file_wav)
            return output

        except Exception as e:
            os.remove(audio_file_wav)
            raise RuntimeError("Error Running inference with local model", e)

    transcription = speech_to_text(filename, model, num_speakers, prompt)
    os.remove(filename)
    print(f'{filename} removed, done with inference')
    return Response(json={"segments": transcription}, status=200)


if __name__ == "__main__":
    app.serve()
