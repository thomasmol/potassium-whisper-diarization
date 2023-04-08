from potassium import Potassium, Request, Response, send_webhook

import os
import time
import requests
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
    context = {"model": model, "embedding_model": embedding_model}

    return context


# @app.async_handler runs for every call
@app.async_handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    filename = request.json.get("filename")
    file = request.json.get("file")
    file_url = request.json.get("file_url")
    file_string = request.json.get("file_string")
    num_speakers = request.json.get("num_speakers")
    chunk_index = request.json.get("chunk_index")
    chunk_count = request.json.get("chunk_count")
    offset_seconds = request.json.get("offset_seconds")
    webhook_url = request.json.get("webhook_url")
    webhook_id = request.json.get("webhook_id")

    model = context.get("model")
    embedding_model = context.get("embedding_model")

    if sum([file_string is not None, file_url is not None, file is not None]) != 1:
        return Response(
            json={
                "message": "No correct input provided, either provide file_string, file_url or a file"
            },
            status=400,
        )

    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H-%M-%S")
    file_start, file_ending = os.path.splitext(f"{filename}")
    filename = f"{ts}-{file_start}{file_ending}"

    if file_string is not None and file_url is None and file is None:
        # TODO: check if file is right format, can also be done in frontend
        base64file = file_string.split(",")[1] if "," in file_string else file_string
        file_data = base64.b64decode(base64file)
        with open(filename, "wb") as f:
            f.write(file_data)

    elif file_url is not None and file_string is None and file is None:
        response = requests.get(file_url)
        with open(filename, "wb") as file:
            file.write(response.content)

    # so i can send it to webhook and delete it
    file_url = file_url if file_url is not None else ""

    segments = speech_to_text(
        filename,
        model,
        embedding_model,
        num_speakers,
        prompt,
        chunk_index,
        chunk_count,
        offset_seconds,
    )

    if file_ending != ".wav":
        print("removing non wav file")
        os.remove(filename)

    print(f"done with inference")
    print("webhook_id:", webhook_id)
    send_webhook(
        url=webhook_url,
        json={
            "segments": segments,
            "webhook_id": webhook_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "offset_seconds": offset_seconds,
            "file_url": file_url,
        },
    )
    return


def convert_time(self, secs, offset_seconds=0):
    return datetime.timedelta(seconds=(round(secs) + offset_seconds))


def speech_to_text(
    filepath,
    model,
    embedding_model,
    num_speakers,
    prompt,
    chunk_index,
    chunk_count,
    offset_seconds=0,
):
    time_start = time.time()

    try:
        _, file_ending = os.path.splitext(f"{filepath}")
        print(f"file ending in {file_ending}")
        if file_ending != ".wav":
            audio_file_wav = filepath.replace(file_ending, ".wav")
            print("-----starting conversion to wav-----")
            os.system(
                f'ffmpeg -i "{filepath}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file_wav}"'
            )
        else:
            audio_file_wav = filepath
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

        # Convert number of speakers to int
        num_speakers = int(num_speakers)
        chunk_index = int(chunk_index)
        chunk_count = int(chunk_count)

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

        # Make output
        output = []  # Initialize an empty list for the output

        # Initialize the first group with the first segment
        current_group = {
            "start": str(round(segments[0]["start"] + offset_seconds)),
            "end": str(round(segments[0]["end"] + offset_seconds)),
            "speaker": segments[0]["speaker"],
            "text": segments[0]["text"],
        }

        for i in range(1, len(segments)):
            # Calculate time gap between consecutive segments
            time_gap = segments[i]["start"] - segments[i - 1]["end"]

            # If the current segment's speaker is the same as the previous segment's speaker,
            # and the time gap is less than or equal to 5 seconds, group them
            if segments[i]["speaker"] == segments[i - 1]["speaker"] and time_gap <= 5:
                current_group["end"] = str(round(segments[i]["end"] + offset_seconds))
                current_group["text"] += " " + segments[i]["text"]
            else:
                # Add the current_group to the output list
                output.append(current_group)

                # Start a new group with the current segment
                current_group = {
                    "start": str(round(segments[i]["start"] + offset_seconds)),
                    "end": str(round(segments[i]["end"] + offset_seconds)),
                    "speaker": segments[i]["speaker"],
                    "text": segments[i]["text"],
                }

        # Add the last group to the output list
        output.append(current_group)

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


if __name__ == "__main__":
    app.serve()
