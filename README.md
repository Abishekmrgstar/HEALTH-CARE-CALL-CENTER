# Audio Transcription and Analysis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13Wj_wPqIQQvfde6A8AAhzPBvdMCZjjhG)

This project aims to transcribe audio files, identify speakers, and analyze sentiment while ensuring HIPAA compliance.

## Overview

The workflow includes:

1. Converting audio to mono.
2. Transcribing using Whisper.
3. Extracting speaker embeddings.
4. Clustering to identify speakers.
5. Performing sentiment analysis on the conversation.
6. Recognizing specific entities in the conversation for HIPAA compliance.

## Requirements

- Python 3.7+
- Whisper
- Pyannote Audio
- Scikit-learn
- SpaCy

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install the required packages:
    ```bash
    pip install git+https://github.com/openai/whisper.git
    pip install git+https://github.com/pyannote/pyannote-audio
    pip install scikit-learn transformers spacy
    python -m spacy download en_core_web_sm
    ```

## Usage

1. Upload your audio file:
    ```python
    from google.colab import files
    uploaded = files.upload()
    path = next(iter(uploaded))
    ```

2. Convert the audio to mono:
    ```python
    import subprocess
    if path[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', path, '-ac', '1', 'audio_mono.wav', '-y'])
        path = 'audio_mono.wav'
    else:
        subprocess.call(['ffmpeg', '-i', path, '-ac', '1', 'audio_mono.wav', '-y'])
        path = 'audio_mono.wav'
    ```

3. Transcribe and identify speakers:
    ```python
    import whisper
    from pyannote.audio import Audio
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.core import Segment
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import datetime
    import wave
    import contextlib
    import torch

    model = whisper.load_model('large')
    result = model.transcribe(path)
    segments = result["segments"]

    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda"))

    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])

    embeddings = np.stack([segment_embedding(segment).detach().numpy() for segment in segments])
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    with open("transcript.txt", "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"][1:] + ' ')

    print(open("transcript.txt", "r").read())
    ```

4. Perform sentiment analysis and check for HIPAA compliance:
    ```python
    import spacy

    nlp = spacy.load("en_core_web_sm")

    with open("transcript.txt", "r") as file:
        transcript = file.read()

    doc = nlp(transcript)
    phi_entities = ["PERSON", "GPE", "DATE", "ORG", "LOC", "NORP", "FAC", "EVENT"]
    detected_entities = {entity: False for entity in phi_entities}

    for entity in doc.ents:
        if entity.label_ in detected_entities:
            detected_entities[entity.label_] = True
            print(f"Detected: {entity.text} ({entity.label_})")

    all_detected = all(detected_entities.values())

    if all_detected:
        print("All required entities are present in the transcript.")
    else:
        print("NOT ASKED: The call does not contain all required entity types.")
        for entity, detected in detected_entities.items():
            if not detected:
                print(f"Missing entity type: {entity}")
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Abishekmrgstar/HEALTH-CARE-CALL-CENTER/blob/main/LICENSE) file for details.
