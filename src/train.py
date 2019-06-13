#### Global imports ####
import json
import librosa
import numpy as np

#### Custom utils imports ####
from utils import silence, words, model

json_data = {}
with open("dataset.json") as json_file:
    json_data = json.load(json_file)

#### Setting silence threshold as a const ####

silence_threshold = 0.008

# Initializing labels & vectors lists

labels = []
vectors = []

for i, object in enumerate(json_data["dataset"]):

    print("==> Proccessing signal " + str(i + 1) + " of " +
          str(len(json_data["dataset"])) + "...")

    # Appending audio signal's label to labels list
    labels.append(object["label"])

    #### Reading audio file ####

    signal, sample_rate = librosa.load(
        "./samples/training/" + object["filename"])

    #### Removing microphone's noise when it opens ####

    signal = signal[5000:]

    #### Removing silence from signal ####

    processed_signal = silence.remove(signal, silence_threshold)

    #### Splitting signal into words ####

    w = words.split(processed_signal)

    if len(w) == 0:
        continue

    #### Generate mfcc features for the word ####

    mfcc_vector = model.create_vector(w[0], sample_rate)

    vectors.append(mfcc_vector)

data = {}
data["model"] = []

for i, val in enumerate(labels):
    data["model"].append({
        "label": val,
        "vector": vectors[i].tolist()
    })

with open("model.json", "w") as json_file:
    json.dump(data, json_file)

print("==> ML model created successfully")
