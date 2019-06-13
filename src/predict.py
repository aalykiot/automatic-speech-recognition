#### Global imports ####
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np

#### Custom utils imports ####
from utils import silence, words, model

json_data = {}

with open("model.json") as json_file:
    json_data = json.load(json_file)

labels = []
vectors = []

for object in json_data["model"]:
    labels.append(object["label"])
    vectors.append(object["vector"])

### Building KNN model ####

knn_model = model.create(vectors, labels, n_neighbors=1)

print("==> Model loaded successfully")

#### Reading audio file ####

signal, sample_rate = librosa.load("./samples/tests/test-1.m4a")
signal = signal[8000:]

#### Setting silence threshold as a const ####

silence_threshold = 0.008

#### Removing silence from signal ####

print("==> Proccessing audio file...")
processed_signal = silence.remove(signal, silence_threshold)

#### Splitting signal into words ####

w = words.split(processed_signal)

#### Generate mfccs for each word signal ####

test_vectors = []
for word in w:
    v = model.create_vector(word, sample_rate)
    test_vectors.append(v)

outputs = []

for vec in test_vectors:
    predicted = knn_model.predict([vec])
    outputs.append(predicted)


output = ", ".join(np.array(outputs).flatten().tolist())
print("==> Prediction output: " + output)
