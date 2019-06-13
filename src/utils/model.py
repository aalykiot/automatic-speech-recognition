import numpy as np
import librosa

#### Scikit-learn imports ####
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def create_vector(signal, sample_rate):

    #### Getting the mean value on time-axes for each feature ####

    mean_mffcs = [np.mean(feature) for feature in librosa.feature.mfcc(
        np.array(signal), sr=sample_rate, n_mfcc=12)]

    #### Scaling vectors with mean=0 ####

    mean_mffcs = np.array(mean_mffcs).reshape(-1, 1)

    scaler = preprocessing.StandardScaler()
    scaler.fit(mean_mffcs)

    scaled_vectors = scaler.transform(mean_mffcs)

    return scaled_vectors.flatten()


def create(vectors, labels, n_neighbors):
    #### Building KNN model ####

    model = KNeighborsClassifier(n_neighbors)
    model.fit(vectors, labels)

    return model
