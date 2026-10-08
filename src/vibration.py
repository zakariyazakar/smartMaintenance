from pathlib import Path

import joblib
import numpy as np

from scipy.io import loadmat
from scipy.stats import kurtosis, skew



MODEL_PATH = (
    "models/vibration/predictive_maintenance_model.pkl"
)

saved = joblib.load(MODEL_PATH)

model = saved["model"]
features = saved["features"]
window_size = saved["window_size"]
fs = saved["fs"]


# =========================================================
# 2. EXTRACTION DES FEATURES
# =========================================================

def extract_features(signal_window):

    signal_window = np.asarray(signal_window).flatten()

    values = {
        "mean": np.mean(signal_window),
        "std": np.std(signal_window),
        "rms": np.sqrt(np.mean(signal_window ** 2)),
        "kurtosis": kurtosis(signal_window),
        "skewness": skew(signal_window),
        "peak": np.max(np.abs(signal_window))
    }

    X = np.array(
        [values[name] for name in features]
    ).reshape(1, -1)

    return X


# =========================================================
# 3. CHARGEMENT DU SIGNAL .MAT
# =========================================================

def load_signal(mat_file):

    data = loadmat(mat_file)

    keys = [
        key
        for key in data.keys()
        if "DE_time" in key
    ]

    if not keys:
        raise ValueError(
            "Aucun signal DE_time trouvé."
        )

    signal_name = keys[0]

    signal = data[signal_name].flatten()

    return signal_name, signal


# =========================================================
# 4. ANALYSE COMPLETE
# =========================================================

def inspect_vibration(mat_file):

    signal_name, signal = load_signal(mat_file)

    predictions = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        window_size
    ):

        window = signal[
            start:start + window_size
        ]

        X = extract_features(window)

        prediction = model.predict(X)[0]

        predictions.append(prediction)

    if not predictions:
        raise ValueError(
            "Signal trop court pour être analysé."
        )

    labels, counts = np.unique(
        predictions,
        return_counts=True
    )

    distribution = {}

    for label, count in zip(labels, counts):

        distribution[str(label)] = {
            "count": int(count),
            "percentage": float(
                count / len(predictions) * 100
            )
        }

    majority_index = np.argmax(counts)

    diagnosis = str(
        labels[majority_index]
    )

    majority_ratio = float(
        counts[majority_index]
        / len(predictions)
    )

    return {
        "signal_name": signal_name,
        "n_points": len(signal),
        "n_windows": len(predictions),
        "diagnosis": diagnosis,
        "majority_ratio": majority_ratio,
        "distribution": distribution,
        "predictions": predictions,
        "fs": fs,
        "window_size": window_size
    }

def inspect_window(mat_file, window_index):

    # Charger le signal
    signal_name, signal = load_signal(mat_file)

    # Nombre total de fenêtres disponibles
    n_windows = len(signal) // window_size

    if n_windows == 0:
        raise ValueError(
            "Signal trop court pour être analysé."
        )

    # Si on arrive à la dernière fenêtre,
    # on recommence depuis le début
    window_index = window_index % n_windows

    # Position de début
    start = window_index * window_size

    # Position de fin
    end = start + window_size

    # Extraire UNE fenêtre
    window = signal[start:end]

    # Calculer les features
    X = extract_features(window)

    # Prédiction
    prediction = model.predict(X)[0]

    # Résultat
    result = {
        "signal_name": signal_name,
        "window_index": window_index,
        "n_windows": n_windows,
        "prediction": str(prediction)
    }

    # Probabilités Random Forest
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

        result["probabilities"] = {
            str(class_name): float(probability)
            for class_name, probability
            in zip(
                model.classes_,
                probabilities
            )
        }

    return result