"""
Standalone inference for the TESS speech emotion recognition model.

Usage:
    python inference.py path/to/audio.wav
    python inference.py path/to/audio.wav --model models/emotion_lstm.keras
"""
import argparse
import os
import sys

import numpy as np
import librosa
import tensorflow as tf

# These must match the values used during training.
SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def extract_mfcc(audio_path: str) -> np.ndarray:
    """Load a WAV, pad/trim to a fixed window, return MFCC sequence (n_frames, n_mfcc)."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.T.astype(np.float32)


def predict(audio_path: str, model_path: str, preprocess_path: str):
    """Return (predicted_class, confidence, probability_dict)."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(preprocess_path):
        raise FileNotFoundError(f"Preprocess file not found: {preprocess_path}")

    model = tf.keras.models.load_model(model_path)
    pp = np.load(preprocess_path, allow_pickle=True)
    mean, std, classes = pp["mean"], pp["std"], pp["classes"]

    feats = extract_mfcc(audio_path)
    feats = (feats - mean) / std
    feats = np.expand_dims(feats, axis=0)

    probs = model.predict(feats, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), dict(zip(classes, probs.tolist()))


def main():
    parser = argparse.ArgumentParser(
        description="Predict emotion from a speech WAV file."
    )
    parser.add_argument("audio_path", help="Path to a WAV file")
    parser.add_argument(
        "--model",
        default="models/emotion_lstm.keras",
        help="Path to the trained Keras model (default: models/emotion_lstm.keras)",
    )
    parser.add_argument(
        "--preprocess",
        default="models/preprocess.npz",
        help="Path to preprocessing stats (default: models/preprocess.npz)",
    )
    args = parser.parse_args()

    try:
        pred, conf, all_probs = predict(args.audio_path, args.model, args.preprocess)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"File:      {args.audio_path}")
    print(f"Predicted: {pred}  (confidence {conf:.2%})")
    print()
    print("All class probabilities:")
    for cls, p in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:12s} {p:.3f}  {bar}")


if __name__ == "__main__":
    main()
