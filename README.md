# Speech Emotion Recognition using Deep Learning 

## Overview

This project builds a deep learning model to recognize human emotions from speech signals using audio feature extraction and LSTM networks. It uses MFCC features and achieves high accuracy on the TESS dataset.

---

## Problem Statement

Understanding human emotions from speech is crucial for applications like:

* Voice assistants
* Mental health monitoring
* Customer sentiment analysis

This project aims to classify emotions directly from audio signals.

---

## Dataset

* **Dataset**: Toronto Emotional Speech Set (TESS)
* **Total Samples**: ~5600 audio files
* **Classes**:

  * Happy
  * Sad
  * Angry
  * Fear
  * Disgust
  * Neutral
  * Pleasant Surprise

---

## Approach

### Audio Processing

* Loaded audio using Librosa
* Trimmed audio to fixed duration
* Extracted **MFCC (Mel-Frequency Cepstral Coefficients)**

### Feature Engineering

* 40 MFCC features extracted per audio
* Converted into structured input for deep learning

### Model Architecture

* LSTM (256 units)
* Dense layers (128 → 64)
* Dropout (regularization)
* Softmax output (7 classes)

---

## Model Architecture

```id="lstmarch"
Input (MFCC 40x1)
   ↓
LSTM (256)
   ↓
Dense (128) + Dropout
   ↓
Dense (64) + Dropout
   ↓
Softmax (7 emotions)
```

---

## Results

* Training Accuracy: **~99%**
* Validation Accuracy: **~98%**

The model generalizes well with minimal overfitting.

---

##  Sample Prediction

```id="predex"
Input: Audio file  
Output: Predicted Emotion → "happy"
```

---

## Tech Stack

* Python
* Librosa
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn


---


##  Real-World Applications

* Voice-based emotion detection
* Call center analytics
* AI therapy assistants
* Human-computer interaction

---

## Future Improvements

* Build Streamlit web app for real-time prediction
* Deploy model (HuggingFace / AWS)
* Add real-time microphone input
* Improve robustness with noisy audio
