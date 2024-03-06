from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Tuple
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

sample_rate = 16000
duration = 6
n_fft = 512
hop_length = 128
n_mels = 128

def predict_class(audio, model):
    # Convert audio to Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Reshape spectrogram to match the input shape expected by the model
    mel_spectrogram = mel_spectrogram.reshape(1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1)
    
    # Perform prediction
    predictions = model.predict(mel_spectrogram)
    
    # Assuming it's a binary classification, use 0.5 as threshold
    predicted_class = "fake" if predictions[0][0] < 0.5 else "real"
    
    return predicted_class, predictions[0][0]

@app.post("/classify/")
async def classify_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"./temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Load the model
        model_path = "./CNNaudio_classifier.h5"
        model = load_model(model_path)

        # Load audio using librosa
        audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration)

        if len(audio) < sample_rate * duration:
            raise HTTPException(status_code=400, detail="Audio file too short")

        # Perform prediction
        result, confidence = predict_class(audio, model)

        # Delete temporary file
        os.remove(file_path)

        # Convert confidence to native Python float
        confidence = float(confidence)

        return {"result": result, "confidence": confidence}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

