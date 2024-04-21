import whisper
import os
import joblib

from flask import Flask, request, jsonify
from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

app = Flask(__name__)

whisperModel = whisper.load_model("base")

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

def inference(audio):
    # with open ('tempSoundFile.mp3', 'wb') as myFile:
    #     myFile.write(audio)

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(whisperModel.device)

    _, probs = whisperModel.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisperModel, mel, options)

    # os.remove("tempSoundFile.mp3")

    return result.text, lang

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def extract_emotion(wav_audio_data):
    raw_text2, lang = inference(wav_audio_data)
    sentimentData = analyze_sentiment(raw_text2)

    sentimentname = ''
    sentimentval = ''

    for key, val in sentimentData.items():
        sentimentname = key
        sentimentval = val
    

    prediction = predict_emotions(raw_text2)
    probability = get_prediction_proba(raw_text2)

    return raw_text2, sentimentname, sentimentval, lang


@app.route('/extract_emotion', methods=['POST'])
def handle_request():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        # Save the file to a temporary location
        file_path = 'temp.wav'
        file.save(file_path)

        # Extract emotion from the sound file
        text, sentimentname, sentimentval, lang = extract_emotion('temp.wav')

        # Delete the temporary file
        os.remove(file_path)

        return jsonify({'text': text, "emotion": sentimentname, "probability": sentimentval, 'lang': lang}), 200
    except Exception as e:
        return jsonify({"error is": str(e)}), 500
    
@app.route('/welcome', methods=['GET'])
def handle_request_hello():
    return jsonify({'text': 'Hello world!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
