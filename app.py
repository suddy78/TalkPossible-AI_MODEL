from flask import Flask, request, jsonify
from model.speakingSpeed import speed_model
from model.stutter import stutter_model
app = Flask(__name__)

@app.route('/')
def success():
    return '접속 성공!'

@app.route('/speaking_model', methods=['POST'])
def speed_model_endpoint():
    data = request.get_json()

    if 'file_names' not in data:
        return jsonify({"error": "file_names 파라미터가 필요합니다."}), 400

    file_names = data['file_names']

    predict = speed_model(file_names)
    return predict

@app.route('/stutter_model', methods=['POST'])
def stutter_model_endpoint():
    data = request.get_json()

    if 'audio_name' not in data:
        return jsonify({"error": "audio_name 파라미터가 필요합니다."}), 400

    audio_name = data['audio_name']

    # stutter_model 함수 호출
    result = stutter_model(audio_name)

    # 결과 반환 (JSON 형태)
    if result['status_code'] == 200:
        return jsonify(result['data']), 200
    else:
        return '', 204  # 말더듬이 감지되지 않으면 204 No Content 반환