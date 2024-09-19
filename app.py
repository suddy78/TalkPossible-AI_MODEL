from flask import Flask, request, jsonify, json
from model.speakingSpeed import speed_model
from model.stutter import stutter_model
app = Flask(__name__)

@app.route('/')
def success():
    return '접속 성공!'

@app.route('/speed_model', methods=['POST'])
def speed_model_endpoint():
    data = request.get_json()

    if 'file_names' not in data:
        return

    file_names = data['file_names']

    predict = speed_model(file_names)
    return jsonify(predict)

@app.route('/stutter_model', methods=['POST'])
def stutter_model_endpoint():
    data = request.get_json()

    if 'audio_name' not in data:
        return

    audio_name = data['audio_name']
    # stutter_model 함수 호출
    result = stutter_model(audio_name)

    # 결과 반환 (JSON 형태)
    if result['status_code'] == 200:
        response = app.response_class(
            response=json.dumps(result['data'], ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return '', 204  # 말더듬이 감지되지 않으면 204 No Content 반환