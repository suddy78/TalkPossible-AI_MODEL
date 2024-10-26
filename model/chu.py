from io import BytesIO
import numpy as np
import librosa
import sklearn
import sklearn.preprocessing
from tensorflow import keras
import requests
import time
from pydub import AudioSegment

def chu_model(audio_name):
    def load_model():
        with open('model/chu_model/chu_model_structure0226.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("model/chu_model/chu_model_weights0226.weights.h5")
        return loaded_model

    # azure key
    subscription_key = "9aebda1f74c84a2cba78b07be0257969"

    # blob storage url
    storage_URL = "https://storage4stt0717.blob.core.windows.net/blob1"
    file_URL = f"{storage_URL}/{audio_name}"

    def get_transcription_id(file_path):
        headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/json'
        }
        data = {
            "contentUrls": file_path,
            "locale": "ko-KR",
            "displayName": "wav to word audio fragments",
            "destinationContainerUrl" : storage_URL,
            "properties": {
                "wordLevelTimestampsEnabled": True,
                "punctuationMode": "DictatedAndAutomatic",
                "profanityFilterMode": "Masked"
            }
        }
        endpoint = f"https://koreacentral.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"


        response = requests.post(endpoint, headers=headers, json=data)
        transcription_id = response.json()["self"].split("/")[-1]

        return transcription_id

    def get_transcription_status(transcription_id):
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
        }
        endpoint = f"https://koreacentral.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions/{transcription_id}"
        status = requests.get(endpoint, headers=headers)
        return status

    def get_transcription_result(status):
        transcription_words_list = []

        results_url = status["links"]['files']
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
        }

        script = requests.get(results_url, headers=headers)

        Transcription_url = script.json()["values"][0]["links"]["contentUrl"]
        transcription_data = requests.get(Transcription_url)
        transcription_json = transcription_data.json()
        if (len(transcription_json["recognizedPhrases"]) >= 1) :
            transcription_words_list = transcription_json["recognizedPhrases"][0]["nBest"][0]["words"]
        else :
            transcription_words_list.append([])
            print(f"해당 오디오는 분석할 수 없음.")

        return transcription_words_list

    def get_offset_duration(transcription_words_list):
        words_stack_list = []

        for word in transcription_words_list:
            # 단어 글자가 1개라면
            if len(word['word']) == 1 :
                offset = float(word['offset'].replace('PT', '').replace('S', '')) - 0.05
                duration = float(word['duration'].replace('PT', '').replace('S', '')) + 0.05
                end_time = offset + duration

                if (offset < 0) :
                    offset = 0

                words_stack_list.append({
                    "word": word['word'],
                    "start_time": f"{offset:.3f}",
                    "end_time": f"{end_time:.3f}"
                })
            else :
                continue

        return words_stack_list

    def cut_audio(audio_url, words_stack_list):
        try:
            audio_list = []

            response = requests.get(audio_url)
            response.raise_for_status()  # 응답 코드가 200이 아닌 경우 예외 발생
            audio_data = BytesIO(response.content)

            # pydub를 사용하여 AudioSegment로 변환
            audio = AudioSegment.from_file(audio_data, format="wav")

            if (len(words_stack_list) == 0) :
                return audio_list # 빈 리스트 반환

            for idx, word_info in enumerate(words_stack_list):
                start_ms = int(float(word_info["start_time"]) * 1000)
                end_ms = int(float(word_info["end_time"]) * 1000)

                word_audio = audio[start_ms:end_ms]
                audio_list.append(word_audio)

            return audio_list

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")
        except Exception as e:
            print(f"An unexpected error occurred with URL: {audio_url}")
            return audio_list # 빈 리스트 반환

    def audiosegment_to_librosa(audio_segment):
        # pydub의 AudioSegment를 numpy 배열로 변환
        samples = np.array(audio_segment.get_array_of_samples())

        # 모노 오디오일 경우 채널이 하나지만, 스테레오일 경우 채널 두 개가 필요
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)  # 스테레오를 모노로 변환

        return samples.astype(np.float32) / 32768.0  # 16-bit PCM을 float32로 변환

    def check_in_model(one_in_audio_segment):
        y = audiosegment_to_librosa(one_in_audio_segment)  # AudioSegment 객체를 numpy로 변환
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=100, n_fft=400, hop_length=160)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, 40)
        padded_mfcc = np.expand_dims(padded_mfcc, 0)

        model = load_model()
        prediction = model.predict(padded_mfcc)
        r = np.argmax(prediction)

        if (r == 0) :
            return True #추임새
        else :
            return False #비추임새

    f_chu_cnt = 0

    f_id = get_transcription_id([file_URL]) # 여기서는 리스트로 줘야해서 []해준게 맞음
    f_status = get_transcription_status(f_id).json()

    while f_status['status'] != 'Succeeded':
        f_status = get_transcription_status(f_id).json()
        time.sleep(2)

    f_transcription_words_list = get_transcription_result(f_status)

    words_stack_list = get_offset_duration(f_transcription_words_list)

    cut_audio_list = cut_audio(file_URL, words_stack_list)

    for idx, audio in enumerate(cut_audio_list):
        check_in_model_result = check_in_model(audio)
        print(check_in_model_result)
        if (check_in_model_result == True) : # 추임새
            f_chu_cnt += 1
        else :
            continue

    print(f"추임새 언급 횟수 : {f_chu_cnt}")
    print(f"한자리 단어 갯수 : {len(cut_audio_list)}")
    return f_chu_cnt