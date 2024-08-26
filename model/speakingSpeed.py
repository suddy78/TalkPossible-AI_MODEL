import azure.cognitiveservices.speech as speechsdk
import os
import requests
import time
from pydub import AudioSegment
import re
from io import BytesIO
import glob
import wave
import contextlib

def speed_model(file_names):

    subscription_key = "9aebda1f74c84a2cba78b07be0257969"

    # blob storage의 url
    storage_URL = "https://storage4stt0717.blob.core.windows.net/blob1"

    def get_transcription_id(file_path):

        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-Type': 'application/json'
        }
        data = {
            "contentUrls": file_path,
            "locale": "ko-KR",
            "displayName": "wav to result",
            "destinationContainerUrl" : storage_URL,
            "properties": {
                "wordLevelTimestampsEnabled": True,
                "languageIdentification": {
                    "candidateLocales": [
                        "ko-KR", "en-US"
                    ],
                },
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
        current_status = requests.get(endpoint, headers=headers)
        return current_status

    def get_transcription_result(status, file_cnt):
        transcription_data_list = []

        results_url = status["links"]['files']
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
        }

        script = requests.get(results_url, headers=headers)

        # kind가 Transcription인 것의 contentUrl로 get 요청을 보내면 됨. (header 필요 x)
        for i in range(file_cnt):
            Transcription_url = script.json()["values"][i]["links"]["contentUrl"]
            transcription_data = requests.get(Transcription_url)
            transcription_json = transcription_data.json()
            transcription_sentence = transcription_json["combinedRecognizedPhrases"][0]["display"]
            transcription_data_list.append(transcription_sentence)

        return transcription_data_list

    def get_words_cnt(sentence_lst):
        ttl_words_cnt = 0

        for sentence in sentence_lst:
            words_cnt = len(sentence.split())
            ttl_words_cnt += words_cnt

        return ttl_words_cnt

    def get_audio_sec(full_file_names):
        ttl_audio_sec = 0

        for file_URL in full_file_names:

            # 파일 다운로드
            response = requests.get(file_URL)

            # 응답이 성공적일 경우에만 처리
            if response.status_code == 200:
                audio_data = BytesIO(response.content)

                # WAV 파일의 길이 계산
                with contextlib.closing(wave.open(audio_data, 'r')) as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration_ori = (frames / float(rate))  # 초 단위로 계산
                    duration = round(duration_ori, 2)    # 소수점 2자리로 반올림
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")
                return None

            ttl_audio_sec += duration

        return ttl_audio_sec

    # 변수 초기 설정
    file_cnt = len(file_names)
    ttl_word_cnt = 0
    ttl_audio_sec = 0
    ttl_audio_min = 0
    words_per_min = 0
    full_file_names = []

    # 음성 파일 총경로 생성
    for name in file_names:
        full_file_names.append(f"{storage_URL}/{name}.wav")

    # id
    t_id = get_transcription_id(full_file_names)

    # status
    while True:
        t_status = get_transcription_status(t_id).json()

        if t_status['status'] == 'Succeeded':
            break
        elif t_status['status'] == 'Failed':
            raise Exception("Transcription failed.")

        time.sleep(5)  # 5초 대기 후 다시 상태 확인

    # result
    t_data_lst = get_transcription_result(t_status, file_cnt)

    # words_cnt
    ttl_word_cnt = get_words_cnt(t_data_lst)
    print("총 단어 수: ", ttl_word_cnt)

    # audio_time
    ttl_audio_sec = get_audio_sec(full_file_names)
    print("총 sec 수: ", ttl_audio_sec)

    # 발화속도 계산
    ttl_audio_min = round(ttl_audio_sec / 60, 2)
    words_per_min = round(ttl_word_cnt / ttl_audio_min, 2)

    print("발화 속도(분당 어절 수): ", words_per_min)

    return words_per_min