from azure.storage.blob import BlobServiceClient
import requests
import librosa
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.feature
from io import BytesIO
import boto3
from pydub import AudioSegment
import re
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')

def stutter_model(audio_name):
    # Azure Blob 설정
    connect_str = "DefaultEndpointsProtocol=https;AccountName=storage4stt0717;AccountKey=5kLAv9agt3F7Bndp5SvoTUmxmKPPfpxj67YDsgMoF4NRLoNqvmWnG/fVm8V/zoT4BXnXM47SmlV6+AStt4KmXA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    subscription_key = "9aebda1f74c84a2cba78b07be0257969"

    # AWS S3 설정
    aws_access_key = "AKIA2UC27XQO7KTCEFUI"
    aws_secret_key = "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"
    s3_bucket_name = "talkpossible-stutter-img-bucket"
    s3_region = "ap-northeast-2"

    # Boto3 S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=s3_region
    )

    # Blob Storage의 URL
    storage_URL = "https://storage4stt0717.blob.core.windows.net/blob1"
    file_URL = [f"{storage_URL}/{audio_name}"]

    def get_transcription_id(file_path):
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-Type': 'application/json'
        }
        data = {
            "contentUrls": file_path,
            "locale": "ko-KR",
            "displayName": "wav to result",
            "destinationContainerUrl": storage_URL,
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

    def get_transcription_result(status):
        results_url = status["links"]['files']
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
        }
        script = requests.get(results_url, headers=headers)
        Transcription_url = script.json()["values"][0]["links"]["contentUrl"]
        transcription_data_response = requests.get(Transcription_url)
        transcription_data = transcription_data_response.json()
        return transcription_data

    t_id = get_transcription_id(file_URL)

    # Status 체크
    while True:
        t_status = get_transcription_status(t_id).json()
        if t_status['status'] == 'Succeeded':
            break
        elif t_status['status'] == 'Failed':
            raise Exception("Transcription failed.")
        time.sleep(5)  # 5초 대기 후 다시 상태 확인

    # 결과 받기
    transcription_data = get_transcription_result(t_status)

    def get_offset_duration(transcription_data):
        words_raw = transcription_data["recognizedPhrases"][0]["nBest"][0]["words"]
        words_list = []
        for word in words_raw:
            offset = float(word['offset'].replace('PT', '').replace('S', ''))
            duration = float(word['duration'].replace('PT', '').replace('S', ''))
            end_time = offset + duration
            words_list.append({
                "word": word['word'],
                "start_time": f"{offset:.3f}",
                "end_time": f"{end_time:.3f}"
            })
        return words_list

    def cut_audio_by_words(audio_url, words_without_punctuation):
        response = requests.get(audio_url)
        audio_data = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_data, format="wav")

        words =[]

        for idx, word_info in enumerate(words_without_punctuation):
            start_ms = int(float(word_info["start_time"]) * 1000)
            end_ms = int(float(word_info["end_time"]) * 1000)
            word_audio = audio[start_ms:end_ms]

            word_buffer = BytesIO()
            word_audio.export(word_buffer, format="wav")
            word_buffer.seek(0)

            words.append((word_buffer, f"{audio_name[:-4]}_word_{idx}_{word_info['word']}.wav"))

        return words

    # 모델 로드 (더듬음 판별용 모델과 종류 분류용 모델)
    stutter_model = joblib.load('model/stutter_model/stutter_model_1.pkl')
    xgb_model = xgb.XGBClassifier()

    # 저장한 모델 불러오기
    xgb_model.load_model('model/stutter_model/xgb_bi_1026.model')


    # MFCC 특성 추출 함수
    def get_scaled_mfcc(audio_buffer):
        audio_buffer.seek(0)
        y, sr = librosa.load(audio_buffer, sr=None)
        if len(y) == 0:
            return None
        y_mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        return y_mfcc

    # 말더듬 종류 분류를 위한 MFCC 추출 함수
    def extract_mfcc_from_audio_buffer(audio_buffer):
        audio_buffer.seek(0)
        audio, sample_rate = librosa.load(audio_buffer, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20).T, axis=0)
        return mfccs

    # 파장 이미지 생성 함수
    def create_waveform_image(wav, sr):
        plt.figure(figsize=(10, 4))
        plt.plot(wav)
        plt.axis('off')
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_buffer.seek(0)
        return img_buffer

    # 문장 저장
    EXTEND_DURATION_MS = 200

    def upload_to_blob(blob_name, original_start_time, original_end_time, full_audio):
        # 기존 오디오 파일에서 앞뒤로 0.5초씩 추가하여 잘라내기
        start_time_ms = max(0, original_start_time * 1000 - EXTEND_DURATION_MS)  # 시작 시간이 0보다 작아지지 않게 함
        end_time_ms = min(len(full_audio), original_end_time * 1000 + EXTEND_DURATION_MS)  # 끝 시간이 파일 길이를 넘지 않게 함

        extended_audio = full_audio[start_time_ms:end_time_ms]

        # 확장된 오디오를 버퍼에 저장
        extended_buffer = BytesIO()
        extended_audio.export(extended_buffer, format="wav")
        extended_buffer.seek(0)

        blob_client = blob_service_client.get_blob_client(container="blob2", blob=blob_name)
        try:
            print(f"Uploading {blob_name} to Azure Blob Storage...")
            blob_client.upload_blob(extended_buffer, overwrite=True)
            blob_url = f"https://storage4stt0717.blob.core.windows.net/blob2/{blob_name}"
            print(f"Successfully uploaded {blob_name}. URL: {blob_url}")
            return blob_url
        except Exception as e:
            print(f"Failed to upload {blob_name} to Azure Blob Storage: {str(e)}")
            return None

    # 파장 이미지 AWS S3에 업로드 함수
    def upload_waveform_to_s3(img_buffer, image_name):
        img_buffer.seek(0)
        s3_client.upload_fileobj(img_buffer, s3_bucket_name, image_name, ExtraArgs={"ContentType": "image/png"})
        image_url = f"https://{s3_bucket_name}.s3.{s3_region}.amazonaws.com/{image_name}"
        return image_url

    # 주요 로직 - 어절 분석 및 저장
    def process_stutter_detection(audio_url, words_without_punctuation, audio_name):
        detected_words = []
        audio_urls = []
        image_urls = []
        stutter_types = []

        response = requests.get(audio_url)
        audio_data = BytesIO(response.content)
        full_audio = AudioSegment.from_file(audio_data, format="wav")

        # 어절별로 더듬음 감지
        word_audio_data = cut_audio_by_words(audio_url, words_without_punctuation)
        for word_buffer, word_name in word_audio_data:
            # 더듬음 모델에 넣을 MFCC 추출
            mfcc_features = get_scaled_mfcc(word_buffer)
            if mfcc_features is not None:
                stutter_probabilities = stutter_model.predict_proba([mfcc_features])

                # 클래스 1(더듬음) 확률이 일정값 이상일 때 더듬음으로 예측
                if stutter_probabilities[0][1] >= 0.9:  # 클래스 1에 대한 확률 확인
                    stutter_prediction = 1  # 더듬음으로 분류
                else:
                    stutter_prediction = 0  # 정상 발화로 분류

                if stutter_prediction == 1:  # 더듬음인 경우
                    detected_word_idx = int(word_name.split("_")[-2])
                    word_info = words_without_punctuation[detected_word_idx]
                    detected_word = words_without_punctuation[detected_word_idx]['word']
                    detected_words.append(detected_word)

                    # 더듬음 종류 분류 모델 실행
                    mfcc_features_for_type = extract_mfcc_from_audio_buffer(word_buffer)
                    stutter_type_prediction = xgb_model.predict(mfcc_features_for_type.reshape(1, -1))

                    # 더듬음 종류 출력 및 결과 저장
                    stutter_type = "Sound Repetition" if stutter_type_prediction == 0 else "Prolongation"
                    stutter_types.append(stutter_type)

                    # 파장 이미지 생성 및 S3 업로드
                    word_buffer.seek(0)
                    wav, sr = librosa.load(word_buffer, sr=None)
                    waveform_image = create_waveform_image(wav, sr)
                    image_name = f"{audio_name[:-4]}_{word_name.split('.')[0]}_waveform.png"
                    image_url = upload_waveform_to_s3(waveform_image, image_name)
                    image_urls.append(image_url)

                    # 감지된 어절의 오디오 저장
                    word_url = upload_to_blob(word_name, float(word_info["start_time"]),
                                              float(word_info["end_time"]), full_audio)
                    audio_urls.append(word_url)


        # 최종 결과 반환
        if detected_words:
            return {
                "status_code": 200,
                "data": {
                    "audio_url": audio_urls,
                    "image_url": image_urls,
                    "words": detected_words,
                    "stutter_type": stutter_types
                }
            }
        else:
            return {"status_code": 204}


    # Blob Storage의 퍼블릭 URL
    audio_url = f"{storage_URL}/{audio_name}"

    words_without_punctuation = get_offset_duration(transcription_data)

    return process_stutter_detection(audio_url, words_without_punctuation, audio_name)
