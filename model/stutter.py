# import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient
import os
import requests
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment
import re
from io import BytesIO
import glob
import boto3
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
from flask import jsonify

def stutter_model(audio_name):

    # azure blob 설정
    connect_str = "DefaultEndpointsProtocol=https;AccountName=storage4stt0717;AccountKey=5kLAv9agt3F7Bndp5SvoTUmxmKPPfpxj67YDsgMoF4NRLoNqvmWnG/fVm8V/zoT4BXnXM47SmlV6+AStt4KmXA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    subscription_key = "9aebda1f74c84a2cba78b07be0257969"
    container_name = "blob2"

    # aws s3 설정
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

    # blob storage의 url
    storage_URL = "https://storage4stt0717.blob.core.windows.net/blob1"
    file_URL = f"{storage_URL}/{audio_name}.wav"

    def get_transcription_id(file_path):

        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-Type': 'application/json'
        }
        data = {
            "contentUrls": file_path,
            "locale": "ko-KR",
            "displayName": "wav to result plzzzz",
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

    def get_transcription_result(transcription_id):
        status = get_transcription_status(transcription_id)

        if status.json()["status"] == "Succeeded":

            results_url = status.json()["links"]['files']
            headers = {
                'Ocp-Apim-Subscription-Key': subscription_key,
            }

            script = requests.get(results_url, headers=headers)

            # kind가 Transcription인 것의 contentUrl로 get 요청을 보내면 됨. (header 필요 x)
            Transcription_url = script.json()["values"][0]["links"]["contentUrl"]
            transcription_data = requests.get(Transcription_url)
            return transcription_data
        else:
            return status

    transcription_data = get_transcription_result(transcription_id)

    def get_offset_duration(transcription_data):
        words_raw = transcription_data.json()["recognizedPhrases"][0]["nBest"][0]["words"]
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

    words = get_offset_duration(transcription_data)
    time_stamps = get_offset_duration(transcription_data)

    # 구두점이 포함된 어절 리스트와 타임스탬프가 포함된 어절 리스트 매칭
    def match_punctuation_and_timestamps(words_with_punctuation, words_without_punctuation):
        matched_timestamps = []
        word_idx = 0

        for idx, word_with_punc in enumerate(words_with_punctuation):
            cleaned_word = word_with_punc.rstrip('.,!?')

            while word_idx < len(words_without_punctuation):
                word_without_punc = words_without_punctuation[word_idx]["word"]

                if cleaned_word == word_without_punc:
                    matched_timestamps.append({
                        "word": word_with_punc,
                        "start_time": words_without_punctuation[word_idx]["start_time"],
                        "end_time": words_without_punctuation[word_idx]["end_time"]
                    })
                    word_idx += 1
                    break
                else:
                    matched_timestamps.append({
                        "word": word_with_punc,
                        "start_time": matched_timestamps[-1]["end_time"] if matched_timestamps else "0.000",
                        "end_time": matched_timestamps[-1]["end_time"] if matched_timestamps else "0.000"
                    })
                    break

        return matched_timestamps

    def cut_audio_by_sentence(audio_url, matched_timestamps, words_without_punctuation):
        # 웹 URL에서 오디오 파일 다운로드
        response = requests.get(audio_url)
        audio_data = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_data, format="wav")

        sentence_start_time = float(words_without_punctuation[0]["start_time"])  # 첫 문장의 시작 시간 설정

        # 어절 단위 오디오 커팅 및 문장 구분
        sentences = []
        for idx, word_info in enumerate(matched_timestamps):
            word = word_info["word"]
            start_ms = int(float(word_info["start_time"]) * 1000)
            end_ms = int(float(word_info["end_time"]) * 1000)

            # 어절 오디오 커팅
            word_audio = audio[start_ms:end_ms]

            # 구두점이 포함된 어절을 만나면 문장을 끝냄
            if any(punct in word for punct in [".", "!", "?"]):
                sentence_end_time = float(words_without_punctuation[idx]["end_time"]) * 1000
                sentence_audio = audio[int(sentence_start_time * 1000):int(sentence_end_time)]

                # 문장 단위로 자른 오디오를 메모리에 저장
                sentence_buffer = BytesIO()
                sentence_audio.export(sentence_buffer, format="wav")
                sentence_buffer.seek(0)

                # 다음 문장의 시작 시간 설정
                sentence_start_time = float(words_without_punctuation[idx + 1]["start_time"]) if idx + 1 < len(words_without_punctuation) else None

                # 문장을 리스트로 저장
                sentences.append((sentence_buffer, f"sentence_{idx + 1}.wav"))

            yield word_audio, f"word_{idx + 1}_{word}", sentences

    # (3) MFCC 이미지 생성 함수
    def create_mfcc_image(audio_buffer):
        audio_buffer.seek(0)
        audio, sample_rate = librosa.load(audio_buffer, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10, n_fft=400, hop_length=160)

        img_buffer = BytesIO()
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sample_rate, hop_length=160)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_buffer.seek(0)

        return img_buffer

    # (4) CNN 모델을 이용한 예측 함수
    def load_model():
        with open('model/stutter_model/model_structure.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json, custom_objects={"Sequential": tf.keras.models.Sequential})
        loaded_model.load_weights("model/stutter_model/model_weights.h5")
        return loaded_model

    def predict_image(model, img_buffer):
        img_buffer.seek(0)
        img = image.load_img(img_buffer, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        return predictions[0, 0] >= 0.65

    # (5) Blob에 메모리에서 업로드하는 함수
    def upload_to_blob(data_buffer, blob_name):
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(data_buffer)

    # (6) AWS S3에 업로드 함수
    def upload_to_s3(img_buffer, image_name):
        img_buffer.seek(0)
        s3_client.upload_fileobj(img_buffer, s3_bucket_name, image_name)
        image_url = f"https://{s3_bucket_name}.s3.{s3_region}.amazonaws.com/{image_name}"
        return image_url

    # (7) 워크플로우 실행 (저장 X-메모리에서)
    def process_audio_workflow(audio_url, words_with_punctuation, words_without_punctuation, transcription_data):
        matched_timestamps = match_punctuation_and_timestamps(words_with_punctuation, words_without_punctuation)
        model = load_model()
        words = []
        audio_urls = []
        image_urls = []

        for word_audio, word_name, sentences in cut_audio_by_sentence(audio_url, matched_timestamps, words_without_punctuation):
            word_buffer = BytesIO()
            word_audio.export(word_buffer, format="wav")
            word_buffer.seek(0)

            img_buffer = create_mfcc_image(word_buffer)

            if predict_image(model, img_buffer):
                # 해당 어절 기록
                words.append(word_name)

                # 문장 Blob 업로드
                for sentence_audio, sentence_name in sentences:
                    upload_to_blob(sentence_audio, sentence_name)
                    audio_url = f"{storage_URL}/{sentence_name}"
                    audio_urls.append(audio_url)

                # MFCC 이미지 S3 업로드
                image_name = f"{word_name}_mfcc.png"
                image_url = upload_to_s3(img_buffer, image_name)
                image_urls.append(image_url)

        if words:
            return {
                "status_code": 200,
                "data": {
                    "audio_url": audio_urls,
                    "image_url": image_urls,
                    "words": words
                }
            }
        else:
            return {"status_code": 204}

    # Blob Storage의 퍼블릭 URL
    audio_url = file_URL

    # 시작점
    words_with_punctuation = transcription_data["combinedRecognizedPhrases"][0]["display"].split()
    words_without_punctuation = get_offset_duration(transcription_data)

    return process_audio_workflow(audio_url, words_with_punctuation, words_without_punctuation, transcription_data)