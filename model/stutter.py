from azure.storage.blob import BlobServiceClient
import requests
import librosa
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa.display
import librosa.feature
from io import BytesIO
import boto3
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
from pydub import AudioSegment
import re

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

    def cut_audio_by_words(audio_url,words_without_punctuation):
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

            words.append((word_buffer, f"word_{idx}_{word_info['word']}.wav"))

        return words

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

    # 구두점이 포함된 인덱스 번호를 찾는 함수 = 문장 끝을 저장하는 리스트 함수
    def get_punctuation_indices(words):
        # 구두점을 포함한 인덱스를 저장할 리스트
        punctuation_indices = []

        # 구두점 패턴
        punctuation_pattern = r'[.,!?;]'

        # 리스트를 순회하며 구두점 포함 여부 확인
        for i, word in enumerate(words):
            if re.search(punctuation_pattern, word):
                punctuation_indices.append(i)

        return punctuation_indices

    def find_sentence_indices(detected_word_indices, punctuation_indices):
        sentence_indices = []
        sentence_start_idx = 0

        for punct_idx in punctuation_indices:
            sentence_end_idx = punct_idx
            sentence_indices.append((sentence_start_idx, sentence_end_idx))
            sentence_start_idx = sentence_end_idx + 1

        if sentence_start_idx < len(detected_word_indices):
            sentence_indices.append((sentence_start_idx, len(detected_word_indices) - 1))

        return sentence_indices

    def cut_audio_by_sentence(audio_url, words_without_punctuation, sentence_indices, detected_word_indices):
        response = requests.get(audio_url)
        audio_data = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_data, format="wav")

        sentences = []

        # `Detected word indices`를 포함하는 문장 인덱스 추출
        for idx in detected_word_indices:
            for i, (start_idx, end_idx) in enumerate(sentence_indices):
                if start_idx <= idx <= end_idx:
                    try:
                        if start_idx >= len(words_without_punctuation) or end_idx >= len(words_without_punctuation):
                            continue

                        sentence_start_time = float(words_without_punctuation[start_idx]["start_time"]) * 1000

                        if i + 1 < len(sentence_indices):
                            next_start_idx = sentence_indices[i + 1][0]
                            if next_start_idx < len(words_without_punctuation):
                                sentence_end_time = float(words_without_punctuation[next_start_idx]["start_time"]) * 1000
                            else:
                                sentence_end_time = float(words_without_punctuation[end_idx]["end_time"]) * 1000
                        else:
                            sentence_end_time = float(words_without_punctuation[end_idx]["end_time"]) * 1000

                        if sentence_start_time < sentence_end_time:
                            sentence_audio = audio[int(sentence_start_time):int(sentence_end_time)]
                            if len(sentence_audio) > 0:
                                sentence_buffer = BytesIO()
                                sentence_audio.export(sentence_buffer, format="wav")
                                sentence_buffer.seek(0)
                                sentences.append((sentence_buffer, f"{audio_name[:-4]}_sentence_{i + 1}.wav"))
                                print(f"문장 {i + 1} 메모리에 저장 완료: {start_idx}부터 {end_idx}까지")
                            else:
                                print(f"문장 {i + 1} 오디오 잘라내기 실패: {start_idx}부터 {end_idx}까지")
                        else:
                            print(f"문장 {i + 1} 오디오 잘라내기 실패: 시작 시간이 끝 시간보다 큽니다.")
                            continue

                    except ValueError as e:
                        print(f"문장 인덱스 처리 오류: {e}")
                        continue

        if not sentences:
            print("저장된 문장이 없습니다. 오디오 업로드 중단.")

        return sentences


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
        return predictions[0, 0] > 0.98


    def upload_to_blob(data_buffer, blob_name):
        blob_client = blob_service_client.get_blob_client(container="blob2", blob=blob_name)
        try:
            print(f"Uploading {blob_name} to Azure Blob Storage...")
            blob_client.upload_blob(data_buffer, overwrite=True)
            blob_url = f"https://storage4stt0717.blob.core.windows.net/blob2/{blob_name}"
            print(f"Successfully uploaded {blob_name}. URL: {blob_url}")
            return blob_url
        except Exception as e:
            print(f"Failed to upload {blob_name} to Azure Blob Storage: {str(e)}")
            return None

    def upload_to_s3(img_buffer, image_name):
        img_buffer.seek(0)
        s3_client.upload_fileobj(img_buffer, s3_bucket_name, image_name,
                                 ExtraArgs={
                                     "ContentType": "image/png"
                                 })
        image_url = f"https://{s3_bucket_name}.s3.{s3_region}.amazonaws.com/{image_name}"
        return image_url

    def process_audio_workflow(audio_url, words_with_punctuation, words_without_punctuation, transcription_data):
        matched_timestamps = match_punctuation_and_timestamps(words_with_punctuation, words_without_punctuation)
        punctuation_indices = get_punctuation_indices(words_with_punctuation)
        model = load_model()
        detected_words = []
        detected_word_indices = []
        audio_urls = []
        image_urls = []

        # 어절별 오디오 자르고 MFCC 이미지 생성
        word_audio_data = cut_audio_by_words(audio_url, words_without_punctuation)

        # 각 어절에 대해 더듬이 감지
        for word_buffer, word_name in word_audio_data:
            img_buffer = create_mfcc_image(word_buffer)

            if predict_image(model, img_buffer):
                detected_word_idx = int(word_name.split("_")[1])
                detected_word = words_without_punctuation[detected_word_idx]['word']
                detected_words.append(detected_word)

                # 감지된 단어의 인덱스 저장
                detected_word_indices.append(detected_word_idx)

                # 디버깅: 감지된 단어 출력
                print(f"감지된 단어: {detected_word} (인덱스: {detected_word_idx})")

                # MFCC 이미지 업로드
                image_name = f"{audio_name[:-4]}_{word_name.split('.')[0]}_mfcc.png"
                image_url = upload_to_s3(img_buffer, image_name)
                image_urls.append(image_url)

        # 감지된 단어가 포함된 문장을 자르고 업로드
        if detected_words:
            sentence_indices = find_sentence_indices(detected_word_indices, punctuation_indices)
            sentences = cut_audio_by_sentence(audio_url, words_without_punctuation, sentence_indices, detected_word_indices)
            if not sentences:
                print("저장된 문장이 없습니다. 오디오 업로드 중단.")
            else:
                for i, (sentence_audio, sentence_name) in enumerate(sentences):
                    if sentence_audio:  # sentence_audio가 유효한지 확인
                        sentence_url = upload_to_blob(sentence_audio, sentence_name)
                        if sentence_url:
                            audio_urls.append(sentence_url)
                            print(f"오디오 업로드 성공: {sentence_name}")
                        else:
                            print(f"오디오 업로드 실패: {sentence_name}")
                    else:
                        print(f"오디오 잘라내기 실패: {sentence_name}")

        if detected_words:
            return {
                "status_code": 200,
                "data": {
                    "audio_url": audio_urls,
                    "image_url": image_urls,
                    "words": detected_words
                }
            }
        else:
            return {"status_code": 204}

    # Blob Storage의 퍼블릭 URL
    audio_url = f"{storage_URL}/{audio_name}"

    # 워크플로우 실행
    words_with_punctuation = transcription_data["combinedRecognizedPhrases"][0]["display"].split()
    words_without_punctuation = get_offset_duration(transcription_data)

    return process_audio_workflow(audio_url, words_with_punctuation, words_without_punctuation, transcription_data)
