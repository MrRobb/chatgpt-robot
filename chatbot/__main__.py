
import argparse
import mimetypes
import tempfile
import time

import requests
import speech_recognition as sr
import pyaudio
from revChatGPT.ChatGPT import Chatbot
from gtts import gTTS
import playsound
import langdetect
from dotenv import load_dotenv
import os

AUDIO_DEVICE = 9
WHISPER_MODEL_ID = "openai/whisper-medium"
HF_API_TOKEN = "Insert here your Hugging Face token"
CHATGPT_SESSION_TOKEN_ENV_VAR = "CHATGPT_SESSION_TOKEN"
INITIAL_PROMPT = """
The following is a conversation with a robot waiter named Richi. The waiter is helpful, creative, very fun and very friendly. The human transcription is error-prone but the waiter able to understand it. The waiter can take orders from the customer and provide useful information about the cafe.

Hello.
"""


def recognize_speech(audio: sr.AudioData) -> str:
    bts = audio.get_wav_data()
    content_type = mimetypes.guess_type("data.wav")[0]

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": content_type}
    api_url = f"https://api-inference.huggingface.co/models/{WHISPER_MODEL_ID}"
    response = requests.post(api_url, headers=headers, data=bts, timeout=60)
    if "text" in response.json():
        print(response.json())
        return response.json()["text"]
    else:
        print(f"ERROR: {response.json()}")


def speech_to_text(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
    try:
        print("Recognizing...")
        start = time.time()
        recognized_text = recognize_speech(audio)
        # recognized_text = recognizer.recognize_whisper(audio, model="tiny")
        # recognized_text = recognizer.recognize_google(audio)
        end = time.time()
        print("Took {}".format(end - start))
        return recognized_text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="medium",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large"]
    )
    parser.add_argument(
        "--non_english",
        action='store_true',
        help="Don't use the english model."
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we consider it a new line in the transcription.",
        type=float
    )
    args = parser.parse_args()
    return args


def print_audio_devices():
    # Get audio
    audio = pyaudio.PyAudio()
    count = audio.get_device_count()
    print(f"Audio Devices: {count}")

    # Print audio devices
    for device_index in range(count):
        device_info = audio.get_device_info_by_index(device_index)
        print(device_index, device_info)


def init_conversation() -> Chatbot:
    # Load env
    load_dotenv()
    chatbot = Chatbot({"session_token": os.getenv(CHATGPT_SESSION_TOKEN_ENV_VAR)})
    chatbot.ask(INITIAL_PROMPT)
    return chatbot


def get_response(chatbot: Chatbot, text: str) -> str:
    print("Thinking response...")
    start = time.time()
    response = chatbot.ask(text)
    end = time.time()
    print("Response took {}".format(end - start))
    return response["message"] if "message" in response else ""


def text_to_speech(text: str, language: str = 'en'):

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        tts = gTTS(text=text, lang=language)
        tts.save(temp_file.name)
        playsound.playsound(temp_file.name)


def main():
    # Parse args
    args = parse_args()

    # Print audio devices
    print_audio_devices()

    # Create recognizer
    r = sr.Recognizer()
    r.energy_threshold = args.energy_threshold
    r.dynamic_energy_threshold = False

    # Create microphone
    m = sr.Microphone(device_index=AUDIO_DEVICE)
    with m as source:
        r.adjust_for_ambient_noise(source)

    # Create conversation
    conversation = init_conversation()

    # Introduce the robot
    response = get_response(conversation, "Introduce yourself")
    language = langdetect.detect(response)
    print(f"Bot ({language}): {response}")

    # Speak response
    # text_to_speech(response, language)

    # Start listener
    while True:
        try:
            with m as source:
                # Speech to text
                print("Listening...")
                audio = r.listen(source)
                text = speech_to_text(r, audio)
                print(f"Human: {text}")

                if text is not None:
                    # Get response
                    response = get_response(conversation, text)
                    language = langdetect.detect(response)
                    print(f"Bot ({language}): {response}")

                    # Speak response
                    text_to_speech(response, language)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
