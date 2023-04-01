import os
import sys
import openai
from moviepy.editor import *
from pydub import AudioSegment
from tqdm import tqdm
import math
import shutil


def extract_audio(input_file, output_file):
    video = VideoFileClip(input_file)
    audio = video.audio
    audio.write_audiofile(output_file)


def split_audio_dynamic(input_file, max_size):
    audio = AudioSegment.from_file(input_file)
    initial_size = os.path.getsize(input_file)
    size_ratio = max_size / initial_size
    total_duration = len(audio)
    part_duration = math.floor(total_duration * size_ratio)
    parts = []

    for i in range(0, total_duration, part_duration):
        part = audio[i:i + part_duration]
        parts.append(part)

    return parts


def transcribe_audio(input_file, max_size):
    parts = split_audio_dynamic(input_file, max_size)
    transcripts = []

    temp_file = os.path.join(temp_folder, "待处理.txt")
    with open(temp_file, "w", encoding="utf-8") as temp_file_obj:
        for i, part in enumerate(tqdm(parts, desc="Transcribing audio")):
            part_file = os.path.join(temp_folder, f"part_{i}.mp3")
            part.export(part_file, format="mp3")

            with open(part_file, "rb") as file:
                transcript = openai.Audio.transcribe("whisper-1", file)
                transcripts.append(transcript["text"])

            summary = summarize_text(transcripts[-1])
            temp_file_obj.write(summary)

    return temp_file


def summarize_text(text, context=""):
    summary = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[{"role": "user", "content": f"Summarize the following text in 6 bullet points: {context} {text}"}]
    )
    return summary["choices"][0]["message"]["content"]


def is_video_file(file_path):
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov', '.wmv']
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


def discuss_summary(summary):
    print("\nEntering discussion mode. Type 'quit' to exit.")
    while True:
        prompt = input("Please enter your question: ").strip()

        if prompt.lower() == "quit":
            break

        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": f"I have summarized the video content as follows: \n\n{summary}\n"},
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )

        answer = response["choices"][0]["message"]["content"].strip()
        print(f"Assistant: {answer}")


def create_temp_folder():
    temp_folder = "temp_folder"
    os.makedirs(temp_folder, exist_ok=True)
    return temp_folder


def delete_temp_folder(temp_folder):
    shutil.rmtree(temp_folder)


if len(sys.argv) != 3:
    print("Usage: python video_summary.py input_file openai_api_key")
    sys.exit(1)

input_file = sys.argv[1]
openai.api_key = sys.argv[2]

temp_folder = create_temp_folder()

if is_video_file(input_file):
    output_file = os.path.join(temp_folder, os.path.splitext(os.path.basename(input_file))[0] + "_audio.mp3")
    print("Extracting audio from video...")
    extract_audio(input_file, output_file)
    print("Audio extracted.")
else:
    output_file = input_file

print("Transcribing audio...")
max_size = 5 * 1024 * 1024
transcript_text = transcribe_audio(output_file, max_size)
print("Transcription completed.")

transcript_file = os.path.splitext(os.path.basename(input_file))[0] + "_转录.txt"
with open(transcript_file, "w", encoding="utf-8") as file:
    file.write(transcript_text)

print("Transcription saved.")

print("Summarizing transcript...")
token_limit = 4096
summary_file = os.path.splitext(os.path.basename(input_file))[0] + "_汇总.md"

with open(os.path.join(temp_folder, "pending.txt"), "r", encoding="utf-8") as temp_file_obj:
    transcript_text = temp_file_obj.read()

transcript_tokens = transcript_text.split()
num_slices = (len(transcript_tokens) + token_limit - 1) // token_limit
summaries = []

for i in range(num_slices):
    start_token = i * token_limit
    end_token = min((i + 1) * token_limit, len(transcript_tokens))
    text_slice = " ".join(transcript_tokens[start_token:end_token])
    summary = summarize_text(text_slice)
    summaries.append(summary)

final_summary = "\n".join(summaries)

with open(summary_file, "w", encoding="utf-8") as file:
    file.write(final_summary)

print("Summary saved.")

discuss = input("\nWould you like to start a discussion on summary? (y/n): ").lower()

if discuss == "y":
    print("\nLoading GPT-3.5-turbo model...")
    discuss_summary(final_summary)

delete_temp_folder(temp_folder)
