from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse , StreamingResponse
import openai
import os
import aiofiles
from typing import Optional
import asyncio
from asyncio import Semaphore

import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import ast
from functools import lru_cache
# Set your OpenAI API key
#openai.api_key = "sk-8ROaM2JeFBdJDsO4JfJstjNNZyxXoO5WSt8tIbcLsZT3BlbkFJ5zAk_DsuR9LyKQ0eYnqh1W1XqQtbTeuIgzqOIi36wA"


GENDERS = ["Female", "Male"]
VALID_VOICES = ["nova", "shimmer", "echo", "onyx", "fable", "alloy"]
FEMALE_SPEAKERS = ["Malak" , "Mayar"]
MALE_SPEAKERS = ["Omar" , "Mahmoud" , "Braa" , "Ramy"]
VALID_CHATGPT_MODELS = ["gpt-3.5-turbo", "gpt-4"]
VALID_OUTPUTS = ["text" , "Text" , "audio" , "Audio"]
LANGUAGES = [
    "af",  # Afrikaans
    "ar",  # Arabic
    "hy",  # Armenian
    "az",  # Azerbaijani
    "be",  # Belarusian
    "bs",  # Bosnian
    "bg",  # Bulgarian
    "ca",  # Catalan
    "zh",  # Chinese
    "hr",  # Croatian
    "cs",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "gl",  # Galician
    "de",  # German
    "el",  # Greek
    "he",  # Hebrew
    "hi",  # Hindi
    "hu",  # Hungarian
    "is",  # Icelandic
    "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    "kn",  # Kannada
    "kk",  # Kazakh
    "ko",  # Korean
    "lv",  # Latvian
    "lt",  # Lithuanian
    "mk",  # Macedonian
    "ms",  # Malay
    "mr",  # Marathi
    "mi",  # Maori
    "ne",  # Nepali
    "no",  # Norwegian
    "fa",  # Persian
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "sr",  # Serbian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sw",  # Swahili
    "sv",  # Swedish
    "tl",  # Tagalog
    "ta",  # Tamil
    "th",  # Thai
    "tr",  # Turkish
    "uk",  # Ukrainian
    "ur",  # Urdu
    "vi",  # Vietnamese
    "cy"   # Welsh
]



# Initialize FastAPI and semaphore
app = FastAPI()

# Load Embeddings and Sentence Chunks Data (From CSV)
@lru_cache
def load_embedding_data():
    df = pd.read_csv("sentence_chunks_emb.csv")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    embeddings = [np.array(e) for e in df["embedding"].values]
    return embeddings, df.to_dict(orient="records")

embeddings, pages_and_chunks = load_embedding_data()


semaphore = Semaphore(10)  # Limit concurrency to 3 requests at a time

# Folder to store temporary files
TEMP_FOLDER = "temp_audio"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Function to transcribe audio using OpenAI's Whisper model
async def transcribe_audio(file_path: str, language: str) -> str:
    try:
        with open(file_path, "rb") as audio_file:
            transcription_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                #language=language
            )
        return transcription_response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


# Function to get embeddings using OpenAI API

def get_embedding(text, model="text-embedding-ada-002"):  # Updated model name
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)

    # Properly access the embedding from the response object
    embedding = response.data[0].embedding
    return np.array(embedding)  # Ensure it returns a numpy array for consistency


# Function to retrieve relevant resources
def retrieve_relevant_resources(query, embeddings, n_resources=5):
    query_embedding = get_embedding(query)
    # Compute cosine similarity
    similarities = [cosine_similarity([query_embedding], [e])[0][0] for e in embeddings]
    indices = np.argsort(similarities)[-n_resources:][::-1]  # Sort by similarity
    return indices


# Context generation function
def generate_context(query, embeddings, pages_and_chunks):
    indices = retrieve_relevant_resources(query, embeddings)
    context = "- " + "\n- ".join([pages_and_chunks[i]["sentence_chunks"] for i in indices])
    return context

# Function to get response from ChatGPT based on transcription
async def get_chatgpt_response(transcription: str , model_name: str) -> str:
    context = generate_context(transcription, embeddings, pages_and_chunks)
    print(context)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Only answer based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\n Question: {transcription}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChatGPT response failed: {e}")

# Function to generate speech using OpenAI's TTS API
async def text_to_speech(text: str, voice: str) -> str:
    try:
        # Create TTS audio using OpenAI's API
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

#["Omar" , "Mahmoud" , "Braa" , "Ramy"]
def get_speaker(gender, speaker):
    if gender == "Female":
        speaker_files = {
            "Malak": "nova",
            "Mayar": "shimmer",
        }
    else:
        speaker_files = {
            "Omar": "alloy",
            "Mahmoud": "echo",
            "Braa" : "fable",
            "Ramy" : "onyx"
        }
    speaker_name = speaker_files.get(speaker)

    return speaker_name


# FastAPI route to handle the entire process: upload audio, transcribe, generate response, convert to speech
@app.post("/process-audio/")
async def process_audio(
        file: UploadFile = Form(None , description = "Audio File for Speech to text"),
        input_text: str = Form("no_content", description="Input Text instead of Audio file to get response"),  # Add allowance parameter
        language: str = Form(..., description="Language code (e.g., 'ar', 'en')"),
        gender: str = Form(..., description="Gender ('Female' or 'Male')"),
        speaker: str = Form(..., description="Text to convert to speech"),
        model_name: str = Form(..., description="Chatgpt Models"),
        output_type: str = Form("audio" , description="The output format"),
        client_ip: str = Form("Unknown", description="Client IP for requester"),
        open_ai_api_key: str =Form(None , description = "OpenAI API Key is required")
):
    try:
        # Validate inputs
        if open_ai_api_key == None:
            raise HTTPException(status_code=400, detail="OpenAI API Key is required. you can use parameter 'open_ai_api_key' and enter you valid openai api key. ")
        if file == None and input_text == "no_content":
            raise HTTPException(status_code=400, detail="You didn't upload the audio file or input text.")
        if client_ip == "Unknown":
            raise HTTPException(status_code=400, detail="Please add your client_ip.")
        if language not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language. Choose from {LANGUAGES}.")
        if gender not in GENDERS:
            raise HTTPException(status_code=400, detail=f"Unsupported gender. Choose from {GENDERS}.")
        if output_type not in VALID_OUTPUTS:
            raise HTTPException(status_code=400, detail=f"Unsupported gender. Choose from {VALID_OUTPUTS}.")
        if model_name not in VALID_CHATGPT_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model selected. Choose from: {', '.join(VALID_CHATGPT_MODELS)}")
        if gender == "Male" and speaker not in MALE_SPEAKERS:
            raise HTTPException(status_code=400,
                                detail=f"Invalid male speaker selected. Choose From these {MALE_SPEAKERS}")
        if gender == "Female" and speaker not in FEMALE_SPEAKERS:
            raise HTTPException(status_code=400, detail=f"Invalid female speaker selected. Choose From these {FEMALE_SPEAKERS}")
        async with semaphore:
            # Set environment variable
            os.environ['OPENAI_API_KEY'] = open_ai_api_key
            global client
            client = OpenAI(
                # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            if input_text == "no_content" :# Handle Audio input Case 1 --> Audio output , Case 2 --> Text output
                print(f"{client_ip} in case 1 : Audio to Audio")
                if output_type == "audio": # Case 1 : Audio to Audio
                    # Save uploaded audio file temporarily
                    file_path = os.path.join(TEMP_FOLDER, file.filename)

                    async with aiofiles.open(file_path, 'wb') as out_file:
                        content = await file.read()
                        await out_file.write(content)

                    # Step 1: Transcribe the audio
                    transcription = await transcribe_audio(file_path, language)

                    # Step 2: Get response from ChatGPT
                    chatgpt_response = await get_chatgpt_response(transcription , model_name)

                    # Step 3: Convert the ChatGPT response to speech
                    speaker = get_speaker(gender,speaker)
                    tts_audio_path = await text_to_speech(chatgpt_response, speaker)
                    tts_audio_path.stream_to_file(os.path.join(TEMP_FOLDER, f"{client_ip} response.wav"))
                    # Stream the audio file back in the response
                    def audio_streaming():
                        with open(os.path.join(TEMP_FOLDER, f"{client_ip} response.wav"), "rb") as audio_file:
                            yield from audio_file

                    # Remove temporary files if needed (optional)
                    os.remove(file_path)  # Remove uploaded audio file

                    # Create a JSON response
                    json_response = JSONResponse(content={
                        "transcription": transcription,
                        "chatgpt_response": chatgpt_response,
                    }, media_type="application/json; charset=utf-8")

                    # Include the audio stream as a separate response (header or attachment)
                    return StreamingResponse(audio_streaming(), media_type="audio/wav",
                                         headers={"Content-Disposition": "attachment; filename=generated_voice.wav"})
                else: # Case 2 : Audio to Text
                    print(f"{client_ip} in case 2 : Audio to Text")
                    # Save uploaded audio file temporarily
                    file_path = os.path.join(TEMP_FOLDER, file.filename)

                    async with aiofiles.open(file_path, 'wb') as out_file:
                        content = await file.read()
                        await out_file.write(content)

                    # Step 1: Transcribe the audio
                    transcription = await transcribe_audio(file_path, language)

                    # Step 2: Get response from ChatGPT
                    chatgpt_response = await get_chatgpt_response(transcription, model_name)

                    # Remove temporary files if needed (optional)
                    os.remove(file_path)  # Remove uploaded audio file

                    # Include the audio stream as a separate response (header or attachment)
                    return JSONResponse(content={
                        "transcription": transcription,
                        "chatgpt_response": chatgpt_response,
                    }, media_type="application/json; charset=utf-8")
            else:
                if output_type == "audio":# Case 3 : Text to Audio
                    print(f"{client_ip} in case 3 : Text to Audio")
                    # Step 2: Get response from ChatGPT
                    chatgpt_response = await get_chatgpt_response(input_text, model_name)

                    # Step 3: Convert the ChatGPT response to speech
                    speaker = get_speaker(gender, speaker)
                    tts_audio_path = await text_to_speech(chatgpt_response, speaker)
                    tts_audio_path.stream_to_file(os.path.join(TEMP_FOLDER, f"{client_ip} response.wav"))

                    # Stream the audio file back in the response
                    def audio_streaming():
                        with open(os.path.join(TEMP_FOLDER, f"{client_ip} response.wav"), "rb") as audio_file:
                            yield from audio_file

                    # Include the audio stream as a separate response (header or attachment)
                    return StreamingResponse(audio_streaming(), media_type="audio/wav",
                                             headers={"Content-Disposition": "attachment; filename=generated_voice.wav"})
                else: # Case 4: Text to Text
                    print(f"{client_ip} in case 4 : Text to Text")
                    # Step 2: Get response from ChatGPT
                    chatgpt_response = await get_chatgpt_response(input_text, model_name)

                    # Include the audio stream as a separate response (header or attachment)
                    return JSONResponse(content={
                        "transcription": input_text,
                        "chatgpt_response": chatgpt_response,
                    }, media_type="application/json; charset=utf-8")

    except HTTPException as he:
        return JSONResponse(content={
            "detail": he.detail
        }, status_code=he.status_code)

    except Exception as e:
        return JSONResponse(content={
            "detail": f"An unexpected error occurred: {str(e)}"
        }, status_code=500)

# Clean up temporary files (optional)
@app.on_event("shutdown")
async def cleanup():
    for file in os.listdir(TEMP_FOLDER):
        os.remove(os.path.join(TEMP_FOLDER, file))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
