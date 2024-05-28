from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound
import random 

#download embedding model , and sentence 
def download_hf_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeddings

def speak(text):
    tts = gTTS(text=text, lang='en')
    ls=[i for i in range(1,100)]
    r1=random.choice(ls)
    r2=random.choice(ls)
    filename = 'voice_'+str(r1)+'_'+str(r2)+'.mp3'
    tts.save(filename)
    playsound.playsound(filename)

def get_audio():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		audio = r.listen(source)
		said = ""

		try:
		    said = r.recognize_google(audio)
		    print("Query: "+said)
		except Exception as e:
		    print("Exception: " + str(e))

	return said


    


