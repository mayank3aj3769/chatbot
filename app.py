import os 
from dotenv import load_dotenv
from pathlib import Path
import warnings
import threading
import keyboard
import pyttsx3
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

from pdf import load_pdf,text_split
from model import get_audio,speak,download_hf_embeddings

skip_audio = False

def wait_for_audio_or_key():
    audio_input = None
    key_input = None

    def listen_for_audio():
        nonlocal audio_input
        audio_input =  get_audio()

    def listen_for_key():
        nonlocal key_input
        key_input = input("Input Prompt: ")

    audio_thread = threading.Thread(target=listen_for_audio)
    key_thread = threading.Thread(target=listen_for_key)

    audio_thread.start()
    key_thread.start()

    while audio_thread.is_alive() and key_thread.is_alive():
        pass

    if key_input is not None:
        return key_input
    return audio_input

def skip_audio_output():
    global skip_audio
    skip_audio = False

    def listen_for_skip():
        global skip_audio
        print("Press 's' to skip the audio output.")
        while True:
            if keyboard.is_pressed('s') or keyboard.is_pressed('S'):
                skip_audio = True
                break

    skip_thread = threading.Thread(target=listen_for_skip)
    skip_thread.start()

    return skip_thread

# def speak(text):
#     global skip_audio
#     engine = pyttsx3.init()
#     engine.say(text)

#     def on_start(name):
#         global skip_audio
#         skip_audio = False

#     def on_word(name, location, length):
#         global skip_audio
#         if skip_audio:
#             engine.stop()

#     engine.connect('started-utterance', on_start)
#     engine.connect('started-word', on_word)
#     engine.runAndWait()

def speak(text):
    global skip_audio

    engine = pyttsx3.init()

    def on_word(name, location, length):
        global skip_audio
        if skip_audio:
            engine.stop()

    engine.connect('started-word', on_word)
    
    engine.say(text)
    
    # Run the event loop
    engine.runAndWait()


def app():
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    global skip_audio
    print("Starting Voice-enabled RAG chatbot")
    extracted_data=load_pdf("data/")

    print(f'Lenght of extracted pdf : {len(extracted_data)}') #sample data

    print(f' Extracted data sample : {extracted_data[10]}')
    
    text_chunks=text_split(extracted_data)

    print("length of chunks: ",len(text_chunks))

    embeddings=download_hf_embeddings()
    print(f"Embeddings shape: {embeddings}")

    #run sample query
    query_result=embeddings.embed_query("Hello World")
    print("Length of sample Hello world embedding ",len(query_result))

    ## Initializing a given pinecone index/knowledge base
    index_name="mchatbot"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    docs_chunks =[t.page_content for t in text_chunks]
    docsearch=PineconeVectorStore.from_texts(docs_chunks ,embeddings, index_name=index_name)
    prompt_template=""" 
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just state that you don't know, don't try to make up an answer.


    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else.

    Helpful answer: 
    """
    PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain_type_kwargs={"prompt":PROMPT} # model/llama-7b.ggmlv3.q4_1.bin
    llm=CTransformers(model="model/llama-7b.ggmlv3.q6_K.bin",
                 model_type="llama",
                 config={'max_new_tokens':256,
                        'temperature':0.8})
    
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k':2}),
        chain_type_kwargs=chain_type_kwargs)

    #query = "What is Bayesian inference?"
    while True:
        print("Enter audio or text to search , say or type 'Exit' to quit ")
        user_input = wait_for_audio_or_key()
        if user_input is None:
            continue

        if(user_input=='exit' or user_input=='Exit'):
            print('Shutting down RAG')
            break
        result=qa({"query":user_input})
        print("Response : ",result["result"])
        print("".join(["*"]*100))
        skip_thread = skip_audio_output()
        if skip_audio==False:
            speak(result["result"])

        if skip_audio:
            print("Audio output skipped.")

        skip_thread.join()

if __name__=="__main__":
    dotenv_path = Path('.env')
    load_dotenv(dotenv_path=dotenv_path)
    PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
    os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
    app()