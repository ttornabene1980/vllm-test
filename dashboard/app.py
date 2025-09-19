import os
from langchain_openai import ChatOpenAI
from dashboard.service.ai import ReasoningLogger, llm_create
from dashboard.service.ai import ReasoningLogger, llm_create
from  dashboard.service.db import db_create
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.agent_toolkits import create_sql_agent
from langchain.callbacks import StdOutCallbackHandler

from dotenv import load_dotenv
# carica variabili dal file .env nella root
loaded = load_dotenv()
print("dotenv loaded",loaded)

db = db_create()

API_URL = os.environ.get("API_URL")

st.set_page_config(page_title="VLLM", layout="wide")
st.title("⚡ VLLM Sql agent  deepseek-coder-6.7b-instruct ")
st.write("Env"+os.environ .__str__() )
user_question = st.text_input("Ask openai:" ,"list all tables")
if st.button("Generate openai"):
    st.write("Sto pensando... 💭")
    llm = llm_create()
    reasoning =  ReasoningLogger()
    llm = llm_create()
    reasoning =  ReasoningLogger()
    agent = create_sql_agent(
            llm=llm,
            db=db,
            # prompt=prompt,
            verbose=False,
            # prompt=prompt,
            verbose=False,
            handle_parsing_errors=True,
            callbacks=[  StdOutCallbackHandler()  ] 
            callbacks=[  StdOutCallbackHandler()  ] 
        )
    try:
        answer = agent.run( user_question )
        print( "answer",answer )
    except Exception as e:
        answer = f"Errore: {e}"
        print( "Errore",e )
    finally:
        st.write("Fatto! ✅")
    st.write("Risposta del modello:")
    try:
        answer = agent.run( user_question )
        print( "answer",answer )
    except Exception as e:
        answer = f"Errore: {e}"
        print( "Errore",e )
    finally:
        st.write("Fatto! ✅")
    st.write("Risposta del modello:")
    st.write(answer)
    
    
    
st.title("⚡ VLLM deepseek-coder-6.7b-instruct ")
user_question2 = st.text_input("Ask :" ,"create html of table id|nome|citta ")
if st.button("Generate vllm2"):
    llm = llm_create()
    try:
        LLMChain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template("{question}"),
            verbose=False,
            callbacks=[  StdOutCallbackHandler()  ]
        )
        st.write("Sto pensando... 💭")
        answer = LLMChain.run( user_question2 )
        print( "answer",answer )
    
    
    
st.title("⚡ VLLM deepseek-coder-6.7b-instruct ")
user_question2 = st.text_input("Ask :" ,"create html of table id|nome|citta ")
if st.button("Generate vllm2"):
    llm = llm_create()
    try:
        LLMChain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template("{question}"),
            verbose=False,
            callbacks=[  StdOutCallbackHandler()  ]
        )
        st.write("Sto pensando... 💭")
        answer = LLMChain.run( user_question2 )
        print( "answer",answer )
    except Exception as e:
        answer = f"Errore: {e}"
        print( "Errore",e )
    finally:
        st.write("Fatto! ✅")
    st.write("Risposta del modello:")
    st.write(answer)
        answer = f"Errore: {e}"
        print( "Errore",e )
    finally:
        st.write("Fatto! ✅")
    st.write("Risposta del modello:")
    st.write(answer)