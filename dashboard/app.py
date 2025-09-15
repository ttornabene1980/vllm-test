from langchain_openai import ChatOpenAI
from  dashboard.service.db import db_create
import streamlit as st
import requests
import re
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.agent_toolkits import create_sql_agent
from langchain.callbacks import StdOutCallbackHandler

# from langchain.agents import create_sql_agent
# from mefa.chainapi.main import db_create, llm_create
# from langchain_community.agent_toolkits.sql.base import create_sql_agent

API_URL = "http://localhost:8001"

st.set_page_config(page_title="VLLM Function Generator", layout="wide")
st.title("⚡ VLLM Function Generator with LangChain")
user_question = st.text_input("Ask openai:" ,"reverse a string")
if st.button("Generate openai"):
    st.write("Sto pensando... 💭")
    # Modello LLM
    llm = ChatOpenAI( model="gpt-4", temperature=0)
    # llm = llm_create()
    # Prompt template
    # prompt = ChatPromptTemplate.from_template(
    #     "Rispondi passo passo alla seguente domanda: {question}"
    # )
    d = ChatPromptTemplate.from_template(
        """Rispondi passo passo a questa domanda:

    {question}
    Mostra il ragionamento prima della risposta finale."""
    )
    db = db_create()
    
    agent = create_sql_agent(
            llm=llm,
            db=db,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[   StdOutCallbackHandler() ],
        )
   
    answer = agent.run( user_question )
    # Mostra output
    st.subheader("Risposta del modello:")
    st.write(answer)



 # # Chain
    # chain = LLMChain(llm=llm, prompt=prompt)
    # # Esegui la chain
    # answer = chain.run(question=user_question)
   # crea il toolkit passando sia l'LLM che il DB
    # ottieni i tool dal toolkit
    # tools = toolkit.get_tools()
    # st.write(tools)
    
query = st.text_input("Describe the function you want:", "reverse a string")
if st.button("Generate"):
    try:
        res = requests.get(f"{API_URL}/langchain", params={"query": query}).json()
        if "result" in res:
            content= res["result"]["content"].replace("\\n", "\n").replace("\\t", "\t")
            st.text("1. Try to extract code block if present")
            code_match = re.search(r"```python(.*?)```", content, re.DOTALL)
            st.text("code_match = " + str(code_match))
            if code_match:
                code = code_match.group(1).strip()
                st.subheader("📝 Generated Python Function")
                st.code(code, language="python")
        with st.expander("🔍 Full response details"):
                st.json(res)

    except Exception as e:
        st.error(f"⚠️ API error: {e}")

st.title("⚡ VLLM Sql Agent")
query2 = st.text_input("Describe the db question you want:", "list all tables")
if st.button("GenerateSql"):
    try:
        res = requests.get(f"{API_URL}/sql-agent", params={"query": query2}).json()
        st.subheader("📝 SQL Agent Response")
        st.json(res)

    except Exception as e:
        st.error(f"⚠️ API error: {e}")

# if st.button("Generate"):
#     # with col1:
#         st.subheader("LangChain Result")
#         try:
#             res = requests.get(f"{API_URL}/langchain", params={"query": query}).json()
#             st.code(res["result"], language="python")
#         except Exception as e:
#             st.error(str(e))
    # with col2:
    #     st.subheader("DSPy Result")
    #     try:
    #         res = requests.get(f"{API_URL}/dspy", params={"query": query}).json()
    #         st.code(res["result"], language="python")
    #     except Exception as e:
    #         st.error(str(e))