import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# App title
st.set_page_config(page_title="LLM PDF Inquirer")

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_TOKEN'] 


if "messages" not in st.session_state.keys():
    st.session_state.messages = []

if "conversation" not in st.session_state.keys():
    st.session_state.conversation = None

if "chat_history" not in st.session_state.keys():
    st.session_state.chat_history = None


def generate_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversational_chain(vectorstore):
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.5, "max_length":512})
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

with st.sidebar:
    st.subheader("Upload PDF files")
    uploaded_files = st.file_uploader("Choose a file",accept_multiple_files=True)
    text=""
    if st.button("Process"):
        with st.spinner("Processing"):
            for file in uploaded_files:
                reader = PdfReader(file)
                for page in reader.pages:
                    text+=page.extract_text()

    
        chunks = generate_text_chunks(text)
    
        vectorstore = get_vectorstore(chunks)

        st.session_state.conversation = get_conversational_chain(vectorstore)

# Store LLM generated responses
def generate_response(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    # print(response['chat_history'])
    return response['chat_history'][-1].content

# # Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages!=[] and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
