"""
A simple web application to implement a chatbot. This app uses Streamlit 
for the UI and the Python requests package to talk to an API endpoint that
implements text generation and Retrieval Augmented Generation (RAG) using LLMs
and Amazon OpenSearch as the vector database.
"""
# import boto3
import streamlit as st
import requests as req
from typing import List, Tuple, Dict

# global constants
STREAMLIT_SESSION_VARS: List[Tuple] = [
    ("generated", []),
    ("past", []),
    ("input", ""),
    ("stored_session", []),
    ("docs", []),
]
HTTP_OK: int = 200

# two options for the chatbot, 1) get answer directly from the LLM
# 2) use RAG (find documents similar to the user query and then provide
# those as context to the LLM).
MODE_RAG: str = "RAG"
MODE_TEXT2TEXT: str = "Text Generation"
# MODE_VALUES: List[str] = [MODE_RAG, MODE_TEXT2TEXT]
MODE_VALUES: List[str] = [MODE_RAG]

TEXT2TEXT_MODEL_LIST: List[str] = [
    "OpenAI(gpt-3.5-turbo)",
    "BedRock(titan-tg1-large)",
    "CSDC(buffer-instruct-internlm-001)",
]

EMBEDDINGS_MODEL_LIST: Dict[str, List[str]] = {
    "OpenAI(gpt-3.5-turbo)": ["OpenAI(text-embedding-ada-002)"],
    "BedRock(titan-tg1-large)": ["BedRock(titan-embeddings)"],
    "CSDC(buffer-instruct-internlm-001)": ["CSDC(buffer-embedding-002)"],
}

API_RAG_PATH: Dict[str, Dict[str, str]] = {
    "OpenAI(gpt-3.5-turbo)": {
        "OpenAI(text-embedding-ada-002)": "qa-openai-opensearch",
    },
    "BedRock(titan-tg1-large)": {
        "BedRock(titan-embeddings)": "qa-csdc-opensearch",
    },
    "CSDC(buffer-instruct-internlm-001)": {
        "CSDC(buffer-embedding-002)": "qa-csdc-opensearch",
    },
}

API_TEXT2TEXT_PATH: Dict[str, Dict[str, str]] = {
    "OpenAI(gpt-3.5-turbo)": {
        "OpenAI(text-embedding-ada-002)": "qa-openai-opensearch",
    },
    "BedRock(titan-tg1-large)": {
        "BedRock(titan-embeddings)": "qa-csdc-opensearch",
    },
    "CSDC(buffer-instruct-internlm-001)": {
        "CSDC(buffer-embedding-002)": "qa-csdc-opensearch",
    },
}

# API endpoint
# This is the base part of the api endpoint
api: str = "https://obj90tyol8.execute-api.ap-northeast-1.amazonaws.com/api"

# api_text2text_ep: str = f"{api}/api/v1/llm/text2text"
# print(f"api_rag_ep={api_rag_ep}\napi_text2text_ep={api_text2text_ep}")

####################
# Streamlit code
####################

# Page title
st.set_page_config(page_title="Virtual assistant for knowledge base 🤖", layout="wide")

# keep track of conversations by using streamlit_session
_ = [st.session_state.setdefault(k, v) for k, v in STREAMLIT_SESSION_VARS]


# Define function to get user input
def get_user_input() -> str:
    """
    Returns the text entered by the user
    """
    print(st.session_state)
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Ask me a question and I will consult the knowledge base to answer...",
        label_visibility="hidden",
    )
    return input_text


# sidebar with options
with st.sidebar.expander("⚙️", expanded=True):
    st.title("Configuration")
    mode = st.selectbox(label="Mode", options=MODE_VALUES)
    text2text_model = st.selectbox(label="Text2Text Model", options=TEXT2TEXT_MODEL_LIST)
    embeddings_model = st.selectbox(label="Embeddings Model", options=EMBEDDINGS_MODEL_LIST.get(text2text_model, []))
    k_value = st.number_input("k", min_value=1, max_value=10, value=3)
    temperature_value = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    # if text2text_model == TEXT2TEXT_MODEL_LIST[0]:
    #     openai_api_key = st.text_input("Please enter your OpenAI API key:", type="password")
    #     if openai_api_key:
    #         pass
    #         # os.environ["OPENAI_API_KEY"] = api_key

# streamlit app layout sidebar + main panel
# the main panel has a title, a sub header and user input textbox
# and a text area for response and history
st.title("🤖 QaBot for a knowledge base")
st.subheader(f" Powered by :blue[{text2text_model}] for text generation and :blue[{embeddings_model}] for embeddings")

# 增加清除历史记录的按钮
if st.button("Clear Chat History"):
    st.session_state.past = []
    st.session_state.generated = []
    st.session_state.docs = []
    st.experimental_rerun()

# get user input
user_input: str = get_user_input()

# based on the selected mode type call the appropriate API endpoint
if user_input:
    # headers for request and response encoding, same for both endpoints
    headers: Dict = {"accept": "application/json", "Content-Type": "application/json"}
    output: str = None
    new_docs = []
    if mode == MODE_TEXT2TEXT:
        api_text2text_ep: str = f"{api}/{API_TEXT2TEXT_PATH[text2text_model][embeddings_model]}"
        data = {
            "model": "gpt",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hi there! I'm QaBot, an AI assistant. I can help you with things like answering questions, providing information, and helping with tasks. How can I help you?",
                },
                {"role": "user", "content": user_input},
            ],
            "temperature": 0.2,
        }
        resp = req.post(api_text2text_ep, headers=headers, json=data)
        if resp.status_code != HTTP_OK:
            output = resp.text
        else:
            output = resp.json()["choices"][0]["message"]["content"]
    elif mode == MODE_RAG:
        api_rag_ep: str = f"{api}/{API_RAG_PATH[text2text_model][embeddings_model]}"
        print(f"+++++++++++ api_rag_ep={api_rag_ep} +++++++++++++++++")
        data = {
            "model": "gpt",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hi there! I'm QaBot, an AI assistant. I can help you with things like answering questions, providing information, and helping with tasks. How can I help you?",
                },
                {"role": "user", "content": user_input},
            ],
            "temperature": temperature_value,
            "env": {
                "k": k_value,
            },
        }
        resp = req.post(api_rag_ep, headers=headers, json=data)
        if resp.status_code != HTTP_OK:
            print(resp)
            output = resp.text
        else:
            response_data = resp.json()
            print(response_data)
            message_content = response_data["choices"][0]["message"]["content"]
            sources = [d["metadata"]["s3_key"] for d in response_data["docs"]]
            output = f"{message_content} \n \n Sources: {sources}"
            # 从响应数据中提取文档信息
            
            for document in response_data["docs"]:
                doc = {"page_content": document["page_content"], "s3_key": document["metadata"]["s3_key"]}
                new_docs.append(doc)

    else:
        print("error")
        output = f"unhandled mode value={mode}"
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.docs.append(new_docs)


# download the chat history
download_str: List = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🙋")
        st.success(
            st.session_state["generated"][i],
            icon="🤖",
        )
        # 检查 'docs' 是否存在并且其长度足够
        if "docs" in st.session_state and len(st.session_state["docs"]) > i:
            # 逐一遍历每一个文档
            for j in range(len(st.session_state["docs"][i])):
                # st.info(f"Page Content: {st.session_state['docs'][i][j]['page_content']}")
                # st.info(f"S3 Key: {st.session_state['docs'][i][j]['s3_key']}")
                st.markdown(
                    f"<div style='background-color: Gainsboro; padding: 1px; border-radius: 5px; margin-bottom: 5px;'>"
                    f"<b>Page Content ({j+1}):</b> {st.session_state['docs'][i][j]['page_content']}"
                    f"<hr style='margin-bottom: -2px;'>"
                    f"<div style='margin-top: -2px;'><b>S3 Key:</b> {st.session_state['docs'][i][j]['s3_key']}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)
