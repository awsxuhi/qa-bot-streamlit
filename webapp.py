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
import hashlib
import uuid

# global constants
STREAMLIT_SESSION_VARS: List[Tuple] = [
    ("generated", []),
    ("past", []),
    ("input", ""),
    ("stored_session", []),
    ("docs", []),
    ("session_id", ""),
]
HTTP_OK: int = 200

# two options for the chatbot, 1) get answer directly from the LLM
# 2) use RAG (find documents similar to the user query and then provide
# those as context to the LLM).

# MODE_RAG: str = "RAG"
# MODE_TEXT2TEXT: str = "Text Generation"
# MODE_VALUES: List[str] = [MODE_RAG, MODE_TEXT2TEXT]

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

CHAIN_LIST: List[str] = [
    "load_qa_chain",
    "RetrievalQA",
    "RetrievalQAwithMemory",
    "ConversationalRetrievalChain",
]

EMBEDDING_NAME_LIST: Dict[str, List[str]] = {
    "OpenAI(text-embedding-ada-002)": "openai",
    "BedRock(titan-embeddings)": "openai",
    "CSDC(buffer-embedding-002)": "csdc",
}

# API endpoint
# This is the base part of the api endpoint
api: str = "https://q05huaibxg.execute-api.ap-northeast-1.amazonaws.com/api/"


####################
# Streamlit code
####################

# Page title
st.set_page_config(page_title="Virtual assistant for knowledge base 🤖", layout="wide")

# keep track of conversations by using streamlit_session
_ = [st.session_state.setdefault(k, v) for k, v in STREAMLIT_SESSION_VARS]


# generate session id
def generate_session_id():
    if not st.session_state.get("session_id"):
        st.session_state.session_id = str(uuid.uuid4())
        print("New session ID generated")  # 添加这行来调试
    else:
        print("Existing session ID retrieved")  # 添加这行来调试
    return st.session_state.session_id


current_session_id = generate_session_id()
# print(f"Current session ID: {current_session_id}")  # 添加这行来调试
# st.markdown(f"Current session ID: {current_session_id}")


# Define function to get user input
def get_user_input() -> str:
    """
    Returns the text entered by the user
    """
    print(st.session_state)
    input_text = st.text_input(
        "You: ",
        st.session_state.get("input", ""),  # 使用get方法来提供一个默认值，xuhi@
        key="input",
        placeholder="Ask me a question and I will consult the knowledge base to answer...",
        label_visibility="hidden",
    )
    return input_text


# sidebar with options
with st.sidebar.expander("⚙️", expanded=True):
    st.title("Configuration")
    # mode = st.selectbox(label="Mode", options=MODE_VALUES)
    RAG = st.checkbox("RAG", value=True)

    # 从session_state获取参数的当前值，如果不存在则设置一个默认值
    text2text_model = st.session_state.get("text2text_model", TEXT2TEXT_MODEL_LIST[0])
    embeddings_model = st.session_state.get("embeddings_model", EMBEDDINGS_MODEL_LIST.get(text2text_model, [])[0])
    chat_history_window_value = st.session_state.get("chat_history_window_value", 3)
    k_value = st.session_state.get("k_value", 3)
    temperature_value = st.session_state.get("temperature_value", 0.0)
    chain = st.session_state.get("chain", CHAIN_LIST[0])

    # 创建UI元素，并将用户的选择保存回session_state
    text2text_model = st.selectbox(
        label="Text2Text Model", options=TEXT2TEXT_MODEL_LIST, index=TEXT2TEXT_MODEL_LIST.index(text2text_model)
    )
    st.session_state.text2text_model = text2text_model

    embeddings_model = st.selectbox(
        label="Embeddings Model",
        options=EMBEDDINGS_MODEL_LIST.get(text2text_model, []),
    )
    st.session_state.embeddings_model = embeddings_model

    chat_history_window_value = st.number_input(
        "chat_history_window", min_value=0, max_value=5, value=chat_history_window_value
    )
    st.session_state.chat_history_window_value = chat_history_window_value

    k_value = st.number_input("k", min_value=1, max_value=10, value=k_value)
    st.session_state.k_value = k_value

    temperature_value = st.slider("Temperature", min_value=0.0, max_value=1.0, value=temperature_value, step=0.1)
    st.session_state.temperature_value = temperature_value

    chain = st.selectbox(label="Chain", options=CHAIN_LIST, index=CHAIN_LIST.index(chain))
    st.session_state.chain = chain

    if st.button("Refresh KB List"):
        response = req.get(f"{api}/kb-list")
        KB_LIST = response.json()
        st.session_state.KB_LIST = KB_LIST
    else:
        KB_LIST = st.session_state.get("KB_LIST", [])

    kb_name = st.selectbox(label="Select KB", options=KB_LIST, index=0)
    st.session_state.kb_name = kb_name

    if kb_name is None:
        st.warning("Please select a knowledge base before asking questions.")
    else:
        kb_name_hash = hashlib.md5(kb_name.encode()).hexdigest()
        index_name = f"{kb_name.lower()}_{EMBEDDING_NAME_LIST[embeddings_model]}_{kb_name_hash}"


# streamlit app layout sidebar + main panel
# the main panel has a title, a sub header and user input textbox
# and a text area for response and history
st.title("🤖 QaBot for a knowledge base")
st.subheader(
    f" Session ID: :blue[{current_session_id}] Powered by :blue[{text2text_model}] and :blue[{embeddings_model}]"
)

# 增加清除历史记录的按钮
if st.button("Clear Chat History"):
    st.session_state.past = []
    st.session_state.generated = []
    st.session_state.docs = []
    st.session_state.session_id = str(uuid.uuid4())
    st.experimental_rerun()


# get user input
user_input: str = get_user_input()


# based on the selected mode type call the appropriate API endpoint
if user_input:
    # headers for request and response encoding, same for both endpoints
    headers: Dict = {"accept": "application/json", "Content-Type": "application/json"}
    output: str = None
    new_docs = []
    if not RAG:
        api_text2text_ep: str = f"{api}/chat-without-rag"
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
                "chat_history_window": chat_history_window_value,
                "text2text_model": text2text_model,
                "session_id": st.session_state.session_id,
            },
        }
        resp = req.post(api_text2text_ep, headers=headers, json=data)
        if resp.status_code != HTTP_OK:
            output = resp.text
        else:
            output = resp.json()["choices"][0]["message"]["content"]
    else:
        api_rag_ep: str = f"{api}/question-answering"
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
                "chat_history_window": chat_history_window_value,
                "k": k_value,
                "embeddings_model": embeddings_model,
                "text2text_model": text2text_model,
                "index_name": index_name,
                "chain": chain,
                "session_id": st.session_state.session_id,
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
                if chain == "load_qa_chain":
                    doc = {
                        "page_content": document["page_content"],
                        "s3_key": document["metadata"]["s3_key"],
                        "score": document["score"],
                    }
                elif chain in ("RetrievalQA", "RetrievalQAwithMemory"):  # without score
                    doc = {
                        "page_content": document["page_content"],
                        "s3_key": document["metadata"]["s3_key"],
                    }
                new_docs.append(doc)

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
                    f"<div style='margin-top: 2px; margin-bottom: 1px;'><b>S3 Key:</b> {st.session_state['docs'][i][j]['s3_key']}</div>"
                    f"<div style='margin-top: 1px; margin-bottom: 1px;'><b>Score:</b> {st.session_state['docs'][i][j]['score'] if chain == 'load_qa_chain' else 'Not Available'}</div>"
                    f"<hr style='margin-top: 1px; margin-bottom: 2px;'>"
                    f"<b>Page Content ({j+1}):</b> {st.session_state['docs'][i][j]['page_content']}"
                    "</div>",
                    unsafe_allow_html=True,
                )

        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)
