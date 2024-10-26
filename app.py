import os
import openai
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from newspaper import Article

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon=":robot_face:", layout="wide")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not (openai_api_key.startswith("sk-") and len(openai_api_key) == 164):
        st.warning("Please enter your OpenAI API key to continue.", icon="⚠️")
    else:
        st.success("API key is valid.", icon="✅")

with st.container() :
    l, m, r = st.columns((1,3,1))
    with l: st.empty()
    with m: st.empty()
    with r: st.empty()

option = option_menu(
    "Dashboard",
    ["Home", "About Us", "Model"],
    icons=["book", "globe", "tools"],
    menu_icon="book",
    default_index=0,
    styles={
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected" : {"background-color":"#262730"}

    }
)

if 'messages' not in st.session_state:
    st.session_state.message['']

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None #placeholder

elif option == "Home" :
    st.Title("Title")
    st.write("Write Text")

elif option == "About Us" :
    st.Title("About Us")

elif option == "Model" :
    st.title("News Summarizer Tool")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        News_Article = st.text_input("News Article", placeholder="News : ")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary"):



url = ""
article = Article('https://news.bloomberglaw.com/privacy-and-data-security/googles-cookie-pivot-eases-ad-concerns-fuels-privacy-dilemma')
article.download()
article.parse()
user_message = article.text