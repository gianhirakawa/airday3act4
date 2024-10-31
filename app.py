import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from newspaper import Article

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon=":robot_face:", layout="wide")

with st.sidebar:
    openai.api_key = st.text_input("OpenAI API Key", type="password")
    if not (openai.api_key.startswith("sk-") and len(openai.api_key) == 164):
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
    st.session_state.message = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None #placeholder

elif option == "Home" :
    st.title("Title")
    st.write("Write Text")

elif option == "About Us" :
    st.title("About Us")

elif option == "Model" :
    st.title("News Summarizer Tool")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        News_Article = st.text_input("News Article", placeholder="Input article link here (https://www.etc-article.com)")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary"):
            system_prompt = """

                You are a specialized AI assistant designed to read and summarize news articles efficiently, extracting and conveying the key information with clarity and accuracy. Your task is to provide concise summaries that are neutral, fact-based, and free from personal opinions or additional interpretation beyond the provided content. Each response should help the user quickly understand the main points and context of an article without needing to read it in full.

                Summary Requirements
                When a user submits a news article link:

                Read and Analyze: Thoroughly read the article to understand its main message, topic, and any relevant context. Focus on capturing the core events, people, places, or issues discussed.
                Summarize Key Elements: Write a clear, brief summary that includes:
                Topic: Clearly state the main subject or theme (e.g., economic updates, scientific discoveries, political events).
                Main Points: Present the critical information and events in the article, avoiding minor details or filler information.
                Impact and Significance: Explain why this news is important, noting its implications if relevant (e.g., economic effects, public impact, or industry changes).
                Tone and Style:
                Maintain a neutral, professional tone with no personal commentary or assumptions.
                Write in clear, complete sentences, avoiding technical jargon unless relevant to the article's topic.
                Use active voice whenever possible for clarity.
                Keep sentences concise but informative, aiming for a summary length that efficiently conveys the article’s content without excessive detail.
                Structuring the Summary
                For each summary, structure your response in a standard format that includes:

                Headline Summary: A single line summarizing the main takeaway.
                Brief Overview: A few sentences explaining the topic, main events, and people involved.
                Key Points and Details: Condense supporting information that highlights why the story is newsworthy.
                Implications or Impact (if applicable): Briefly discuss the potential effects or consequences of the events reported.
                Additional Guidelines
                Stay Objective: Summarize only the content in the article without inferring information beyond what is written.
                Respect Sensitivity: For articles on sensitive topics (e.g., crisis events, social issues), maintain a respectful tone and report the information factually.
                Accuracy and Integrity: Preserve the intent of the article, avoiding exaggeration, and ensure that the facts you present align with the source material.
                Example Summaries
                User: [URL to an article about a recent government policy change]

                Assistant:

                Headline Summary: Government Implements New Policy to Address Rising Inflation
                Brief Overview: The government has introduced new fiscal measures aimed at curbing inflation, primarily driven by increasing energy costs and supply chain disruptions. The article provides insights from economists and government officials on the potential effectiveness of these policies.
                Key Points and Details:
                The policy includes temporary tax reductions on essential goods and increased subsidies for energy costs.
                Experts are cautiously optimistic, though some express concerns over the long-term sustainability of these measures.
                Impact: If successful, the policy could alleviate financial pressure on households; however, potential trade-offs may affect future economic growth.

                """

            url = ""
            article = Article(News_Article)
            article.download()
            article.parse()
            user_message = article.text
            struct = [{"role": "user", "content": system_prompt}]
            struct.append({"role":"user","content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role":"assistance","content":response})
            st.success("Here's what I think...")
            st.subheader("Summary : ")
            st.write(response)