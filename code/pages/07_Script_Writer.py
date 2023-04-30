from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from urllib.error import URLError
import pandas as pd
from utilities import utils, translator
import os
import pdb
import nltk

# nltk.download('punkt') # TODO: add code to know when this needs to be done

from nltk.tokenize import sent_tokenize

df = utils.initialize(engine='davinci')

placeholder = None

@st.cache(suppress_st_warning=True)
def get_languages():
    return translator.get_available_languages()



try:

    default_prompt = "" 
    default_question = "" 
    default_answer = ""

    # These are sometimes not set?
    if 'question' not in st.session_state:
        st.session_state['question'] = default_question
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = os.getenv("QUESTION_PROMPT", "Please reply to the question using only the information present in the text above. If you can't find it, reply 'Not in the text'.\nQuestion: _QUESTION_\nAnswer:").replace(r'\n', '\n')
        print("In here!")
    if 'response' not in st.session_state:
        st.session_state['response'] = {
            "choices" :[{
                "text" : default_answer
            }]
        }    
    if 'limit_response' not in st.session_state:
        st.session_state['limit_response'] = True
    if 'full_prompt' not in st.session_state:
        st.session_state['full_prompt'] = ""
    # if 'st'

    # Set page layout to wide screen and menu item
    menu_items = {
	'Get help': None,
	'Report a bug': None,
	'About': '''
	 ## Embeddings App
	 Embedding testing application.
	'''
    }
    st.set_page_config(layout="wide", menu_items=menu_items)
    # Get available languages for translation
    available_languages = get_languages()

    # def break_into_sentences():
    #     global placeholder
    #     curr_text = st.session_state.input_text
    #     split_text = sent_tokenize(curr_text)
    #     # print(st.session_state) # checking state variables - some aren't set??
    #     for sentence in split_text:
    #         st.session_state['full_prompt'], st.session_state['response'] = utils.get_semantic_answer(df, sentence, st.session_state['prompt'] ,model=model, engine='davinci', limit_response=st.session_state['limit_response'], tokens_response=400, temperature=0.1)
    #         st.write(st.session_state['response']['choices'][0]['text'])
    #         st.write(f"Q: {st.session_state.input_text}")  
    #         st.write(f"{st.session_state['response']['choices'][0]['text']}")
    #         with st.expander("Question and Answer Context"):
    #             st.text(st.session_state['full_prompt'].replace('$', '\$'))

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image(os.path.join('images','microsoft.png'))

    col1, col2, col3 = st.columns([2,2,2])
    with col3:
        with st.expander("Settings"):
            model = st.selectbox(
                "OpenAI GPT-3 Model",
                (os.environ['OPENAI_ENGINES'].split(','))
            )
            st.text_area("Prompt",height=100, key='prompt')
            st.tokens_response = st.slider("Tokens response length", 100, 500, 400)
            st.temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
            st.selectbox("Language", [None] + list(available_languages.keys()), key='translation_language')
    
    question = st.text_area("OpenAI Semantic Answer", default_question)

    if question != '':
        if question != st.session_state['question']:
            st.session_state['question'] = question
            # split_text = sent_tokenize(question)
            key_phrases = utils.key_phrase_extraction([question])
            for elem in key_phrases:
                # TODO: decide if want to concatenate the key phrases of the sentence and then get the embedding, or get one for each key phrase (current implementation). I think the first is better.
                for phrase in elem:
                    st.session_state['full_prompt'], st.session_state['response'] = utils.get_semantic_answer(df, phrase, st.session_state['prompt'] ,model=model, engine='davinci', limit_response=st.session_state['limit_response'], tokens_response=400, temperature=0.1)
                    
                    st.write(f"Q: {phrase}")  
                    st.write(st.session_state['response']['choices'][0]['text'])
                    with st.expander("Question and Answer Context"):
                        st.text(st.session_state['full_prompt'].replace('$', '\$')) 
        else:
            st.write(f"Q: {st.session_state['question']}")  
            st.write(f"{st.session_state['response']['choices'][0]['text']}")
            with st.expander("Question and Answer Context"):
                st.text(st.session_state['full_prompt'].encode().decode())

    if st.session_state['translation_language'] is not None:
        st.write(f"Translation to other languages, 翻译成其他语言, النص باللغة العربية")
        st.write(f"{translator.translate(st.session_state['response']['choices'][0]['text'], available_languages[st.session_state['translation_language']])}")		
		
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )