from datetime import timedelta
import tiktoken
from tiktoken import get_encoding
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (load_data, convert_seconds, generate_prompt_series, 
                          validate_token_threshold)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import css_templates
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
## RETRIEVER
client = WeaviateClient(api_key, url)
logger.info(f"client is live: {client.is_live()}, client is ready: {client.is_ready()}")

## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
## LLM 
model='gpt-3.5-turbo-0613'
llm=GPT_Turbo(model=model)
## ENCODING
encoding=tiktoken.encoding_for_model(model)
## INDEX NAME
class_name = 'Impact_theory_minilm_256'
##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    st.write(css_templates.load_css(), unsafe_allow_html=True)
    
    with st.sidebar:
        #guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        alpha_input=st.slider('alpha parameter',0.00,1.00,0.30)
        retrieval_limit=st.slider('Hybrid search retrieval results', 1,100,10)
        reranker_topk=st.slider('Reranker Top K', 1,50,3)
        temperature_input=st.slider('Temperature of LLM',0.0,2.0,1.0)

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############

            #st.write('Hmmm...this app does not seem to be working yet.  Please check back later.')
            #if guest:
            #    st.write(f'However, it looks like you selected {guest} as a filter.')
            # make hybrid call to weaviate
            properties=['title','guest','summary','content','episode_url','thumbnail_url','length']
            hybrid_response = client.hybrid_search(query,class_name=class_name)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response, query, apply_sigmoid=True,top_k=reranker_topk)
    
            #validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response, 
                                                       question_answering_prompt_series, 
                                                       query=query,
                                                       tokenizer= # variable from ENCODING,
                                                       token_threshold=4000, 
                                                       verbose=True)
            ##############
            #  END CODE  #
            ##############

            # # generate LLM prompt
            prompt = generate_prompt_series(base_prompt=question_answering_prompt_series, query=query, results=valid_response)
            
            # # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                res_box = st.empty()
                chat_container=[]
                
                for resp in llm.get_chat_completion(prompt=prompt,temperature=temperature_input,max_tokens=500,show_response=True,stream=True,):
                    try:
                        with res_box:
                            content=resp.choices[0].delta.content
                            if content:
                                chat_container.append(content)
                                result="".join(chat_container).strip()
                                st.write(f'{result}')
                    except Exception as e:
                        print(e)
                        continue
                
            #     # execute chat call to LLM
            #                  ##############
            #                  # START CODE #
            #                  ##############
            #     

            #                  ##############
            #                  #  END CODE  #
            #                  ##############
            
            # ##############
            # # START CODE #
            # ##############
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length']
                time_string = str(timedelta(seconds=show_length))
            # ##############
            # #  END CODE  #
            # ##############
                with col1:
                    st.write(css_templates.search_result(i=i, 
                                                    url=episode_url,
                                                    episode_num=i,
                                                    title=title,
                                                    content=hit['content'], 
                                                    length=time_string),
                            unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
                    #             unsafe_allow_html=True)
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()