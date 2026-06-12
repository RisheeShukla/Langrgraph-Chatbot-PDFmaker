import streamlit as st
import requests
import PIL
from io import BytesIO
from PIL import Image as PILImage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from backend import workflow
from reportlab.platypus import SimpleDocTemplate, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

from reportlab.lib.pagesizes import A4
from langchain_core.messages import HumanMessage
import uuid
from dotenv import load_dotenv
load_dotenv()
import os
HUGGING_FACE_API_TOKEN=os.getenv('HUGGING_FACE_API_TOKEN')
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    task="text-generation",
     huggingfacehub_api_token=HUGGING_FACE_API_TOKEN,
    temperature=2,
    max_new_tokens=100
)


model=ChatHuggingFace(llm=llm)
def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id
##################################################################UTILITY FUNCTIONS########
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
if 'thread_id'  not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=[]

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]
def add_thread(thread_id):
    if thread_id not in st.session_state:
        st.session_state['chat_threads'].append(thread_id)




    ##################SIDEBAR UI####################################
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    st.sidebar.text(thread_id)

def generate_flux_image(prompt, filename="flux_image.png"):
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}","Accept": "image/png"}  # Important: tell HF to return image bytes!}
    FLUX_ENDPOINT = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    data = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 30,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512
        }
    }
    
    response = requests.post(FLUX_ENDPOINT, headers=headers, json=data)
    content_type = response.headers.get("content-type", "")
    if "image" not in content_type:
        raise ValueError(f"Flux API returned non-image response: {response.text}")

    img = PILImage.open(BytesIO(response.content))

    img.save(filename, format="PNG")

    return filename
def pdf_maker(user_input,ai_response):
    doc = SimpleDocTemplate("chat_output.pdf", pagesize=A4)
    elements = []

    input1=model.invoke([HumanMessage(content=f"Create an unique image description for: {user_input}")]).content
    img_path1 = generate_flux_image(input1, "flux_image_1.png")
    elements.append(Image(img_path1, width=200, height=200))
    elements.append(Spacer(1, 20))
    input2=model.invoke([HumanMessage(content=f"Create an unique image description for: {user_input} keeping in mind of not repeating this result: {input1}")]).content
    img_path2 = generate_flux_image(input2, "flux_image_2.png")
    elements.append(Image(img_path2, width=200, height=200))
    elements.append(Spacer(1, 20))
    input3=model.invoke([HumanMessage(content=f"Create an unique image description for: {user_input} keeping in mind of not repeating the result:{input2}")]).content
    img_path3 = generate_flux_image(input3, "flux_image_3.png")
    elements.append(Image(img_path3, width=200, height=200))
    elements.append(Spacer(1, 20))

    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph

    styles = getSampleStyleSheet()
    user_style = styles["Normal"]
    ai_style = styles["Normal"]

    user_paragraph = Paragraph(f"<b>User:</b> {user_input}", user_style)
    ai_paragraph = Paragraph(f"<b>AI:</b> {ai_response}", ai_style)

    elements.append(user_paragraph)
    elements.append(Spacer(1, 20))
    elements.append(ai_paragraph)

    doc.build(elements)







######################### MAIN UI #######################################
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input=st.chat_input('Type here:')
if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    with st.chat_message('assistant'):
        ai=st.write_stream(
            message_chunk.content for message_chunk,metadata in workflow.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
                
            )
        )
        st.session_state['message_history'].append({'role':'assistant','content':ai})
        pdf_maker(user_input,ai)
if st.session_state['message_history']:
    with open("chat_output.pdf", "rb") as f:
        st.download_button("Download PDF", f, file_name="file.pdf")

       
##########################################################################################################################3        
    