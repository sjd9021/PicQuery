import openai
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,RetrievalQA, LLMChain
from langchain.document_loaders import TextLoader
import subprocess
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

#init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

llm = AzureChatOpenAI(    
                deployment_name=OPENAI_DEPLOYMENT_NAME,
                    model=OPENAI_MODEL_NAME,
                    openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                    openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0.5
                    )
embedding=OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)
list = []

list.append('alcohol | beer | beer bottle | beverage | bottle | can | drink | girl | liquid | party | pink | sip | soda | woman')
list.append('arcade machine | blow | club | dark | person | laugh | man | nightclub | powder | puff | purple | room | smoke | woman')
list.append('family | group photo | person | mountain | peak | photo | picture | pose | stand')
list.append('club | pillar | crowded | girl | person | man | nightclub | party | pose | smile | stand | woman')
model = "RAM"
images_dir = "images/demo"
image_files = [f"{images_dir}/{file}" for file in sorted(os.listdir(
    images_dir)) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
task = "one image"
print('You selected', model)
print('You selected', task)
master_list={}

def download_checkpoint(url, save_path):
    print("downloading......")
    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        print("Downloaded successfully!")
    else:
        print(f"Failed to download. Status code: {response.status_code}")


model = "RAM"  # Specify the model type
if not os.path.exists('pretrained'):
    os.makedirs('pretrained')

# if model == "RAM":
#     ram_weights_path = 'pretrained/ram_swin_large_14m.pth'
#     if not os.path.exists(ram_weights_path):
#         url = "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth"
#         download_checkpoint(url, ram_weights_path)

def run_inference_once(image_path,i):
    # pretrained_path = "pretrained/ram_swin_large_14m.pth"

    # command = [
    #     "python",
    #     "inference_ram.py",
    #     "--image",
    #     image_path,
    #     "--pretrained",
    #     pretrained_path
    # ]

    # # Run the command and capture the output
    # completed_process = subprocess.run(command, capture_output=True, text=True)

    # # Get the standard output and standard error
    # stdout = completed_process.stdout
    # stderr = completed_process.stderr

    # # Print the output (you can also store it in a file or process it as needed)
    # image_tags_line = [line for line in stdout.split(
    #     '\n') if line.startswith("Image Tags:")][0]
    # image_tags = image_tags_line.replace("Image Tags:", "").strip()
    
    # Print the extracted image tags
    image_tags = list[i]
    # master_list[list[i]] = image_path
    template = """an image has been described though tags, these are the tags used: 
    {tags} 
    string these tags together into a full english description of the image, do not make up any details not described by those tags. Do not try to describe how the image feel"""
    prompt = PromptTemplate(input_variables=['tags'], template=template)

    chain = LLMChain(llm=llm,
                    prompt=prompt)


    x = chain(image_tags)
    master_list[x['text']] = image_path




file_len = len(image_files) 
for i in range(0, file_len):
    image_path = image_files[i]
    run_inference_once(image_path,i)


with open("prompts2.txt", 'w') as file:
    for i in master_list:
        file.write(i + "\n\n")


def create_embedding():
    loader = TextLoader('prompts.txt')
    data = loader.load()
    print(data)
    vectordb = FAISS.from_documents(
    documents=data,
    embedding=embedding
)
    vectordb.save_local("embeddings/prompts/faiss_index")

def search():
    vectordb = FAISS.load_local(r"C:\Users\SJ98023\ImageSearch\embeddings\prompts\faiss_index\index.faiss", embedding)
    template =  """Multiple images have been described through tags below, Your role is to identify the most relevant image descriptions based on the user's input query. Below are descriptions of multiple images:
                {context}
                The user will provide an image description query, and your task is to find one pip or more matching descriptions from the provided context. If there are no matching descriptions, respond with "There are no matching descriptions."
                User Query: {question}
                Helpful Answer:"""
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    qa= RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)
    question = str("Girl in the river")
    result = qa({"query": question})
    print(result['result'])

create_embedding
search()