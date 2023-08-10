"""WORKFLOW
1) Convert image to prompt
2) Save prompt in the form of a dict and a list
3) save all the prompts in a vector store
4) Take User input
5) compare user input semantically with the vector store
6) engineer output to give more than one image if semantically close
7) Output image by comparing it to a dictionary 
8) create a simple frontend 
f
"""


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


def create_prompts():
    #Create a dummy list of Tags that the RAM model will return
    list = []
    list.append('alcohol | beer | beer bottle | beverage | bottle | can | drink | girl | liquid | party | pink | sip | soda | woman')
    list.append('arcade machine | blow | club | dark | person | laugh | man | nightclub | powder | puff | purple | room | smoke | woman')
    list.append('family | group photo | person | mountain | peak | photo | picture | pose | stand')
    list.append('club | pillar | crowded | girl | person | man | nightclub | party | pose | smile | stand | woman')

    #creat LLM such that it will convert the tags into prompts and store it in a text file.

    template = """an image has been described though tags, these are the tags used: 
    {tags} 
    string these tags together into a full english description of the image, do not make up any details not described by those tags. Do not try to describe how the image feel"""
    prompt = PromptTemplate(input_variables=['tags'], template=template)

    chain = LLMChain(llm=llm,
                    prompt=prompt)


    with open("prompts.txt", 'w') as file:
        for i in list:
            x = chain(i)
            file.write(x['text'] + "\n\n")


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
    vectordb = FAISS.load_local("embeddings/prompts/faiss_index", embedding)
    template =  """Multiple images have been described through tags below, Your role is to identify the most relevant image descriptions based on the user's input query. Below are descriptions of multiple images:
                {context}
                The user will provide an image description query, and your task is to find one or more matching descriptions from the provided context. If there are no matching descriptions, respond with "There are no matching descriptions."
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

search()