from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import os
from flask import Flask, request, jsonify,render_template,Response
from werkzeug.utils import secure_filename
from os import path

BASE_URL = os.environ.get('OPENAI_API_BASE')
API_KEY = os.environ.get('OPENAI_API_KEY')
Version = os.environ.get('OPENAI_API_VERSION')
DEPLOYMENT_NAME = "chatgpt0301"

app = Flask(__name__)
#print("initial embedding...")
#setup embedding db and build vectorstores
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

db_name='qa_db'
#vectorstore = None
embeddings = OpenAIEmbeddings(
        deployment="embedding"
    )
@app.route('/initdb', methods=['POST'])
def InitialDB():
    global db_name
    data = request.get_json()
    db_name = data['dbname']
    if(db_name==None or db_name==""):
        return
    #global vectordb
    with open('state_of_the_union.txt') as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([state_of_the_union])
    
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    for i in range(0,len(texts)):
        vectorstore.add_texts([texts[i].page_content])
    vectorstore.persist()
    #vectorstore=None
    return jsonify({'process': "done!"})

#InitialDB()
#print("initial llm and chains...")
#vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

@app.route('/loaddb', methods=['POST'])
def LoadDB():
    global vectorstore
    global db_name
    global qa
    data = request.get_json()
    db_name = data['dbname']
    print("change and load db to "+db_name+"...")
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    vectorstore.persist()
    #qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    return jsonify({'process': "done!"})


@app.route('/embedding', methods=['POST'])
def Embedding():
    global vectorstore
    global db_name
    global qa
    data = request.get_json()
    db_name = data['dbname']
    text = data['text']
    print("embedding and load db to "+db_name+"...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([text])
    
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    for i in range(0,len(texts)):
        vectorstore.add_texts([texts[i].page_content])
    vectorstore.persist()
    #qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    return jsonify({'process': "done!"})

    
from langchain.callbacks.base import BaseCallbackHandler

  



@app.route('/answer', methods=['POST'])
def answer():
    print("get answer...")
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    data = request.get_json()
    question = data['question']
    histories = data['histories']
    his=[]
    

    llm = AzureChatOpenAI(
        temperature=0,
        model_name="gpt-35-turbo",
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name="chatgpt0301",
        openai_api_key=API_KEY,
        openai_api_type = "azure",
        verbose=True,
    )

    # define two LLM models from OpenAI    
    streaming_llm = AzureChatOpenAI(
        temperature=0.2,
        model_name="gpt-35-turbo",
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name="chatgpt0301",
        openai_api_key=API_KEY,
        openai_api_type = "azure",
        streaming=False,
        verbose=False
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for i in histories:
        memory.save_context({"input": i["human"]}, {"ouput": i["AI"]})
    
    #qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    qa = ConversationalRetrievalChain.from_llm(streaming_llm, vectorstore.as_retriever(),memory=memory)
    r = qa({"question": question})
    return jsonify({'answer': r['answer']})


@app.route('/answer2', methods=['POST'])
def answer2():
    print("get answer...")
    data = request.get_json()
    question = data['question']
    histories = data['histories']
    his=[]
    for i in histories:
        his.append((i["human"],i["AI"]))

    streaming_llm = AzureChatOpenAI(
        temperature=0.2,
        model_name="gpt-35-turbo",
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name="chatgpt0301",
        openai_api_key=API_KEY,
        openai_api_type = "azure",
        streaming=True,
        #callback_manager=CallbackManager([MyCustomHandler()]),
        verbose=True
    )
#qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    qa = ConversationalRetrievalChain.from_llm(streaming_llm, vectorstore.as_retriever())
    r = qa({"question": question,"chat_history":his })
    return jsonify({'answer': r['answer']})

@app.route('/db', methods=['GET'])
def db():
   return jsonify({'db_name': db_name})

@app.route('/')
def index():
    return render_template("bot.html")

@app.route('/2')
def index2():
    return render_template("bot2.html")

ALLOWED_EXTENSIONS = set(['txt', 'pdf'])
UPLOAD_FOLDER = './upload'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.pdf']
app.config['UPLOAD_PATH'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#os.chmod(UPLOAD_FOLDER, 0o644)
@app.route('/uploadfile', methods=['POST', 'GET'])
def do_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # filename = file.filename
            a=file.read()
            
            
            # Detect the encoding of the file
            import chardet
            encoding = chardet.detect(a)['encoding']
            print(encoding)
            # Decode the bytes using the correct encoding
            contents = a.decode(encoding)
            file.seek(0)
            file.save(filename)
            #print(contents)
            loadFile(contents,filename,encoding)
            

    return render_template('bot.html')

def loadFile(file,filename,encoding):
    global db_name
    db_name=filename.replace(".","_")
    
    print("embedding and load db to "+db_name+"...")
    with open(filename, 'r', encoding = encoding) as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([state_of_the_union])
    
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    for i in range(0,len(texts)):
        vectorstore.add_texts([texts[i].page_content])
    
    
    vectorstore.persist()
    

if __name__ == '__main__':
    app.run(debug=True)