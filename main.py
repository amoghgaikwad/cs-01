import streamlit as st
from streamlit_chat import message
from textractcaller.t_call import call_textract, Textract_Features
import boto3
import botocore
import json
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import botocore.exceptions

from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch

# from sagemaker import get_execution_role
import fitz 
import io
from PIL import Image 

s3_client = boto3.client('s3', region_name='us-east-1')
textract_client = boto3.client('textract', region_name='us-east-1')

client = boto3.client('opensearchserverless')
service = 'aoss'
region = 'us-east-1'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)


if 'S3_BUCKET_NAME' not in os.environ:
    bucket_name = 'textract-eample-ad-cs'
else:
    bucket_name = os.environ['S3_BUCKET_NAME']

if 'OPENSEARCH_HOST' not in os.environ:
    host = 'xxx'
else:
    host = os.environ['OPENSEARCH_HOST']

###############################################################################

SOURCE_BUCKET_NAME = 'raw-dataset-gaiamogh3' # replace with your bucket name
S3_KEY = 'drylab.pdf' # replace with your object key
# FILE_PATH = "C:\\Madhavi\\Workshops\\CS01-GenAI-Image-Textract\\downloads\\sample.pdf" #local path where the file is stored
FILE_PATH = "sample1.pdf" #local path where the file is stored
output_dir = "extracted_images/"
 
s3 = boto3.resource('s3')
oss_client = boto3.client("opensearchserverless")
rekognition_client = boto3.client('rekognition')

bedrock_client = boto3.client("bedrock-runtime")

# role = get_execution_role()

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def upload_file(file):
    try:
        foldername = file.name.split('.')[0]
        s3_client.upload_fileobj(
            Fileobj=file,
            Bucket=bucket_name,
            Key=(file.name)
        )
        return True
    except Exception as e:
        print(e)
        return False

def query(payload):
    response = get_rag_chat_response(payload['inputs']['text'], st.session_state['memory'], os_client)
    return response

def get_text_input():
    input_text = st.sidebar.text_input("Human: ", "", key="input")
    return input_text

def clear_message():
    st.sidebar.empty()
    del st.session_state.past[:]
    del st.session_state.generated[:]

def get_rag_chat_response(input_text, memory, index):  # chat client function
    llm = get_llm()
    # conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(),
    #                                                                     memory=memory)

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(),
                                                                        memory=memory)

    chat_response = conversation_with_retrieval(
        {"question": "\n\nHuman:Explain the details in " + input_text + "\n\nAssistant:"})  # pass the user message, history, and knowledge to the model
    # print(chat_response)
    return chat_response['answer']

def get_llm():
    model_kwargs = {  # Claude-v2
        "max_tokens_to_sample": 1024,
        "temperature": 0.1,
        "top_p": 0.9
    }

    llm = Bedrock(
        model_id="anthropic.claude-v2",  # set the foundation model
        model_kwargs=model_kwargs)  # configure the properties for Claude
    return llm

def get_memory():  # create memory for this chat session
    memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                            return_messages=True)  # Maintains a history of previous messages
    return memory

def downloadPDF():
    try:
        s3.Bucket(bucket_name).download_file(S3_KEY, FILE_PATH)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

def extractImagesFromPdf():
    min_width = 500
    min_height = 500
    output_format = 'png'
   # open the file 
    pdf_file = fitz.open(FILE_PATH) 
      
    # iterate over PDF pages 
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)

        # Iterate over the images on the page
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            # Extract the image bytes
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))

            if image.width >= min_width and image.height >= min_height:
                if not os.path.exists(output_dir): 
                    os.makedirs(output_dir)
                file_name = f"image{page_index + 1}_{image_index}.{output_format}"
                image.save(
                    open(os.path.join(output_dir, file_name), "wb"),
                    format=output_format.upper())

                s3_client.upload_file(
                    Filename=f"{output_dir}\\{file_name}",
                    Bucket=bucket_name,
                    Key=file_name
                )

            else:
                print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")

    pdf_file.close()            

def createEncryptionPolicy(client):
    """Creates an encryption policy that matches all collections beginning with pdf-"""
    try:
        response = client.create_security_policy(
            description='Encryption policy for pdf-images collections',
            name='pdf-image-policy',
            policy="""
                {
                    \"Rules\":[
                        {
                            \"ResourceType\":\"collection\",
                            \"Resource\":[
                                \"collection\/pdf-*\"
                            ]
                        }
                    ],
                    \"AWSOwnedKey\":true
                }
                """,
            type='encryption'
        )
        print('\nEncryption policy created:')
        print(response)
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'ConflictException':
            print(
                '[ConflictException] The policy name or rules conflict with an existing policy. Skipping Creation')
            pass
        else:
            raise error
            
def createAccessPolicy(client):
    """Creates a data access policy that matches all collections beginning with pdf-"""
    try:
        response = client.create_access_policy(
            description="Data access policy pdf collections",
            name="pdf-policy-2",
            policy="""
                [{
                    \"Rules\":[
                        {
                            \"Resource\":[
                                \"index\/pdf-*\/*\"
                            ],
                            \"Permission\":[
                                \"aoss:CreateIndex\",
                                \"aoss:DeleteIndex\",
                                \"aoss:UpdateIndex\",
                                \"aoss:DescribeIndex\",
                                \"aoss:ReadDocument\",
                                \"aoss:WriteDocument\"
                            ],
                            \"ResourceType\": \"index\"
                        },
                        {
                            \"Resource\":[
                                \"collection\/pdf-*\"
                            ],
                            \"Permission\":[
                                \"aoss:CreateCollectionItems\"
                            ],
                            \"ResourceType\": \"collection\"
                        }
                    ],
                    \"Principal\":[
                        \"arn:aws:iam::403618562126:role/service-role\/AmazonSageMaker-ExecutionRole-20210423T101471\"
                    ]
                }]
                """,
            type="data",
        )
        print("\nAccess policy created:")
        print(response)
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "ConflictException":
            print("[ConflictException] An access policy with this name already exists. skipping creation.")
            pass
        else:
            raise error


def createNetworkPolicy(client):
    """Creates a network policy that matches all collections beginning with pdf-"""
    try:
        response = client.create_security_policy(
            description="Network policy for TV collections",
            name="pdf-nw-policy",
            policy="""
                [{
                    \"Description\":\"Public access for PDF OS collection\",
                    \"Rules\":[
                        {
                            \"ResourceType\":\"dashboard\",
                            \"Resource\":[\"collection\/pdf-*\"]
                        },
                        {
                            \"ResourceType\":\"collection\",
                            \"Resource\":[\"collection\/pdf-*\"]
                        }
                    ],
                    \"AllowFromPublic\":true
                }]
                """,
            type="network",
        )
        print("\nNetwork policy created:")
        print(response)
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "ConflictException":
            print("[ConflictException] A network policy with this name already exists. skipping creation")
            pass
        else:
            raise error


def createCollection(client):
    """Creates a OSS collection"""
    try:
        response = client.create_collection(name="pdf-images", type="VECTORSEARCH")
        return response
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "ConflictException":
            print(
                "[ConflictException] A collection with this name already exists. skipping creation"
            )
            return True
        else:
            raise error


def waitForCollectionCreation(client):
    """Waits for the collection to become active"""
    response = client.batch_get_collection(names=["pdf-images"])
    print(response)

    while (response["collectionDetails"][0]["status"]) == "CREATING":
        print("Creating collection...")
        time.sleep(30)
        response = client.batch_get_collection(names=["pdf-images"])
    print("\nCollection successfully created:")
    print(response["collectionDetails"])
    
    # Extract the collection endpoint from the response
    host = response["collectionDetails"][0]["collectionEndpoint"]
    return host


# extract text from the images
def detect_text(image):
    summary = []
    try:
        image = Image.open(image)

        stream = io.BytesIO()
        image.save(stream,format="PNG")
        image_binary = stream.getvalue()

        response = rekognition_client.detect_text(Image={'Bytes':image_binary})
        text_detections = response["TextDetections"]
        for text in text_detections:
            if 'LINE' in text['Type']:
                summary.append(text['DetectedText'])
    except ClientError:
        print("Couldn't detect text in %s.", image)
        raise
    else:
        return ','.join(summary)


def detect_labels(photo):
    try:  
        with open(photo, 'rb') as image:
            labels = rekognition_client.detect_labels(Image={'Bytes': image.read()}, MaxLabels=10)
        labels = [label['Name'] for label in labels['Labels']]
    except ClientError:
        print("Couldn't detect text in %s.", image)
        raise
    else:
        return ','.join(labels)


def build_index_mapping():
    index_mapping = {
        "settings": {"index": {"knn": True}},  # Enable k-NN search for this index
        "mappings": {
            "properties": {
                "embedding": {  # k-NN vector field
                    "type": "knn_vector",
                    "dimension": 1536,  # Dimension of the vector
                    "similarity": "cosine",
                },
                "file_name": {"type": "text"},
                "text": {"type": "text"},
                "s3_key": {"type": "text"},
            }
        },
    }
    # st.write(index_mapping)
    return index_mapping


def indexData():
    """Create an index and add some sample data"""

    # Create index
    try:
        mapping = build_index_mapping()
        if(os_client.indices.exists(index="pdf-images") == False):
            response = os_client.indices.create("pdf-images", mapping)
            print("\nCreating index:")
            print(response)
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "RequestError":
            print("[ConflictException] A index with this name already exists. skipping creation")
            pass
        else:
            raise error

    # Add documents to the index.
    # iterate over all images in the directory
    for images in os.listdir(output_dir):
        if images.endswith(".png"):
            text = detect_text(f"{output_dir}{images}")
            labels = detect_labels(f"{output_dir}{images}")
            extracted_text = {"inputText": labels + text}
            body = json.dumps(extracted_text)
            st.write(extracted_text)

            # call the bedrock client to embedd the text
            br_response = bedrock_client.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json",
            )

            br_response_body = json.loads(br_response.get("body").read())
            embedding = br_response_body.get("embedding")

            # upload_image = upload_file(
            #     f"{output_dir}/{images}", bucket_name, f"op/{images}"
            # )
            
            document = {
                "embedding": embedding,
                "file_name": images,
                "text": labels + text,
                "s3_key": f"{bucket_name}/op/{images}",
            }
            index_response = os_client.index(
                index="pdf-images",
                body=document,
            )
            print("\nDocument added:")
            print(index_response)

st.title('Upload and Display File')


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    if upload_file(uploaded_file):
        st.success("File Uploaded")

        if host == 'xxx':
            createEncryptionPolicy(oss_client)
            createNetworkPolicy(oss_client)
            createAccessPolicy(oss_client)
            createCollection(oss_client)
            host = waitForCollectionCreation(oss_client)
            st.write(host)
        # Build the OpenSearch client
        os_client = OpenSearch(
            # hosts=[{"host": "5fso0en8s31bts1p2so5.us-east-1.aoss.amazonaws.com", "port": 443}],
            hosts=[{"host": host, "port": 443}],
            # opensearch_url=opensearch_endpoint,
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        # st.write(opensearch_endpoint)
        downloadPDF()
        extractImagesFromPdf()

        indexData()

st.sidebar.header('Chat about the uploaded file here')
if 'memory' not in st.session_state:
    st.session_state['memory'] = get_memory()

with st.sidebar.form(key='widget', clear_on_submit=True):
    user_input = get_text_input()
    if user_input:
        output = query({
            "inputs": {
                "past_user_inputs": st.session_state.past,
                "generated_responses": st.session_state.generated,
                "text": user_input,
            }, "parameters": {"repetition_penalty": 1.33},
        })

        st.session_state.past.append('Human: ' + user_input)
        st.session_state.generated.append('Assistant: ' + output)
    if st.sidebar.button("Clear messages"):
        clear_message()

chat_placeholder = st.sidebar.empty()
with chat_placeholder.container():
    if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
    else:
        st.error("Error uploading file")



