import streamlit as st
from streamlit_chat import message
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
import boto3
import botocore
import json
import os
import base64
import fitz
import io
from PIL import Image
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
from botocore.exceptions import ClientError
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch

# from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch


# globals
tmp_dir = "extracted_images"

# Initialize Services
# - S3 Storage
# - Rekognition client
# - OpenSearch Vector Store
# - Amazon Bedrock Embeddings
# - Amazon Bedrock LLM

s3_client = boto3.client("s3", region_name="us-east-1")
aoss_client = boto3.client("opensearchserverless")
s3_resource = boto3.resource("s3")
rekognition_client = boto3.client("rekognition")
bedrock_client = boto3.client("bedrock-runtime")

# OpenSearchVectorSearch Authentication
service = "aoss"
region = "us-east-1"
credentials = boto3.Session().get_credentials()
awsauth = AWSV4SignerAuth(credentials, region, service)

if "S3_BUCKET_NAME" not in os.environ:
    bucket_name = "raw-dataset-gaiamogh3"
else:
    bucket_name = os.environ["S3_BUCKET_NAME"]

if "OPENSEARCH_HOST" not in os.environ:
    list_collections_response = aoss_client.list_collections()
    collection_id = list_collections_response.get("collectionSummaries")[0].get("id")
    host = f"{collection_id}.{region}.aoss.amazonaws.com"
else:
    host = os.environ["OPENSEARCH_HOST"]

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_bedrock_embeddings(boto3_session: boto3.Session) -> BedrockEmbeddings:
    bedrock_client = boto3_session.client("bedrock-runtime", region_name="us-east-1")
    embeddings_model_id = "amazon.titan-embed-text-v1"
    return BedrockEmbeddings(client=bedrock_client, model_id=embeddings_model_id)


def upload_file(file):
    try:
        dir = file.name.split(".")[0]
        s3_client.upload_fileobj(
            Fileobj=file, Bucket=bucket_name, Key=(dir + "/" + file.name)
        )
        return True
    except Exception as e:
        print(e)
        return False


def download_pdf(file):
    try:
        foldername = file.name.split(".")[0]
        s3_resource.Bucket(bucket_name).download_file(f"{foldername}/{file.name}", file.name)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


# conversation history using DynamoDB
def build_dynamodb_table():
    dynamodb = boto3.client("dynamodb")
    table_name = "SessionTable2"
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
    except dynamodb.exceptions.ResourceInUseException:
        pass

    key = {
        "pk": "session_id::0",
        "sk": "langchain_history",
    }
    message_history = DynamoDBChatMessageHistory(
        table_name=table_name,
        session_id="0",
        key=key
    )

    return table_name, message_history


def s3_image_upload(tmp_dir, dest_s3_bucket, file_extension):
    ''' upload images to s3 with the filename as prefix '''
    tmp_dir = os.path.join(os.path.dirname(__file__), tmp_dir)

    for filename in os.listdir(tmp_dir):
        if filename.endswith(file_extension):
            pdf_filename = filename.split("_")[0]
            s3prefix = f"{pdf_filename}/{filename}"
            print(f"storing in this prefix - {s3prefix}")
            s3_client.upload_file(
                os.path.join(tmp_dir, filename), dest_s3_bucket, s3prefix
            )

    # delete images in tmp directory
    for filename in os.listdir(tmp_dir):
        if filename.endswith(file_extension):
            os.remove(os.path.join(tmp_dir, filename))


def query(payload):
    response = get_rag_chat_response(payload['inputs']['text'], st.session_state['memory'], st.session_state['index'])
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
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(
        llm, index.as_retriever(), memory=memory
    )

    chat_response = conversation_with_retrieval(
        {
            "question": "\n\nHuman:Explain the details in "
                        + input_text
                        + "\n\nAssistant:"
        }
    )  # pass the user message, history, and knowledge to the model
    return chat_response["answer"]


def get_llm():
    model_kwargs = {  # Claude-v2
        "max_tokens_to_sample": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
    }

    llm = Bedrock(
        model_id="anthropic.claude-v2",  # set the foundation model
        model_kwargs=model_kwargs,
    )  # configure the properties for Claude
    return llm


def get_memory():
    # create memory for this chat session
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True
    )  # Maintains a history of previous messages
    return memory


def extract_images_from_pdf(file):
    min_width = 50
    min_height = 50
    output_format = "png"

    # open the file
    pdf_file = fitz.open(file.name)

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
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                file_name = f"{pdf_file.name.split('.')[0]}_img_{page_index + 1}_{image_index}.{output_format}"
                # save images in {output_dir} temporarily
                image.save(
                    open(os.path.join(tmp_dir, file_name), "wb"),
                    format=output_format.upper(),
                )
            else:
                print(
                    f"[-] Skipping image {image_index} on page {page_index} due to its small size."
                )
    # upload images to s3 for processing with rekognition
    s3_image_upload(tmp_dir, bucket_name, output_format)
    pdf_file.close()

    # remove temp file
    if os.path.isfile(file.name):
        os.remove(file.name)

    return True


# extract text from the images
def detect_text(image_path):
    text_summary = []
    image = image_path.split("/")[1]
    pdf_file = image.split("_")[0]
    image_s3 = {"S3Object": {"Bucket": bucket_name, "Name": f"{pdf_file}/{image}"}}
    print(image_s3)

    try:
        response = rekognition_client.detect_text(Image=image_s3)
        text_detections = response["TextDetections"]
        for text in text_detections:
            if "LINE" in text["Type"]:
                text_summary.append(text["DetectedText"])
    except botocore.exceptions.ClientError as error:
        print(f"Couldn't detect text in - {image_path} due to {error}")
        raise
    else:
        return ",".join(text_summary)


def detect_labels(image_path):
    image = image_path.split("/")[1]
    pdf_file = image.split("_")[0]
    image_s3 = {"S3Object": {"Bucket": bucket_name, "Name": f"{pdf_file}/{image}"}}
    try:
        labels = rekognition_client.detect_labels(Image=image_s3, MaxLabels=10)
        labels = [label["Name"] for label in labels["Labels"]]
    except botocore.exceptions.ClientError as error:
        print(f"Couldn't detect text in - {image_path} due to {error}")
        raise
    else:
        return ",".join(labels)


def index_data():
    """Create an AOSS index and add some sample data"""
    index_name = "st-imgs"
    bedrock_embeddings = get_bedrock_embeddings(boto3.Session(region_name="us-east-1"))

    # Opensearch collection initiate
    opensearch_vector_search = OpenSearchVectorSearch(
        opensearch_url=host,
        index_name=index_name,
        embedding_function=bedrock_embeddings,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    # Add documents to the index.
    # iterate over all images in the directory
    for images in os.listdir(tmp_dir):
        if images.endswith(".png"):
            text = detect_text(f"{tmp_dir}/{images}")
            labels = detect_labels(f"{tmp_dir}/{images}")
            extracted_text = {"inputText": labels + text}
            texts = [labels + text]
            body = json.dumps(extracted_text)
            st.write(
                f"Extracted Insights from {images.split('_')[0]}:\n\n  {images}:{extracted_text}"
            )
            print(
                f"Extracted Insights from {images.split('_')[0]}:\n\n  {images}:{extracted_text}"
            )

            # call the bedrock client to embed the text
            br_response = bedrock_client.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json",
            )

            br_response_body = json.loads(br_response.get("body").read())
            embedding = br_response_body.get("embedding")

            metadata = [
                {
                    "file_name": images,
                    "text": labels + text,
                    "s3_key": f"{bucket_name}/{images.split('_')[0]}/{images}",
                }
            ]

            opensearch_vector_search.from_embeddings(
                opensearch_url=host,
                http_auth=awsauth,
                index_name=index_name,
                engine="nmslib",
                space_type="cosinesimil",
                embedding=bedrock_embeddings,
                metadatas=metadata,
                texts=texts,
                embeddings=[embedding],
                bulk_size=1536,
            )
    return opensearch_vector_search


def display_pdf_content(file_name):
    # Opening file from file path
    foldername = file_name.split(".")[0]
    obj = s3_client.get_object(Bucket=bucket_name, Key=(foldername + "/" + file_name))
    base64_pdf = base64.b64encode(obj["Body"].read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


st.title("Upload and Display File")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    if upload_file(uploaded_file):
        st.success("File Uploaded")
        download_pdf(uploaded_file)
        with st.spinner("Indexing the file to start the chat session....."):
            if "memory" not in st.session_state:
                st.session_state["memory"] = get_memory()
            if "index" not in st.session_state:
                extract_images_from_pdf(uploaded_file)
                st.session_state["index"] = index_data()
            # st.write(st.session_state['index'].vectorstore.as_retriever())
            # st.write(st.session_state['index'].as_retriever())
            display_pdf_content(uploaded_file.name)
            st.sidebar.header("Chat about the uploaded file here")
            with st.sidebar.form(key="widget", clear_on_submit=True):
                user_input = get_text_input()
                if user_input:
                    output = query(
                        {
                            "inputs": {
                                "past_user_inputs": st.session_state.past,
                                "generated_responses": st.session_state.generated,
                                "text": user_input,
                            },
                            "parameters": {"repetition_penalty": 1.33},
                        }
                    )

                    st.session_state.past.append("Human: " + user_input)
                    st.session_state.generated.append("Assistant: " + output)
                    session_table, message_history = build_dynamodb_table()
                    message_history.add_user_message(user_input)
                    message_history.add_ai_message(output)
                if st.sidebar.button("Clear messages"):
                    clear_message()

            chat_placeholder = st.sidebar.empty()
            with chat_placeholder.container():
                if st.session_state["generated"]:
                    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                        message(
                            st.session_state["past"][i],
                            is_user=True,
                            key=str(i) + "_user",
                        )
                        message(st.session_state["generated"][i], key=str(i))
    else:
        st.error("Error uploading file")
