import boto3 
from botocore.exceptions import ClientError
import base64
import json
from llama_index.llms.bedrock import Bedrock
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.bedrock import BedrockEmbedding
import os
from llama_index.core.settings import Settings
from dotenv import load_dotenv
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext
import pymongo
from llama_index.core.schema import TextNode

load_dotenv()

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

llm = Bedrock(
    model=model_id,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("REGION_NAME"),
)

embed_model = BedrockEmbedding(
    model="amazon.titan-embed-g1-text-02",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("REGION_NAME"),
)

Settings.llm=llm
Settings.embed_model=embed_model

def get_mongo_client(mongo_uri):
  try:
    client = pymongo.MongoClient(mongo_uri)
    print("Connection to MongoDB successful")
    return client
  except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")
    return None

DB_NAME="compound-genai"
COLLECTION_NAME="image-embeddings"

mongo_uri = os.getenv("MONGODB_URI")

mongo_client = get_mongo_client(mongo_uri)

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})

client = boto3.client("bedrock-runtime", region_name="us-west-2")

def parseImageFile(imageFilepath):
    with open(imageFilepath, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
    return base64_string

def prepareMessagePrompt(prompt, base64_string):
    message_list = [
        {
            "role": 'user',
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_string
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    return message_list



def fetchLLMResponse(message_list, model_id):
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "messages": message_list,
                "max_tokens": 2048,
                "temperature": 0,
                "top_p": 1,
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        return json.loads(response['body'].read())
    except ClientError as e:
        print(f"Error invoking model: {e}")
        return None

def get_image_analysis_prompt():
    return """
            Analyze the provided image and extract key information to create a basic product listing. Focus on the following 3 aspects:

            1. Product Type
            2. Main Color
            3. Key Feature
            4. Product Size

            Provide the information in a structured JSON format. If any information is not discernible from the image, use "Not visible" as the value.

            Your response should be in the following JSON format:

            {
                "product_type": "Brief description of what the product is",
                "main_color": "Primary color of the product",
                "key_feature": "One notable feature or characteristic of the product"
                "dimensions": "Appropriate size like length and breadth of the product"
            }

            Analyze the image and provide only the JSON output. Do not include any additional text or explanation outside the JSON structure.
        """


def analyze_image_with_bedrock(image_path):
    base64_string = parseImageFile(image_path)
    prompt = get_image_analysis_prompt()

    messages = [
        ChatMessage(role="system", content="You are an AI named Amelia trained to analyze images. Your job is to generates descriptions, tags, and category suggestions"),
        ChatMessage(role="user", content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_string
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ])
    ]

    resp = llm.chat(messages)
    print(resp)

    return resp

def get_social_media_template_prompt():
    return """
    Given a response about a product or item, create two different templates:

    1. Amazon Template:
       Extract or creatively infer the following information:
       - Product Name: [Name of the product]
       - Estimated Price: $[Price] (use a reasonable estimate based on the description)
       - Number of Items Sold: [Number] (use a plausible number based on the product's appeal)

       Format the extracted information as follows:
       ```
       Product Name: [Extracted name]
       Estimated Price: $[Extracted price]
       Number of Items Sold: [Extracted number]
       ```

    2. Instagram Template:
       Extract or creatively infer the following information:
       - Item Name: [Name of the item]
       - Genre: [Category or type of the item]
       - Like Meter Rating: [Number] out of 10 (based on the item's appeal)

       Format the extracted information as follows:
       ```
       Item Name: [Extracted name]
       Genre: [Extracted genre]
       Like Meter Rating: [Extracted rating]/10
       ```

    For both templates, use the information provided in the original response. If specific details are not explicitly mentioned, use creative inference based on the description to fill in the required fields. Ensure that the inferred information is plausible and consistent with the tone and content of the original response.
    """

def analyze_response_for_templates(response):
    prompt = get_social_media_template_prompt()

    if hasattr(response, 'content'):
        response_text = response.content
    elif isinstance(response, str):
        response_text = response
    else:
        raise TypeError("response must be either a string or a ChatResponse object")

    messages = [
        ChatMessage(role="system", content="You are an AI trained to extract information from product descriptions and create social media templates."),
        ChatMessage(role="user", content=prompt + "\n\nHere's the product description:\n" + response_text)
    ]

    resp = llm.chat(messages)
    # print(resp)

    return resp

def generate_embeddings(response):
    embeddings = embed_model.get_text_embedding(str(response))
    print(embeddings)
    return embeddings

def store_in_vector_db(embedding):

    node = TextNode(text=str(response))
    
    node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
    node.embedding = node_embedding

    vector_store = MongoDBAtlasVectorSearch(
        mongo_client, 
        db_name=DB_NAME, 
        collection_name=COLLECTION_NAME, 
        index_name="vector_index"
    )

    vector_store.add([node])

    index = VectorStoreIndex.from_vector_store(vector_store)

    return index

def query_vector_db(index, queries):
    response_list = []
    for query in queries:
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        print("Returning response:")
        print("-"*10)
        print(response)
        print("*"*50)
        print("\n")
        response_list.append(str(response))

    return response_list

if __name__ == "__main__":
    image_paths = ["coffee_mug.png", "llama.jpg"]
    for image_path in image_paths:
        response = analyze_image_with_bedrock(image_path)
        index = store_in_vector_db(response)

    queries = [
        "Recommend a coffee mug and describe it.",
        "Recommend a fun pet animal and describe it.",
    ]
    response_list = query_vector_db(index, queries)

    # template_types = ["amazon", "instagram"]

    # for template_type in template_types:
    #     response_template = analyze_template(template_type, response)

    for response in response_list:
        result = analyze_response_for_templates(response)
        print(result)
        print("*"*50)
