import os
from hashlib import sha256
from opensearchpy import OpenSearch, RequestError
from llama_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import Document, TextNode, NodeWithScore
from llama_index.core import PromptTemplate
from llama_index.core import QueryBundle
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from PyPDF2 import PdfReader
from pydantic import Field

Settings.llm = Ollama(model="qwen:14b")
Settings.embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")


class CustomDirectoryReader(SimpleDirectoryReader):
    def _read_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def load_data(self):
        documents = []
        for file in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file)
            if file.endswith(".pdf"):
                text = self._read_pdf(file_path)
                documents.append(Document(text=text))
            elif file.endswith(".txt") or file.endswith(".md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(text=text))
        return documents

reader = CustomDirectoryReader(input_dir="car/data")
documents = reader.load_data()

node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

opensearch_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    # http_auth=('admin', 'OpenSearch1206')
)
print(opensearch_client.info())

index_name = "llamaindex-opensearch"

mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {"type": "object"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 256, "m": 48},
                }
            }
        }
    }
}

if opensearch_client.indices.exists(index=index_name):
    print(f"Index '{index_name}' exists.")
else:
    response = opensearch_client.indices.create(index=index_name, body=mapping, ignore=400)
    if 'acknowledged' in response:
        print(f"Index '{index_name}' created successfully: {response['acknowledged']}")
    else:
        print(f"Failed to create index '{index_name}': {response}")

client = OpensearchVectorClient(
        endpoint="http://localhost:9200",
        index=index_name,
        dim=768,
        embedding_field="embedding",
        text_field="content",
        search_pipeline="hybrid-search-pipeline",
        # http_auth=('admin', 'OpenSearch1206'),
        # use_ssl=False,
        # verify_certs=False,
        # ssl_assert_hostname=False,
        # ssl_show_warn=False,
        timeout=30,
    )

vectore_store = OpensearchVectorStore(
    client = client,
)

index = VectorStoreIndex.from_vector_store(vector_store=vectore_store)


def compute_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()

# 定义函数：存储前检查是否已存在
def insert_nodes_with_delta_check(nodes):
    for node in nodes:
        node_hash = compute_hash(node.get_text())
        document = Document(text=node.get_text(), metadata={"node_hash": node_hash})
        
        search_query = {
            "query": {
                "term": {
                    "metadata.node_hash.keyword": node_hash
                }
            }
        }
        response = opensearch_client.search(index=index_name, body=search_query)
        if response["hits"]["total"]["value"] == 0:
            index.insert(document)
            if not response:
                print("Insert operation returned None.")
        else:
            print(f"Node already exists: {document.get_text()[:30]}")
    
print("Inserting nodes with delta check...")
insert_nodes_with_delta_check(nodes)
print("nodes insertion complete.")


class CustomRetriever(BaseRetriever):
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name

    def _retrieve(self, query_str):
        if isinstance(query_str, QueryBundle):
            query_str = query_str.query_str
        else:
            query_str = query_str
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": query_str}}
                    ],
                }
            },
            "size": 2  # 限制返回结果的数量
        }
        response = self.client.search(index=self.index_name, body=query)
        # print(f"response: {response}")
        # print(f"response['hits']['hits'][0]: {response['hits']['hits'][0]}")
        nodes_with_scores = [
            NodeWithScore(
                node=TextNode(
                    text=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {})
                ),
                score=hit["_score"]
            ) for hit in response["hits"]["hits"]
        ]
        return nodes_with_scores

class MyLLMQueryEngine(CustomQueryEngine):
    llm: Ollama = Field(default=None, description="llm model")
    retriever: BaseRetriever = Field(default=None, description="retriever model")

    def __init__(self, retriever: BaseRetriever, llm: Ollama):
        super().__init__()
        self.llm = llm
        self.retriever = retriever

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        print(f"nodes: {nodes}")
        print(f"nodes[0]: {nodes[0]}")

        context_str = "\n\n".join([node.get_text() for node in nodes])
        print(f"context_str: {context_str}")
        
        qa_prompt = PromptTemplate(
            "根据以下上下文回答输入问题： \n"
            "----------------------\n"
            "{context_str}\n"
            "回答以下问题：不要编造答案，只回答问题中提到的内容。\n"
            "问题：{question_str}\n"
            "答案："
        )
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, question_str=query_str),
            temperature = 0.0,    #禁用随机性
            stop = ["答案：", "问题："],
            stream = True,
            timeout = 180
        )
        return str(response)
    

retriever = index.as_retriever()
# retriever = CustomRetriever(client=opensearch_client, index_name=index_name)
llm = Ollama(model="qwen:14b")

my_query_engine = MyLLMQueryEngine(retriever=retriever, llm=llm)
while True:
    query_str = input("请输入问题：")
    if query_str == "quit":
        break
    response = my_query_engine.custom_query(query_str)
    print("AI回答: ", response)
