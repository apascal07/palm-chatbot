import pinecone
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from google.cloud import storage
from typing import List
from firebase_admin import auth, firestore, initialize_app
import json
import os
import tempfile
import google.cloud.firestore

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

initialize_app()

chunk_size = 500
chunk_overlap = 50

embeddings = VertexAIEmbeddings()
llm = VertexAI()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
db = Pinecone(index=pinecone.Index(os.getenv("PINECONE_INDEX")), embedding_function=embeddings.embed_query, text_key="text")
retriever = db.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

def firestorechatbot(event, response):
    firestore_client: google.cloud.firestore.Client = firestore.client()
    ref = event["value"]["name"].split("/", 5)[5]
    prompt = event["value"]["fields"]["prompt"]["stringValue"]
    res = qa(prompt)
    doc_ref = firestore_client.document(ref)
    doc_ref.set(
        {"response": res["result"]}, merge=True
    )


def ingester(event, response):
    doc_file = tempfile.NamedTemporaryFile(suffix=event['name'])
    download_blob(event['bucket'], event['name'], doc_file.name)
    ingest_file(doc_file.name)

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def ingest_file(file_path: str):
    documents = load_single_document(file_path)
    texts = text_splitter.split_documents(documents)
    db.add_documents(texts)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))
