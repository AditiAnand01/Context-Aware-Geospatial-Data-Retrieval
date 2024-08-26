import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Elasticsearch connection details
cloud_id = "My_deployment:YXNpYS1zb3V0aDEuZ2NwLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyRiNWQ1NDFhMThiNjg0MDU5OWYyMDU0ZWEwOTA0YzczZCRjOTJmNGJlODNmZTA0MzY3YjZmOTc3YjY4NGVjYmFlZA=="
user = "elastic"
password = "5zMY5jfeJVIPCcAvBrqPxr4f"

# Initialize Elasticsearch client
es_client = Elasticsearch(
    cloud_id=cloud_id,
    basic_auth=(user, password),
    timeout=60  # Increase timeout to 60 seconds
)

# Initialize spaCy and BERT-based model
nlp = spacy.load("en_core_web_sm")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Create index if it doesn't exist
def create_index(es_client, index_name):
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        logger.info(f"Index '{index_name}' created.")
    else:
        logger.info(f"Index '{index_name}' already exists.")

# Function to perform NER
def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to extract keywords using TF-IDF
def extract_keywords(texts, n_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = X.toarray().argsort()[:, -n_keywords:]
    return [[feature_array[i] for i in doc] for doc in tfidf_sorting]

# Function to classify intent
def classify_intent(text, intents):
    result = intent_classifier(text, intents)
    return result["labels"][0]

# Function to bulk index documents
def bulk_index_documents(es_client, index_name, documents, intents):
    actions = []
    for i, doc in enumerate(documents):
        content = doc.get('content', '')
        
        # Extract named entities
        named_entities = extract_named_entities(content)
        
        # Extract keywords
        extracted_keywords = extract_keywords([content])
        
        # Classify intent
        intent = classify_intent(content, intents)
        
        # Create labeled document
        labeled_doc = {
            "_index": index_name,
            "_id": i + 1,
            "_source": {
                "Intent(type)": intent,
                "Intent(what)": "Extracted data from the content",
                "named_entity": named_entities,
                "keywords": extracted_keywords[0],
                "tags": doc.get("tags", ""),
                "location": doc.get("location", ""),
                "numerical": doc.get("numerical", ""),
                "entities": doc.get("entities", "")
            }
        }
        
        actions.append(labeled_doc)

    # Bulk index documents
    try:
        success, failed = bulk(es_client, actions)
        logger.info(f"Successfully indexed {success} documents.")
        if failed:
            logger.warning(f"Failed to index {failed} documents.")
    except Exception as e:
        logger.error(f"Error during bulk indexing: {e}")

# Function to load documents from a CSV file
def load_documents_from_csv(csv_file_path):
    try:
        chunks = pd.read_csv(csv_file_path, chunksize=1000, on_bad_lines='skip', delimiter=',', quotechar='"')
        documents = []
        for chunk in chunks:
            documents.extend(chunk.to_dict(orient='records'))
        logger.info(f"Loaded {len(documents)} documents from CSV.")
        return documents
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        return []

# Function to perform a search query
def search_documents(es_client, index_name, query):
    return es_client.search(index=index_name, body=query)

# Main function to execute the workflow
def main():
    csv_file_path = r"C:\Users\Admin\OneDrive\Pictures\Desktop\ISRO_Hackathon\CodeWork\doc.csv"
    index_name = "documents"
    create_index(es_client, index_name)
    
    intents = ["information_request", "actionable_task", "general_inquiry"]
    
    documents = load_documents_from_csv(csv_file_path)
    bulk_index_documents(es_client, index_name, documents, intents)
    
    logger.info("Documents indexed successfully.")
    
    # Construct the search query
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"tags": "technology"}},
                    {"match": {"location": "San Francisco"}},
                    {"match": {"numerical": "2024"}},
                    {"match": {"entities": "John Doe"}},
                    {"term": {"Intent(type)": "factual"}},
                    {"match": {"Intent(what)": "technology"}}
                ]
            }
        }
    }
    
    # Execute the search query
    try:
        response = search_documents(es_client, index_name, query)
        logger.info("Query Results: %s", response)
    except Exception as e:
        logger.error(f"Error during search query: {e}")

# Run the main function
if __name__ == "__main__":
    main()
