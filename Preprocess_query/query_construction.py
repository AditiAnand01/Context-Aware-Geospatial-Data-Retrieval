import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from elasticsearch import Elasticsearch
import json

# Initialize spaCy and BERT-based model
nlp = spacy.load("en_core_web_sm")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize Elasticsearch client
es = Elasticsearch(
    cloud_id="My_deployment:YXNpYS1zb3V0aDEuZ2NwLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyRiNWQ1NDFhMThiNjg0MDU5OWYyMDU0ZWEwOTA0YzczZCRjOTJmNGJlODNmZTA0MzY3YjZmOTc3YjY4NGVjYmFlZA==",
    basic_auth=("elastic", "5zMY5jfeJVIPCcAvBrqPxr4f")
)

# Define index name
index_name = "documents"

# Create index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

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

# Function to construct Elasticsearch query
def construct_query(keywords, location_entities, numerical_values, other_entities, inquiry_type, intent):
    query = {
        "query": {
            "bool": {
                "must": []
            }
        }
    }

    # Add keywords to the query
    if keywords:
        query["query"]["bool"]["must"].append({
            "match": {
                "tags": ' '.join(keywords)
            }
        })

    # Add location entities to the query
    if location_entities:
        query["query"]["bool"]["must"].append({
            "match": {
                "location": ' '.join(location_entities)
            }
        })

    # Add numerical values to the query (if needed, adjust according to your use case)
    if numerical_values:
        query["query"]["bool"]["must"].append({
            "match": {
                "numerical": ' '.join(numerical_values)
            }
        })

    # Add other entities to the query
    if other_entities:
        query["query"]["bool"]["must"].append({
            "match": {
                "entities": ' '.join(other_entities)
            }
        })

    # Handle different inquiry types
    if inquiry_type == "factual":
        query["query"]["bool"]["must"].append({
            "term": {
                "type": "factual"
            }
        })
    elif inquiry_type == "yes_no":
        query["query"]["bool"]["must"].append({
            "term": {
                "type": "yes_no"
            }
        })

    # Handle intent (e.g., places, events)
    if intent:
        query["query"]["bool"]["must"].append({
            "match": {
                "intent": intent
            }
        })

    return query

# Example usage of the construct_query function
def main():
    # Define example values for the parameters
    keywords = ["technology", "innovation", "AI"]
    location_entities = ["San Francisco", "New York"]
    numerical_values = ["2024", "1000"]
    other_entities = ["John Doe", "NASA"]
    inquiry_type = "factual"  # This could be "factual" or "yes_no"
    intent_what = "technology"  # Example intent

    # Construct the query
    constructed_query = construct_query(
        keywords=keywords,
        location_entities=location_entities,
        numerical_values=numerical_values,
        other_entities=other_entities,
        inquiry_type=inquiry_type,
        intent=intent_what
    )

    # Print the constructed query
    print("Constructed Query:")
    print(json.dumps(constructed_query, indent=4))

if __name__ == "__main__":
    main()
