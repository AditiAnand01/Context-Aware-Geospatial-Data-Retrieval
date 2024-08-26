import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import CrossEncoder
import torch

# Load DPR model and tokenizer
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')

# Load Cross-Encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def encode_text(texts, tokenizer, model, max_length=512):
    # Tokenize the input texts
    inputs = tokenizer(texts, 
                       return_tensors='pt', 
                       padding=True, 
                       truncation=True, 
                       max_length=max_length)
    
    # Move tensors to the same device as the model
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Compute embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output

    return embeddings

def compute_tfidf_scores(documents, query):
    vectorizer = TfidfVectorizer() 
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    query_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]
    tfidf_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    return np.array(tfidf_similarities)

def compute_dpr_scores(documents, query):
    doc_embeddings = encode_text(documents, context_tokenizer, context_encoder)
    query_embedding = encode_text([query], context_tokenizer, context_encoder)
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    return np.array(similarities)

def compute_cross_encoder_scores(query, documents):
    queries_docs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(queries_docs)
    return np.array(scores)

def normalize_scores(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

def rank_documents(query, documents, tfidf_weight=0.33, dpr_weight=0.33, cross_weight=0.34):
    tfidf_scores = compute_tfidf_scores(documents, query)
    dpr_scores = compute_dpr_scores(documents, query)
    cross_scores = compute_cross_encoder_scores(query, documents)
    
    # Normalize scores
    tfidf_scores = normalize_scores(tfidf_scores)
    dpr_scores = normalize_scores(dpr_scores)
    cross_scores = normalize_scores(cross_scores)
    
    # Combine scores
    combined_scores = (tfidf_weight * tfidf_scores + 
                       dpr_weight * dpr_scores + 
                       cross_weight * cross_scores)
    
    # Rank documents based on combined scores
    ranked_docs = sorted(enumerate(documents), key=lambda x: combined_scores[x[0]], reverse=True)
    
    return ranked_docs

def process_text(query, documents_sources):
    documents = [
        documents_sources["text_our_model"],
        documents_sources["text_chatgpt"],
        documents_sources["text_microsoft_bing"],
        documents_sources["text_google"],
        documents_sources["text_gemini"]
    ]
    
    # Rank the documents
    ranked_results = rank_documents(query, documents)
    
    # Get the top-ranked document
    top_ranked_doc = ranked_results[0][1]  # Get the document content
    
    print(f"Top-Ranked Document: {top_ranked_doc}")
    
    return top_ranked_doc

# Example usage
def main():
    query = "How is AI used in industries?"
    
    # Example inputs
    documents_sources = {
        "text_our_model": "AI is applied in multiple industries, transforming operations with machine learning.",
        "text_chatgpt": "Artificial intelligence is revolutionizing industries by automating tasks and optimizing processes.",
        "text_microsoft_bing": "AI finds use in various industries including healthcare, finance, and manufacturing.",
        "text_google": "Industries like healthcare and finance are being transformed by AI applications.",
        "text_gemini": "AI is widely used in industries for improving efficiency and innovation."
    }
    
    # Process and get the top-ranked document
    top_document = process_text(query, documents_sources)

# Run the main function
if __name__ == "__main__":
    main()
