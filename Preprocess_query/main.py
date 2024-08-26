from test_bert import predict_intent, get_intent_label
from Preprocess_query.spaCy import extract_entities_and_numerical_ranges
from bertkey import extract_keywords_excluding_entities_and_numericals
from query_construction import construct_query
from elastic_search_query_ask import search_documents 
from Preprocess_query.gpt import query_chatgpt
from Preprocess_query.microsoft_bing import get
from Preprocess_query.google import fetch
from Preprocess_query.gpt_and_gemini import retrieve
from Preprocess_query.integrated_doc_ranking import rank_documents
from classify_more import process_questions
import warnings
warnings.filterwarnings("ignore")




def generate_output(query):
    """
    Takes an input query and processes it using a function from text_processing.py.
    """
    # Process the query using the imported function
    
    # BERT
    predicted_label = predict_intent(query)
    initial_inquiry_type = get_intent_label(predicted_label)
    inquiry_type = process_questions(initial_inquiry_type)
    
    # # #spacy customised
    # # # intent_what = 
    
    # # spaCy
    named_entities = extract_entities_and_numerical_ranges(query)
    # Separate data into different arrays
    numerical_values = [num['value'] for num in named_entities['numerical_values']]
    location_entities = [loc['entity_text'] for loc in named_entities['location_entities']]
    other_entities = [ent['entity_text'] for ent in named_entities['other_entities']]
    
    # # bertkey - keywords 
    keywords = extract_keywords_excluding_entities_and_numericals(query)
    
    # Query contruction
    # Construct the query
    constructed_query = construct_query(
        keywords=keywords,
        location_entities=location_entities,
        numerical_values=numerical_values,
        other_entities=other_entities,
        inquiry_type=inquiry_type,
        intent="temperature"
    )
    
    # # Document retrieval 
    text_our_model = search_documents("documents", construct_query)
    text_chatgpt = query_chatgpt(query)
    text_microsoft_bing = get(query)
    text_google = fetch(query)
    text_gemini = retrieve(query)
    
    documents = [
        text_our_model,
        text_chatgpt,
        text_microsoft_bing,
        text_google,
        text_gemini
    ]
    
    ranked_results = rank_documents(query, documents)
    output_text = ranked_results[0][1]  
    return output_text

