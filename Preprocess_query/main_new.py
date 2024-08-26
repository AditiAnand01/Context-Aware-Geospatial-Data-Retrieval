# main.py

from test_bert import predict_intent, get_intent_label
from spaCy import extract_entities_and_numerical_ranges
from bertkey import extract_keywords_excluding_entities_and_numericals
from query_construction import construct_query
from elastic_search_query_ask import search_documents 
from gpt import query_chatgpt
from microsoft_bing import get
from google import fetch
from gpt_and_gemini import retrieve
from integrated_doc_ranking import rank_documents
from classify_more import process_questions
from test_spacy import test_custom_ner_model
from askif_weather import check_if_query_is_finding
from askif_maps import check_if_query_is_finding_nearby
from askif_routes import check_if_query_is_finding_routes
from maps_api import find_places_nearby
from weather import get_weather_data
from route import get_driving_directions
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
    
    #spacy customised
    intent_what = test_custom_ner_model(query)
    
    # spaCy
    named_entities = extract_entities_and_numerical_ranges(query)
    # Separate data into different arrays
    numerical_values = [num['value'] for num in named_entities['numerical_values']]
    location_entities = [loc['entity_text'] for loc in named_entities['location_entities']]
    other_entities = [ent['entity_text'] for ent in named_entities['other_entities']]
    
    # bertkey - keywords 
    keywords = extract_keywords_excluding_entities_and_numericals(query)
    
    # Query contruction
    # Construct the query
    constructed_query = construct_query(
        keywords=keywords,
        location_entities=location_entities,
        numerical_values=numerical_values,
        other_entities=other_entities,
        inquiry_type=inquiry_type,
        intent=intent_what
    )
    
    # Check if the query is about finding weather conditions
    result, latitude, longitude = check_if_query_is_finding(query)
    if result == 1 :
        text_our_model = get_weather_data(latitude, longitude)
    else:
        # Check if the query is about finding nearby places
        result, place_type, latitude, longitude = check_if_query_is_finding_nearby(query)
        if result == 1:
            text_our_model = find_places_nearby(place_type, latitude, longitude)
        else:
            # Check if the query is about finding routes
            result, source_lat, source_lng, dest_lat, dest_lng = check_if_query_is_finding_routes(query)
            if result == 1:
                text_our_model = get_driving_directions(source_lat, source_lng, dest_lat, dest_lng)
            else:
               # Normal search retrieval
               text_our_model = search_documents("documents", constructed_query)
    
    
    # Document retrieval 
    #text_our_model = search_documents("documents", construct_query)
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
    output_text = ranked_results[0][1]  # Get the document content
    return output_text

# Example usage
if __name__ == "__main__":
    input_query = "weather of Hyderabad"
    result = generate_output(input_query)
    print("Processed Output:", result)
