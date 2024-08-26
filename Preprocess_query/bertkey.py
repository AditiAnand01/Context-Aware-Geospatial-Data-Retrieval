# keyword_extraction.py

import spacy
from keybert import KeyBERT

# Initialize the spaCy model and KeyBERT model
nlp = spacy.load('en_core_web_sm')
keybert_model = KeyBERT()

def extract_keywords_excluding_entities_and_numericals(text):
    """
    Extracts keywords from the given text while excluding named entities and numerical values.
    
    Args:
        text (str): The text from which to extract keywords.
    
    Returns:
        list: A list of tuples with keywords and their scores, excluding named entities and numerical values.
    """
    # Process the text with spaCy
    doc = nlp(text)
    
    # Extract named entities and numerical values
    excluded_terms = {ent.text.lower() for ent in doc.ents}
    excluded_terms.update(token.text.lower() for token in doc if token.like_num)
    
    # Extract keywords using KeyBERT
    all_keywords = keybert_model.extract_keywords(text)
    
    # Filter out named entities and numerical values from the keywords
    filtered_keywords = [
        (keyword, score) for keyword, score in all_keywords
        if keyword.lower() not in excluded_terms and not keyword.replace('.', '', 1).isdigit()
    ]
    
    return filtered_keywords



def main():
    # Example text
    text = "List the 10 nearest gas stations to latitude 37.7749 and longitude -122.4194."
    
    # Extract keywords excluding named entities
    keywords = extract_keywords_excluding_entities_and_numericals(text)
    
    # Print the extracted keywords
    print("Extracted Keywords:")
    for keyword, score in keywords:
        print(f"{keyword}: {score:.4f}")

if __name__ == "__main__":
    main()