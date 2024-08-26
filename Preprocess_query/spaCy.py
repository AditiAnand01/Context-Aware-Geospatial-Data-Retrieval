import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_entities_and_numerical_ranges(text):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize lists for categorized entities
    numerical_values = []
    location_entities = []
    other_entities = []
    
    # Extract named entities and categorize them
    for ent in doc.ents:
        start = ent.start
        end = ent.end
        prefix = doc[start-1].text if start > 0 else ''
        suffix = doc[end].text if end < len(doc) else ''
        
        entity_info = {
            'entity_text': ent.text,
            'label': ent.label_,
            'prefix': prefix,
            'suffix': suffix
        }
        
        if ent.label_ in ['GPE', 'LOC']:  # GPE (Geopolitical Entity) or LOC (Location)
            location_entities.append(entity_info)
        elif ent.label_ in ['DATE', 'TIME', 'QUANTITY', 'MONEY']:  # Some numerical data categories
            numerical_values.append(entity_info)
        else:
            other_entities.append(entity_info)
    
    return {
        'numerical_values': numerical_values,
        'location_entities': location_entities,
        'other_entities': other_entities
    }

# # Example usage:
# query = "Get all parks located within the bounding box defined by latitude 34.0522 to 34.1622 and longitude -118.2437 to -118.1437."
# result = extract_entities_and_numerical_ranges(query)

# print("Numerical Values:")
# for num in result['numerical_values']:
#     print(f"Value: {num['entity_text']}, Prefix: '{num['prefix']}', Suffix: '{num['suffix']}'")

# print("\nLocation Entities:")
# for loc in result['location_entities']:
#     print(f"Entity: {loc['entity_text']}, Label: {loc['label']}, Prefix: '{loc['prefix']}', Suffix: '{loc['suffix']}'")

# print("\nOther Entities:")
# for other in result['other_entities']:
#     print(f"Entity: {other['entity_text']}, Label: {other['label']}, Prefix: '{other['prefix']}', Suffix: '{other['suffix']}'")
