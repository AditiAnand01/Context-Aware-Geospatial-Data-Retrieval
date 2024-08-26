import spacy

model_path = "custom_ner_model"

def test_custom_ner_model(test_sentences):
    # Load the trained model
    nlp = spacy.load(model_path)

    # Loop through the test sentences and print detected entities
    for sentence in test_sentences:
        doc = nlp(sentence)
        print(f"Text: {sentence}")
        if doc.ents:
            for ent in doc.ents:
                print(f"Entity: {ent.text}, Label: {ent.label_}")
        else:
            print("No entities detected.")
        print("-" * 40)  # Add a separator between sentences for clarity

# Test sentences
test_sentences = [
    "Find all the restaurants",
    "List all schools located within the administrative boundary of New York City",
    "Retrieve the address for the coordinates latitude 37.421999 and longitude -122.084058.",
    "Get all parks located within the bounding box defined by latitude 34.0522 to 34.1622 and longitude -118.2437 to -118.1437.",
    "Where are parks",
    "Find restaurants?"
]

# Call the function with the model path and test sentences
test_custom_ner_model(test_sentences)
