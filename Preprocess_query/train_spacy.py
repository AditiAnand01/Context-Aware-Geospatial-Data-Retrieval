import spacy
import pandas as pd
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags
import matplotlib.pyplot as plt

# Load the CSV dataset
df = pd.read_csv(r'/home/ubuntu/Documents/EarthWise/corrected_dataset.csv')


nlp = spacy.load("en_core_web_sm")  # Load a pre-trained model


# Create an entity recognizer if it's not already present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER component
for label in df['Entity_Label'].unique():
    ner.add_label(label)

# Prepare training data in the required format
TRAIN_DATA = []
for _, row in df.iterrows():
    text = row['Text']
    start = int(row['Start'])
    end = int(row['End'])
    entity_label = row['Entity_Label']
    entities = [(start, end, entity_label)]
    
    # Check alignment
    tags = offsets_to_biluo_tags(nlp.make_doc(text), entities)
    if '-' in tags:
        print(f"Misaligned entity in text: '{text}' with entities {entities}")
        continue
    
    TRAIN_DATA.append((text, {"entities": entities}))

# Track losses and accuracies
losses_list = []
accuracies_list = []

# Disable other pipelines during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    # Initialize the model's weights
    optimizer = nlp.begin_training()
    
    # Training loop
    for iteration in range(100):  # Adjust the number of iterations as needed
        print(f"Starting iteration {iteration}")
        losses = {}
        correct_predictions = 0
        total_predictions = 0
        
        for text, annotations in TRAIN_DATA:
            # Create an Example object
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            
            # Update the model
            nlp.update([example], drop=0.2, losses=losses)  # Adjust dropout rate
            
            # Get the model's predictions
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]
            true_entities = annotations['entities']
            
            # Calculate accuracy: match start, end, and label
            correct_predictions += sum(1 for pred in pred_entities if pred in true_entities)
            total_predictions += len(true_entities)
        
        # Calculate and store accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracies_list.append(accuracy)
        
        # Store loss
        losses_list.append(losses.get('ner', float('inf')))  # Handle missing 'ner' key
        
        print(f"Losses at iteration {iteration}: {losses}")
        print(f"Accuracy at iteration {iteration}: {accuracy:.4f}")

# Save the model
nlp.to_disk("custom_ner_model")

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(len(losses_list)), losses_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('spacy_loss.png') 

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracies_list)), accuracies_list, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.savefig('spacy_accuracy.png') 

plt.tight_layout()
plt.show()