import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Constants
MAX_LEN = 512

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model Definition
class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 2)  # 2 classes: 'find' and 'is'

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.fc(output)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = BERTClass()
model.load_state_dict(torch.load(r'C:\VSCode\isro-Website - Copy\bert_modelbert_model_question_classification.pth'))
model.to(device)

# Function to predict the intent of a single text
def predict_intent(text):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        _, preds = torch.max(outputs, dim=1)
    
    return preds.item()

# Function to map the predicted label to the corresponding intent
def get_intent_label(label):
    label_map = {0: 'find', 1: 'is'}
    return label_map.get(label, "Unknown")

# Test the function with a single input text
def main():
    test_text = "Is the city of San Diego entirely within the state of California?"
    predicted_label = predict_intent(test_text)
    
    # Print the result
    print(f"Text: {test_text}")
    print(f"Predicted Intent: {get_intent_label(predicted_label)}")

if __name__ == "__main__":
    main()
