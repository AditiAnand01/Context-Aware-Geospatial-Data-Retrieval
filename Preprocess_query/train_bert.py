import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Load dataset
train_path = r'/home/ubuntu/Documents/EarthWise/Dataset.csv'
df = pd.read_csv(train_path)

# Preprocessing
df['Intent'] = df['Intent'].astype(int)

# Constants
MAX_LEN = 512
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS =70
LEARNING_RATE = 1e-5
PATIENCE = 3  # Number of epochs with no improvement after which training will be stopped

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['Query'].values
        self.labels = df['Intent'].values
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }

# Train-Validation Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

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

# Initialize Model
model = BERTClass()
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=total_steps
)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Early Stopping Parameters
best_val_loss = float('inf')
patience_counter = 0

# Training Function
def train_epoch(model, data_loader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Validation Function
def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Lists to store loss and accuracy for each epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training Loop with Early Stopping
for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
    val_acc, val_loss = eval_model(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print("Validation loss improved, saving model...")
        # Save model checkpoint
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break


# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.savefig('BERTloss.png') 

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.savefig('BERT_accuracy.png') 

plt.show()

# Save the model
torch.save(model.state_dict(), r'/home/ubuntu/Documents/EarthWise/bert_model\bert_model_question_classification.pth')