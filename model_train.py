import torch
import torch.utils.data
import transformers.models.albert.modeling_albert
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F


class EnglishTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)


class EnglishTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Pad sequences to max_length
        padding_length = self.max_length - input_ids.size(0)
        input_ids = F.pad(input_ids, (0, padding_length),
                          value=self.tokenizer.pad_token_id)
        attention_mask = F.pad(attention_mask, (0, padding_length), value=0)

        print("Input Text:", text)
        print("Label:", label)
        print("Input IDs Shape:", input_ids.shape)
        print("Attention Mask Shape:", attention_mask.shape)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MyModelClass:
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, max_length=128, batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, texts, labels):
        # Splitting data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=42)

        # Creating datasets
        train_dataset = EnglishTextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = EnglishTextDataset(
            val_texts, val_labels, self.tokenizer, self.max_length)

        # Creating data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, num_epochs=6):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{num_epochs} completed.')

            self.evaluate()

    def evaluate(self):
        self.model.eval()
        val_preds, val_labels = [], []

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Validation Accuracy: {val_acc}')

    def classify_text(self, text):
        encoded_text = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            return "Human-written" if prediction == 0 else "AI-generated"

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='eng_model.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)  # Ensure the model is on the correct device
        print(f'Model loaded from {path}')

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()

class MyModelClass:
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, max_length=128, batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, texts, labels):
        # Splitting data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=42)

        # Creating datasets
        train_dataset = EnglishTextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = EnglishTextDataset(
            val_texts, val_labels, self.tokenizer, self.max_length)

        # Creating data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, num_epochs=6):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{num_epochs} completed.')

            self.evaluate()

    def evaluate(self):
        self.model.eval()
        val_preds, val_labels = [], []

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Validation Accuracy: {val_acc}')

    def classify_text(self, text):
        encoded_text = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            return "Human-written" if prediction == 0 else "AI-generated"

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='eng_model.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)  # Ensure the model is on the correct device
        print(f'Model loaded from {path}')

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()

df = pd.read_csv("../data/ETrain.csv")
subset_label_1 = label_1_data.sample(
    frac=1, random_state=42)  # 50% of label 1 data
subset_label_0 = label_0_data.sample(frac=0.5, random_state=42)
new_dataset = pd.concat([subset_label_1, subset_label_0], ignore_index=True)
df = new_dataset
texts = df['text']
labels = df['generated']
texts = texts.values
labels = labels.values
from torch.optim.lr_scheduler import StepLR
model = MyModelClass()
# Assuming train_texts and train_labels are defined
model.prepare_data(texts, labels)
model.train()
model.save_model(path='english_model_40k.pth')