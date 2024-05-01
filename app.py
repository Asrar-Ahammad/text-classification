from flask import Flask, render_template,request
import random
from transformers import BertTokenizer, BertForSequenceClassification, AdamW , BertModel
from langdetect import detect
import torch
from indicnlp.tokenize import sentence_tokenize


app = Flask(__name__, template_folder='templates')
class MyModelClass:
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, max_length=128, batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        # self.bert  = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, texts, labels):
        # Splitting data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

        # Creating datasets
        train_dataset = HindiTextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = HindiTextDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        # Creating data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, num_epochs=6.):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
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
        encoded_text = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            return "0" if prediction == 0 else "1"
    def save_model(self, path='nlp_model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path='nlp_model.pth'):
        state_dict = torch.load(path, map_location=self.device)
        missing_keys = set(self.model.state_dict().keys()) - set(state_dict.keys())

        if 'bert.embeddings.position_ids' in missing_keys:
            print("Warning: The 'position_ids' parameter is missing in the loaded state_dict. Ignoring it.")
            missing_keys.remove('bert.embeddings.position_ids')

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        print(f'Model loaded from {path}')
    def load_eng_model(self, path='eng_model.pth'):
        state_dict = torch.load(path, map_location=self.device)
        missing_keys = set(self.model.state_dict().keys()) - set(state_dict.keys())

        if 'bert.embeddings.position_ids' in missing_keys:
            print("Warning: The 'position_ids' parameter is missing in the loaded state_dict. Ignoring it.")
            missing_keys.remove('bert.embeddings.position_ids')
        
        # Load the state_dict into the model without raising an error for missing keys
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        print(f'English model loaded from {path}')
    
    def load_tamil_model(self, path='tamil_model.pth'):
        state_dict = torch.load(path, map_location=self.device)
        missing_keys = set(self.model.state_dict().keys()) - set(state_dict.keys())

        if 'bert.embeddings.position_ids' in missing_keys:
            print("Warning: The 'position_ids' parameter is missing in the loaded state_dict. Ignoring it.")
            missing_keys.remove('bert.embeddings.position_ids')
        
        # Load the state_dict into the model without raising an error for missing keys
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        print(f'Tamil model loaded from {path}')

    def eval(self):
        """Sets the model to evaluation mode."""
        self.model.eval()
arch_model = MyModelClass(model_name='bert-base-multilingual-cased', num_labels=2)

# Load the saved Hindi model
hindi_model_path = 'nlp_model.pth'
arch_model.load_model(path=hindi_model_path)

# Load the saved English model
english_model_path = 'eng_model.pth'
arch_model.load_eng_model(path=english_model_path)

# Load the saved Tamil model
# tamil_model_path = 'tamil_model.pth'
# arch_model.load_tamil_model(path=tamil_model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detection():
    result_text = ""
    ai_percentage = 0  
    Human_percentage = 0
    selected_toggle = ""  # Initialize selected toggle variable
    prediction = ""
    sentence = ""
    Detected_text = ""
    ai_result = ""
    if request.method == 'POST':
        input_text = request.form['input_text']
        Detected_text = detect(input_text)
        ai_text_count = 0  # Counter for AI-generated text
        total_text_count = 0  # Total text count
        print("Form Data:", request.form)
        # Check language
        language_toggle = request.form.get('language')
        
        # Print the selected toggle
        print("Selected Toggle:", language_toggle)
        
        # Set selected_toggle value for use in the template
        selected_toggle = language_toggle
        
        if selected_toggle =='hindi' and Detected_text=='hi':
           
            sentences = sentence_tokenize.sentence_split(input_text, lang='hi')
            
            for sentence in sentences:
                print(sentence)
                if sentence.strip() and prediction != "ERROR":  # Skip 
                    prediction = arch_model.classify_text(sentence)
                    if prediction == "1":  # Assuming threshold for AI detection
                        ai_text_count += 1
                        result_text += f"<span style='color:red'>{sentence.strip()}</span>. "
                        ai_result = "The text is likely to be AI generated"
                    elif prediction == "0":
                        result_text += f"{sentence.strip()}. "
                        ai_result = "The text is likely to be Human generated"
                    total_text_count += 1
            if total_text_count > 0:
                ai_percentage = (ai_text_count / total_text_count) * 100
                ai_percentage = round(ai_percentage,0)
                Human_percentage = 100 - ai_percentage
        elif selected_toggle =='english' and Detected_text=='en':
            sentences = input_text.split('.')
            for sentence in sentences:
                print(sentence )
                if sentence.strip() and prediction != "ERROR":  # Skip 
                    prediction = arch_model.classify_text(sentence)
                    if prediction == "1":  # Assuming threshold for AI detection
                        ai_text_count += 1
                        result_text += f"<span style='color:red'>{sentence.strip()}</span>. "
                        
                    elif prediction == "0":
                        result_text += f"{sentence.strip()}. "
                        
                    total_text_count += 1
            if total_text_count > 0:
                ai_percentage = (ai_text_count / total_text_count) * 100
                ai_percentage = round(ai_percentage,0)
                Human_percentage = 100 - ai_percentage

        # if selected_toggle =='on' and Detected_text=='ta':
        #     sentences = input_text.split('.')
        #     for sentence in sentences:
        #         print(sentence )
        #         if sentence.strip() and prediction != "ERROR":  # Skip 
        #             prediction = arch_model.classify_text(sentence)
        #             if prediction == "1":  # Assuming threshold for AI detection
        #                 ai_text_count += 1
        #                 result_text += f"<span style='color:red'>{sentence.strip()}</span>. "
                        
        #             elif prediction == "0":
        #                 result_text += f"{sentence.strip()}. "
                        
        #             total_text_count += 1
        #     if total_text_count > 0:
        #         ai_percentage = (ai_text_count / total_text_count) * 100
        
        else:
            prediction = "ERROR"
            result_text = ""
            result_text += f"<span style='color:red'>Entered Language Text Error :</span>"
            result_text += f"<span style='color:blue'>Please Enter Suitable Language</span>. "
            ai_result = "The Entered Text is Not applicable "     
        if ai_percentage > 50:
            ai_result = "The text is likely to be AI generated"
        elif ai_percentage <50:
            ai_result = "The text is likely to be Human generated"
        else:
            ai_result = "The text is likely to be both AI generated and Human generated"
    return render_template('detect.html', result=result_text, ai_percentage=ai_percentage, Human_percentage = Human_percentage, selected_toggle=selected_toggle, ai_result = ai_result)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)


