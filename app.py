from flask import Flask, request, jsonify, render_template
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Log in to Hugging Face Hub
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
model = AutoModelForSeq2SeqLM.from_pretrained("manu2501sharma/my_summarization_model")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_text = request.form['text_input']
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate prediction
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True)
    prediction =tokenizer.decode(outputs[0], skip_special_tokens=True) 
     #tokenizer.decode(outputs[0], skip_special_tokens=True)  
  


    return render_template('index1.html', prediction_text='Prediction: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
