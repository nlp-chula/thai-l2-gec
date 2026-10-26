import os
import time
import pandas as pd
from tqdm import tqdm

# IMPORTANT: You need to install these packages first:
# pip install openai google-generativeai pandas tqdm

import openai
import google.generativeai as genai

# 1. Load API Keys from keys.txt
api_keys = {}
try:
    with open('keys.txt', 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                api_keys[key] = value
except FileNotFoundError:
    print("Error: keys.txt not found. Please create it and add your API keys.")
    exit(1)

openai_key = api_keys.get('OPENAI_API_KEY', '')
gemini_key = api_keys.get('GEMINI_API_KEY', '')

if not openai_key or openai_key == 'your_openai_api_key_here':
    print("Warning: OpenAI API Key is missing or invalid in keys.txt")
if not gemini_key or gemini_key == 'your_gemini_api_key_here':
    print("Warning: Gemini API Key is missing or invalid in keys.txt")

# 2. Configure API Clients
# OpenAI Setup
client = openai.OpenAI(api_key=openai_key)

# Gemini Setup
genai.configure(api_key=gemini_key)
# Using gemini-3.5-flash as default
gemini_model = genai.GenerativeModel('gemini-3.5-flash')

def generate_with_openai(prompt):
    if not openai_key or openai_key == 'your_openai_api_key_here':
        return "N/A"
    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini-2026-03-17",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return f"Error: {e}"

def generate_with_gemini(prompt):
    if not gemini_key or gemini_key == 'your_gemini_api_key_here':
        return "N/A"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"Error: {e}"

# 3. Process the CSV File
input_csv = '../csv_data/ctfl_cleaned_test.csv'
output_csv = 'generated_gec_output.csv'

print(f"Reading input from {input_csv}...")
df = pd.read_csv(input_csv)

# Lists to store the generated outputs
openai_outputs = []
gemini_outputs = []

print("Starting generation...")
# Iterate over each row with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating GEC"):
    input_sentence = row['text']
    prompt = f"แก้ไขประโยคต่อไปให้ถูกต้องตามหลักไวยากรณ์: {input_sentence}"
    
    # Generate with OpenAI
    openai_edited = generate_with_openai(prompt)
    openai_outputs.append(openai_edited)
    
    # Generate with Gemini
    gemini_edited = generate_with_gemini(prompt)
    gemini_outputs.append(gemini_edited)
    
    # Small sleep to respect API rate limits (adjust if needed)
    time.sleep(1)

# 4. Save the results to a new CSV
# We will create a new DataFrame with just the essential columns
output_df = pd.DataFrame({
    'writing_id': df['writing_id'],
    'sentence_id': df['sentence_id'],
    'original_sentence': df['text'],
    'openai_edited': openai_outputs,
    'gemini_edited': gemini_outputs
})

output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\\nGeneration complete! Results saved to {output_csv}")
