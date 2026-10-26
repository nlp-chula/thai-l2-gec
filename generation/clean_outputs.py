import os
import pandas as pd
import google.generativeai as genai
import json
import time
from tqdm import tqdm

# Load Gemini API Key
api_keys = {}
with open('keys.txt', 'r') as f:
    for line in f:
        if '=' in line:
            key, val = line.strip().split('=', 1)
            api_keys[key] = val

genai.configure(api_key=api_keys.get('GEMINI_API_KEY', ''))
gemini_model = genai.GenerativeModel('gemini-3.5-flash')

df = pd.read_csv('generated_gec_output.csv')

openai_clean_list = [None] * len(df)
gemini_clean_list = [None] * len(df)

batch_size = 10
total_rows = len(df)

print(f"Cleaning {total_rows} rows in batches of {batch_size}...")

for start_idx in tqdm(range(0, total_rows, batch_size)):
    end_idx = min(start_idx + batch_size, total_rows)
    batch_data = []
    
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        batch_data.append({
            "id": idx,
            "original": row['original_sentence'],
            "openai_raw": row['openai_edited'],
            "gemini_raw": row['gemini_edited']
        })
        
    prompt = f"""
You are an expert editor. You are given a JSON array containing raw model responses for Grammatical Error Correction (GEC) in Thai.
Your task is to extract ONLY the final, single, best corrected version of the sentence for both OpenAI and Gemini.

Rules:
1. Do not include explanations, notes, prefixes (such as "ประโยคที่ถูกต้องคือ:"), markdown bolding (**), quotes, bullet points, or numbering.
2. Output ONLY the clean, corrected Thai sentence itself.
3. If multiple correction options are suggested in the reply, select the most natural/correct one.
4. If the raw input is already a clean sentence, keep it as is.
5. Return the result as a JSON array of objects with the fields: "id", "openai_clean", "gemini_clean".

Input data:
{json.dumps(batch_data, ensure_ascii=False, indent=2)}

Response MUST be a valid JSON array matching the request.
"""
    
    success = False
    retries = 3
    while not success and retries > 0:
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            results = json.loads(response.text)
            
            # Map results back
            for item in results:
                item_id = int(item['id'])
                openai_clean_list[item_id] = item['openai_clean']
                gemini_clean_list[item_id] = item['gemini_clean']
            
            success = True
        except Exception as e:
            retries -= 1
            print(f"\nError processing batch {start_idx}-{end_idx}: {e}. Retries left: {retries}")
            time.sleep(2)
            
    # Small sleep to respect rate limits
    time.sleep(1)

# Double check if any elements were missed/None and handle them
for idx in range(total_rows):
    if openai_clean_list[idx] is None:
        openai_clean_list[idx] = df.iloc[idx]['openai_edited']
    if gemini_clean_list[idx] is None:
        gemini_clean_list[idx] = df.iloc[idx]['gemini_edited']

# Save cleaned output
clean_df = pd.DataFrame({
    'writing_id': df['writing_id'],
    'sentence_id': df['sentence_id'],
    'original_sentence': df['original_sentence'],
    'openai_clean': openai_clean_list,
    'gemini_clean': gemini_clean_list
})

clean_df.to_csv('generated_gec_output_clean.csv', index=False, encoding='utf-8-sig')
print("\nCleaning complete! Results saved to generated_gec_output_clean.csv")
