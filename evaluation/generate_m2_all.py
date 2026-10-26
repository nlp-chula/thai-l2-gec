import csv
import difflib
import re
from pythainlp.tokenize import word_tokenize

def generate_m2(original_sentences, target_sentences, output_path, annotator_id=0):
    with open(output_path, "w", encoding="utf-8") as out:
        for orig, tgt in zip(original_sentences, target_sentences):
            out.write("S " + " ".join(orig) + "\n")
            sm = difflib.SequenceMatcher(None, orig, tgt)
            edits_found = False
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    continue
                edits_found = True
                out.write(f"A {i1} {i2}|||Token|||{' '.join(tgt[j1:j2])}|||REQUIRED|||-NONE-|||{annotator_id}\n")
            if not edits_found:
                out.write(f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{annotator_id}\n")
            out.write("\n")

def load_reference(csv_file):
    original, target = [], []
    with open(csv_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if len(row) > 1:
                original.append(word_tokenize(row[2]))
                target.append(word_tokenize(row[3]))
    return original, target

def load_prediction(csv_file):
    preds = []
    with open(csv_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)
        if rows and len(rows[0]) > 0 and rows[0][0].lower() in ['text', 'writing_id', 'source', 'text']:
            rows = rows[1:]
        for row in rows:
            if len(row) > 1:
                text = row[1]
                splits = text.split('\\n')
                if len(splits) > 1:
                    text = re.sub("แก้ไขประโยคต่อไปนี้ให้ถูกต้องตามไวยากรณ์ แสดงผลเฉพาะประโยคที่แก้ไขแล้ว","",splits[1])
                else:
                    text = re.sub("แก้ไขประโยคต่อไปนี้ให้ถูกต้องตามไวยากรณ์ แสดงผลเฉพาะประโยคที่แก้ไขแล้ว","",text)
                if "#" in text:
                    text = text.split("#")[0]
                text = text.strip('<pad> ').strip('edit:').strip('แก้ไข:').strip('</s>').strip('|').strip(' ')
                preds.append(word_tokenize(text))
    return preds

if __name__ == "__main__":
    base_dir = "../csv_data"
    ref_file = f"{base_dir}/ctfl_cleaned_test.csv"
    pred_file = f"{base_dir}/ctfl_cleaned_test - gemini.csv"
    
    orig, target = load_reference(ref_file)
    preds = load_prediction(pred_file)
    
    # Generate ref.m2
    generate_m2(orig, target, f"{base_dir}/ref.m2")
    # Generate hyp.m2
    generate_m2(orig, preds, f"{base_dir}/gemini_hyp.m2")
