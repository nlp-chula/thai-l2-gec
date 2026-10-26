import csv
import difflib
from pythainlp.tokenize import word_tokenize

def generate_m2(original_sentences, target_sentences, output_path, annotator_id=0):
    with open(output_path, "w", encoding="utf-8") as out:
        for orig, tgt in zip(original_sentences, target_sentences):
            # Write original sentence
            out.write("S " + " ".join(orig) + "\n")
            
            # Find edits
            sm = difflib.SequenceMatcher(None, orig, tgt)
            edits_found = False
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    continue
                
                edits_found = True
                
                # Format: A start end|||ErrorType|||Correction|||REQUIRED|||-NONE-|||AnnotatorID
                start = i1
                end = i2
                error_type = "Token"
                correction = " ".join(tgt[j1:j2])
                
                out.write(f"A {start} {end}|||{error_type}|||{correction}|||REQUIRED|||-NONE-|||{annotator_id}\n")
            
            # If no edits, we write a special "noop" edit (M2 scorer usually expects at least an A line for sentences or it's fine without it, but standard practice is A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0)
            if not edits_found:
                out.write(f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{annotator_id}\n")
            
            out.write("\n") # Blank line between sentences

if __name__ == "__main__":
    base_dir = "../csv_data"
    ref_file = f"{base_dir}/ctfl_cleaned_test.csv"
    
    original = []
    target = []
    with open(ref_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) > 1:
                original.append(word_tokenize(row[2]))
                target.append(word_tokenize(row[3]))
                
    m2_out = f"{base_dir}/ctfl_test.m2"
    generate_m2(original, target, m2_out, annotator_id=0)
    print(f"Generated {m2_out} with {len(original)} sentences.")
