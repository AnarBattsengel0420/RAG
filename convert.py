import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt') 

txt_folder = "sample_docs"
json_folder = "json_output"
jsonl_file = "merged.jsonl"


os.makedirs(json_folder, exist_ok=True)
all_data = []
seen_content = set()  

def chunk_text(text, max_length=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""], 
        length_function=len
    )
    sentences = sent_tokenize(text)
    text = " ".join(sentences) 
    chunks = splitter.split_text(text)
    return chunks

for file in os.listdir(txt_folder):
    if file.endswith(".txt"):
        try:
            with open(os.path.join(txt_folder, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                
                if content in seen_content:
                    print(f"Skipping duplicate content in {file}")
                    continue
                seen_content.add(content)

                chunks = chunk_text(content)

                for i, chunk in enumerate(chunks):
                    doc = {
                        "filename": file,
                        "chunk_index": i,
                        "content": chunk,
                        "metadata": {
                            "source": "school_text",
                            "category": "general",
                            "keywords": " ".join(nltk.word_tokenize(chunk.lower())[:10]) 
                        }
                    }

                    json_path = os.path.join(json_folder, file.replace(".txt", f"_chunk{i}.json"))
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(doc, jf, ensure_ascii=False, indent=2)

                    all_data.append(doc)

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

try:
    with open(jsonl_file, "w", encoding="utf-8") as out:
        for item in all_data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Converted {len(all_data)} chunks to {jsonl_file}")
except Exception as e:
    print(f"⚠️ Error writing to {jsonl_file}: {e}")