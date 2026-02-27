"""
RAG Disk Search — Офлайн, CPU-д зориулсан
Файлуудыг chunk-лэж индекслээд, асуултад AI-р хариулна.
Ollama (үндсэн) эсвэл flan-t5 (нөөц) ашиглана.
"""
import os
import json
import pickle
import hashlib
import requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Файл уншигч сангууд (optional)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    CSV_AVAILABLE = True
except ImportError:
    CSV_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

load_dotenv()

# === CONFIG ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FOLDER = "faiss_rag_index"
METADATA_FILE = "rag_metadata.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = {
    '.txt', '.pdf', '.docx', '.doc', '.json', '.jsonl',
    '.csv', '.md', '.pptx', '.ppt', '.log',
}

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2"


# ══════════════════════════════════════════════════════════
#  Ollama LLM — офлайн, CPU
# ══════════════════════════════════════════════════════════
class OllamaLLM:
    def __init__(self, model=OLLAMA_MODEL, base_url=OLLAMA_URL):
        self.model = model
        self.base_url = base_url

    def is_available(self):
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def list_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def generate_stream(self, prompt, temperature=0.3):
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True,
                      "options": {"temperature": temperature}},
                stream=True, timeout=300,
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            yield "\n❌ Ollama холбогдсонгүй. 'ollama serve' ажиллуулна уу."
        except Exception as e:
            yield f"\n❌ Алдаа: {e}"

    def generate(self, prompt, temperature=0.3, max_tokens=512):
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": {"temperature": temperature, "num_predict": max_tokens}},
                timeout=300,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            return f"Ollama алдаа: {r.status_code}"
        except Exception as e:
            return f"❌ Алдаа: {e}"


# ══════════════════════════════════════════════════════════
#  RAG систем
# ══════════════════════════════════════════════════════════
class DiskSearchRAG:
    def __init__(self, search_paths=None):
        self.search_paths = search_paths or []
        self.db = None
        self.metadata = {}

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        print(f"🔄 Embedding ачаалж байна ({EMBEDDING_MODEL})...")
        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("✅ Embedding бэлэн\n")

        # AI model автоматаар сонгох
        self.llm = None
        self.pipe = None
        self.ai_mode = "none"
        self._init_ai()

    # ── AI model сонгох ──────────────────────────────────
    def _init_ai(self):
        ollama = OllamaLLM()
        if ollama.is_available():
            models = ollama.list_models()
            if models:
                names = [m.split(":")[0] for m in models]
                if OLLAMA_MODEL in names or OLLAMA_MODEL.split(":")[0] in names:
                    ollama.model = OLLAMA_MODEL
                else:
                    ollama.model = models[0]
                self.llm = ollama
                self.ai_mode = "ollama"
                print(f"✅ Ollama бэлэн — model: {ollama.model}")
                if len(models) > 1:
                    print(f"   Бусад: {', '.join(models[:5])}")
                return

        print("⚠️ Ollama олдсонгүй → flan-t5-base ачаалж байна...")
        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=200,
                device=-1,
            )
            self.ai_mode = "flan-t5"
            print("✅ flan-t5-base бэлэн\n")
        except Exception as e:
            print(f"⚠️ AI model ачаалж чадсангүй: {e}")
            self.ai_mode = "none"

    # ── Файл уншигчид ────────────────────────────────────
    @staticmethod
    def _read_txt(filepath):
        for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc, errors="ignore") as f:
                    return f.read(200_000)
            except Exception:
                continue
        return None

    @staticmethod
    def _read_pdf(filepath):
        if not PDF_AVAILABLE:
            return None
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [p.extract_text() or "" for p in reader.pages[:50]]
            return "\n".join(pages).strip()
        except Exception:
            return None

    @staticmethod
    def _read_docx(filepath):
        if not DOCX_AVAILABLE:
            return None
        try:
            doc = docx.Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception:
            return None

    @staticmethod
    def _read_csv(filepath):
        if not CSV_AVAILABLE:
            return None
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(filepath, encoding=enc, nrows=500).to_string()
            except Exception:
                continue
        return None

    @staticmethod
    def _read_json(filepath):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                if filepath.endswith(".jsonl"):
                    lines = [json.loads(ln) for i, ln in enumerate(f) if i < 100]
                    return "\n---\n".join(
                        json.dumps(d, indent=2, ensure_ascii=False) for d in lines
                    )
                return json.dumps(json.load(f), indent=2, ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _read_pptx(filepath):
        if not PPTX_AVAILABLE:
            return None
        try:
            prs = Presentation(filepath)
            texts = []
            for slide in prs.slides[:50]:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        texts.append(shape.text)
            return "\n".join(texts)
        except Exception:
            return None

    def read_file(self, filepath):
        ext = Path(filepath).suffix.lower()
        readers = {
            ".txt": self._read_txt, ".md": self._read_txt, ".log": self._read_txt,
            ".pdf": self._read_pdf,
            ".docx": self._read_docx, ".doc": self._read_docx,
            ".csv": self._read_csv,
            ".json": self._read_json, ".jsonl": self._read_json,
            ".pptx": self._read_pptx, ".ppt": self._read_pptx,
        }
        reader = readers.get(ext)
        return reader(filepath) if reader else None

    # ── Туслах функцууд ──────────────────────────────────
    @staticmethod
    def _file_hash(filepath):
        try:
            md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _should_skip(dirpath):
        skip = {
            "node_modules", "__pycache__", ".git", ".venv", "venv",
            "appdata", "windows", "program files", "system32",
            "$recycle.bin", "recovery", "programdata",
        }
        lower = dirpath.lower()
        return any(s in lower for s in skip)

    # ── Scan + Index ─────────────────────────────────────
    def scan_and_index(self, max_files=1000, max_size_mb=10):
        """Файлуудыг уншиж, chunk-лэж, FAISS индекс үүсгэнэ."""
        print(f"\n🔍 Хайж байна: {', '.join(self.search_paths)}")
        print(f"📂 Төрлүүд: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

        max_bytes = max_size_mb * 1024 * 1024
        all_chunks = []
        file_count = 0
        file_list = []

        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                print(f"⚠️ Олдсонгүй: {search_path}")
                continue

            for root, dirs, files in os.walk(search_path):
                if self._should_skip(root):
                    dirs[:] = []
                    continue

                for filename in files:
                    if file_count >= max_files:
                        break

                    ext = Path(filename).suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue

                    filepath = os.path.join(root, filename)
                    try:
                        fsize = os.path.getsize(filepath)
                        if fsize > max_bytes or fsize == 0:
                            continue
                    except Exception:
                        continue

                    content = self.read_file(filepath)
                    if not content or len(content.strip()) < 50:
                        continue

                    # Chunk-лэх
                    chunks = self.splitter.split_text(content)

                    file_meta = {
                        "filename": filename,
                        "filepath": filepath,
                        "extension": ext,
                        "size_kb": fsize / 1024,
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat(),
                        "hash": self._file_hash(filepath),
                    }
                    file_list.append(file_meta)

                    for ci, chunk_text in enumerate(chunks):
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                **file_meta,
                                "chunk_index": ci,
                                "total_chunks": len(chunks),
                            },
                        )
                        all_chunks.append(doc)

                    file_count += 1
                    if file_count % 50 == 0:
                        print(f"  ✅ {file_count} файл ({len(all_chunks)} chunk)...")

                if file_count >= max_files:
                    break

        print(f"\n📊 {file_count} файл → {len(all_chunks)} chunk")

        if not all_chunks:
            print("❌ Файл олдсонгүй")
            return False

        print("🔄 FAISS индекс үүсгэж байна...")
        self.db = FAISS.from_documents(all_chunks, self.embedding)
        os.makedirs(INDEX_FOLDER, exist_ok=True)
        self.db.save_local(INDEX_FOLDER)

        self.metadata = {
            "created": datetime.now().isoformat(),
            "num_files": file_count,
            "num_chunks": len(all_chunks),
            "files": file_list,
        }
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"✅ Индекс хадгалагдлаа: {INDEX_FOLDER}/")
        return True

    def load_index(self):
        if not os.path.exists(INDEX_FOLDER):
            return False
        try:
            print("🔄 Индекс ачаалж байна...")
            self.db = FAISS.load_local(
                INDEX_FOLDER, self.embedding,
                allow_dangerous_deserialization=True,
            )
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "rb") as f:
                    self.metadata = pickle.load(f)
            nf = self.metadata.get("num_files", self.metadata.get("num_documents", 0))
            nc = self.metadata.get("num_chunks", "?")
            print(f"✅ Ачаалагдлаа: {nf} файл, {nc} chunk")
            return True
        except Exception as e:
            print(f"❌ Индекс ачаалахад алдаа: {e}")
            return False

    # ── Хайлт ────────────────────────────────────────────
    def search(self, query, k=5):
        if not self.db:
            print("❌ Индекс ачаалаагүй")
            return []
        try:
            return self.db.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"❌ Хайлтад алдаа: {e}")
            return []

    # ── AI хариулт ───────────────────────────────────────
    def answer(self, question, k=5):
        results = self.search(question, k=k)
        if not results:
            print("❌ Холбогдох мэдээлэл олдсонгүй.\n")
            return

        # Хайлтын үр дүн харуулах
        seen_files = set()
        sources = []
        context_parts = []

        print(f"\n📚 {len(results)} холбогдох хэсэг олдлоо:")
        for i, (doc, score) in enumerate(results, 1):
            fn = doc.metadata.get("filename", "?")
            ci = doc.metadata.get("chunk_index", 0)
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"  {i}. {fn} [#{ci}] (score: {score:.3f}) — {snippet}...")

            context_parts.append(f"[{fn} хэсэг {ci}]:\n{doc.page_content}")
            if fn not in seen_files:
                sources.append(fn)
                seen_files.add(fn)

        context = "\n\n---\n\n".join(context_parts)

        # AI хариулт
        if self.ai_mode == "ollama":
            self._answer_ollama(context, question, sources)
        elif self.ai_mode == "flan-t5":
            self._answer_flan(context, question, sources)
        else:
            self._answer_fallback(results, sources)

    def _answer_ollama(self, context, question, sources):
        prompt = f"""Дараах эх сурвалжуудаас ЗӨВХӨН мэдээлэл ашиглан асуултад хариулна уу.

ЭХ СУРВАЛЖУУД:
{context[:3000]}

АСУУЛТ: {question}

ЗААВАРЧИЛГАА:
- Зөвхөн өгөгдсөн эх сурвалжаас хариулна
- Товч, тодорхой хариулна
- Мэдээлэл байхгүй бол "Энэ мэдээлэл эх сурвалжид байхгүй" гэж хэлнэ
- Асуулт ямар хэлээр байна тэр хэлээр хариулна

ХАРИУЛТ:"""

        print(f"\n🤖 AI ({self.llm.model}):")
        print("─" * 50)
        for chunk in self.llm.generate_stream(prompt):
            print(chunk, end="", flush=True)
        print(f"\n\n📚 Эх: {', '.join(sources)}")
        print("─" * 50)

    def _answer_flan(self, context, question, sources):
        prompt = (
            f"Based on these documents, answer concisely:\n\n"
            f"{context[:800]}\n\n"
            f"Question: {question}\nAnswer:"
        )
        if len(prompt.split()) > 400:
            prompt = f"Answer based on:\n{context[:500]}\n\nQ: {question}\nA:"

        try:
            result = self.pipe(
                prompt,
                max_new_tokens=150,
                do_sample=False, num_beams=1, early_stopping=True,
            )
            text = ""
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "")

            if not text or len(text.strip()) < 5:
                print(f"\n📋 AI задлаж чадсангүй.\n📚 Эх: {', '.join(sources)}")
                return

            text = text.strip()

            # Давталт шалгах
            words = text.split()
            if len(words) > 3:
                counts = {}
                for w in words:
                    counts[w] = counts.get(w, 0) + 1
                if max(counts.values()) > len(words) / 3:
                    print(f"\n⚠️ AI давталт илэрсэн.\n📚 Эх: {', '.join(sources)}")
                    return

            # Prompt давталт шалгах
            if "Answer:" in text:
                text = text.split("Answer:")[-1].strip()

            print(f"\n🤖 AI (flan-t5):")
            print("─" * 50)
            print(text)
            print(f"\n📚 Эх: {', '.join(sources)}")
            print("─" * 50)

        except Exception as e:
            print(f"\n⚠️ AI алдаа: {e}")

    def _answer_fallback(self, results, sources):
        print(f"\n📋 AI идэвхгүй — файлын агуулга:")
        print("─" * 50)
        for i, (doc, score) in enumerate(results[:3], 1):
            fn = doc.metadata.get("filename", "?")
            print(f"\n[{i}] {fn}:")
            print(doc.page_content[:400].strip())
            if len(doc.page_content) > 400:
                print("...")
        print(f"\n📚 Эх: {', '.join(sources)}")
        print("─" * 50)

    # ── Статистик ────────────────────────────────────────
    def show_stats(self):
        if not self.metadata:
            print("📊 Статистик байхгүй\n")
            return
        print("\n" + "=" * 50)
        print("📊 Статистик")
        print("=" * 50)
        print(f"📅 Үүссэн: {self.metadata.get('created', '?')}")
        nf = self.metadata.get("num_files", self.metadata.get("num_documents", 0))
        print(f"📄 Файл: {nf}")
        print(f"📦 Chunk: {self.metadata.get('num_chunks', '?')}")
        print(f"🤖 AI: {self.ai_mode}"
              + (f" ({self.llm.model})" if self.ai_mode == "ollama" else ""))

        if "files" in self.metadata:
            exts = {}
            total_kb = 0.0
            for fm in self.metadata["files"]:
                ext = fm.get("extension", "?")
                exts[ext] = exts.get(ext, 0) + 1
                total_kb += fm.get("size_kb", 0)
            print(f"💾 Хэмжээ: {total_kb / 1024:.1f} MB")
            print("\n📂 Төрлүүд:")
            for ext, cnt in sorted(exts.items(), key=lambda x: -x[1]):
                print(f"   {ext}: {cnt}")
            print(f"\n📋 Файлууд ({min(20, len(self.metadata['files']))}"
                  f"/{len(self.metadata['files'])}):")
            for i, fm in enumerate(self.metadata["files"][:20], 1):
                print(f"   {i}. {fm.get('filename')} ({fm.get('extension')})")
            if len(self.metadata["files"]) > 20:
                print(f"   ... +{len(self.metadata['files']) - 20}")
        print("=" * 50 + "\n")

    # ── Интерактив горим ─────────────────────────────────
    def interactive(self):
        print("\n" + "=" * 50)
        print("🧠 RAG Хайлтын Систем")
        print("=" * 50)
        ai_label = self.ai_mode
        if self.ai_mode == "ollama":
            ai_label += f" ({self.llm.model})"
        print(f"🤖 AI: {ai_label}")
        print("💡 Командууд: stats | rescan | model <нэр> | exit")
        print("=" * 50 + "\n")

        while True:
            try:
                q = input("❓ Асуулт: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Баяртай!")
                break

            if not q:
                continue

            cmd = q.lower()
            if cmd == "exit":
                print("👋 Баяртай!")
                break
            if cmd == "stats":
                self.show_stats()
                continue
            if cmd == "rescan":
                self.scan_and_index()
                continue
            if cmd.startswith("model "):
                if self.ai_mode == "ollama":
                    self.llm.model = q[6:].strip()
                    print(f"✅ Model: {self.llm.model}\n")
                else:
                    print("⚠️ Ollama идэвхгүй байна\n")
                continue

            self.answer(q)
            print()


def main():
    print("🚀 RAG Диск Хайлтын Систем\n")

    # Суугаагүй сангуудын мэдэгдэл
    missing = []
    if not PDF_AVAILABLE:
        missing.append("PyPDF2")
    if not DOCX_AVAILABLE:
        missing.append("python-docx")
    if not CSV_AVAILABLE:
        missing.append("pandas")
    if not PPTX_AVAILABLE:
        missing.append("python-pptx")
    if missing:
        print(f"⚠️ Суугаагүй сангууд: {', '.join(missing)}")
        print(f"   pip install {' '.join(missing)}\n")

    # Хайх директори
    print("📁 Хайх директори (таслалаар тусгаарлана):")
    print("   Жишээ: D:/Documents, D:/Projects")
    try:
        user_paths = input("\n📂 Директори: ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    paths = (
        [p.strip() for p in user_paths.split(",") if p.strip()]
        if user_paths
        else ["D:/", "C:/Users"]
    )

    rag = DiskSearchRAG(search_paths=paths)

    # Индекс ачаалах эсвэл шинээр үүсгэх
    if os.path.exists(INDEX_FOLDER):
        print(f"\n✅ Индекс олдлоо: {INDEX_FOLDER}")
        try:
            choice = input("Ашиглах уу? (y/n, default=y): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if choice != "n":
            if rag.load_index():
                rag.interactive()
                return

    print("\n🔄 Шинэ индекс үүсгэж байна...")
    try:
        max_f = input("Файлын дээд хязгаар (default 1000): ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    max_files = int(max_f) if max_f.isdigit() else 1000

    if rag.scan_and_index(max_files=max_files):
        rag.interactive()
    else:
        print("❌ Индекс үүсгэж чадсангүй")


if __name__ == "__main__":
    main()