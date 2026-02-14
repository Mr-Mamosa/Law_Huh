import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import traceback
import logging

# All the QueryEngine related imports and class definition will go here
import os
import time
from pathlib import Path
import re
import torch
from sentence_transformers import SentenceTransformer, util
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import hf_hub_download
import chromadb

# Initialize logger
logger = logging.getLogger(__name__)

# --- Configuration ---
# Build paths from the project root to avoid relative path issues.
# In views.py, Path(__file__).resolve().parent is rag_api, .parent is backend, .parent is project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
MODELS_PATH = PROJECT_ROOT / "data" / "models"
SC_JUDGMENTS_PATH = PROJECT_ROOT / "data" / "sc_judgments_text.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "legal_kb"
LOCAL_LLM_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LOCAL_LLM_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf"

LEGAL_TAGS = [
    "Criminal Intent (Mens Rea)",
    "Actus Reus (Guilty Act)",
    "Bail: Anticipatory",
    "Bail: Regular",
    "Fundamental Rights: Article 14 (Equality)",
    "Fundamental Rights: Article 19 (Freedoms)",
    "Fundamental Rights: Article 21 (Life and Liberty)",
    "Civil Procedure",
    "Criminal Procedure",
    "Contract Law",
    "Property Dispute",
]

def clean_and_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    return None

def find_citations(text: str) -> list[str]:
    patterns = [
        r"\bAIR\s+\d{4}\s+SC\s+\d+\b",
        r"\b\d{4}\s*\(?\s*\d+\s*\)?\s*SCC\s+\d+\b",
        r"\b\d{4}\s*\(?\s*\d+\s*\)?\s+SCR\s+\d+\b",
    ]
    found_citations = []
    for pattern in patterns:
        found_citations.extend(re.findall(pattern, text, re.IGNORECASE))
    return list(set(found_citations))

def get_full_text_for_source(source_name: str) -> str | None:
    full_text_chunks = []
    target_base_path = (
        source_name.replace("Supreme Court Judgment: ", "").strip().lower()
    )
    if not SC_JUDGMENTS_PATH.exists():
        return None
    with open(SC_JUDGMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                record_source = record.get("source", "").strip().lower()
                if record_source == target_base_path:
                    full_text_chunks.append(record.get("text", ""))
            except json.JSONDecodeError:
                continue
    return "\n".join(full_text_chunks) if full_text_chunks else None


class QueryEngine:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QueryEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if QueryEngine._initialized:
            return
        QueryEngine._initialized = True

        logger.info("🚀 Initializing the Legal AI Engine in SINGLE-ENGINE BEAST MODE...")
        MODELS_PATH.mkdir(exist_ok=True)
        llm_path = MODELS_PATH / LOCAL_LLM_FILENAME

        if not llm_path.exists():
            logger.info(f"Downloading LLM model to {llm_path}...")
            hf_hub_download(
                repo_id=LOCAL_LLM_REPO,
                filename=LOCAL_LLM_FILENAME,
                local_dir=str(MODELS_PATH),
                local_dir_use_symlinks=False,
            )
            logger.info("LLM model downloaded.")

        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, device=self.device
        )
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

        gpu_layers = 30 if self.device == "cuda" else 0
        logger.info(
            f"✅ GPU Mode Active: Offloading {gpu_layers} layers to {self.device.upper()}"
        )

        self.llm = CTransformers(
            model=str(llm_path),
            model_type="llama",
            config={
                "max_new_tokens": 1024,
                "temperature": 0.01,
                "context_length": 4096,
                "gpu_layers": gpu_layers,
                "batch_size": 4,
                "threads": 4,
            },
        )

        self.rag_with_claims_prompt = PromptTemplate(
            template="""<INST>
You are an AI legal assistant. Answer the user's question based ONLY on the provided context.
            You MUST provide a 'claims' list where each item is a JSON object with keys "claim" and "source_quote".

            Required JSON Format:
            {{
              "answer": "Your detailed legal answer here...",
              "claims": [
                {{ "claim": "First factual statement", "source_quote": "Exact quote from context supporting this" }}
              ]
            }}

            CONTEXT: {context}
            QUESTION: {question}
            JSON:
            """
        )

        self.precedent_prompt = PromptTemplate(
            template="""<INST>
            Analyze the relationship: PRIMARY CASE vs CITED CASE.
            Classify as: "Relied Upon", "Distinguished", or "Overruled/Modified".
            Output JSON: {{"relationship": "...", "justification": "..."}}
            PRIMARY CASE: {primary_case_text}
            CITED CASE: {cited_case_text}
            </INST>
            JSON:
            """
        )
        logger.info("Engine ready for advisory.")

    def retrieve_with_metadata(self, query_text: str, n_results=5):
        query_embedding = self.embedding_model.encode(
            query_text, convert_to_tensor=False
        ).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        source_names = {
            meta["source"] for meta in results["metadatas"][0] if "source" in meta
        }
        return "\n\n---\n\n".join(results["documents"][0]), list(source_names)

    def classify_query(self, query_text: str):
        prompt = PromptTemplate(
            template="""<INST> Identify relevant legal concepts. JSON {{"tags": []}}. Concepts: {tags}
 Query: {query} </INST>"""
        )
        tag_list_str = "\n".join(LEGAL_TAGS)
        chain = prompt | self.llm | StrOutputParser()
        return clean_and_parse_json(
            chain.invoke({"query": query_text, "tags": tag_list_str})
        )

    def verify_claims(self, claims_with_sources: list) -> tuple[int, int, float]:
        if not claims_with_sources:
            return 0, 0, 0.0
        valid_items = []
        for item in claims_with_sources:
            if (
                isinstance(item, dict)
                and item.get("claim")
                and item.get("source_quote")
            ):
                valid_items.append(item)
        if not valid_items:
            return 0, len(claims_with_sources), 1.0

        claims = [i["claim"] for i in valid_items]
        sources = [i["source_quote"] for i in valid_items]
        try:
            claim_embeddings = self.embedding_model.encode(
                claims, convert_to_tensor=True
            )
            source_embeddings = self.embedding_model.encode(
                sources, convert_to_tensor=True
            )
            similarities = util.cos_sim(claim_embeddings, source_embeddings)
            verified_count = sum(
                1 for i in range(len(claims)) if similarities[i][i].item() >= 0.35
            )
            risk_score = 1.0 - (verified_count / len(claims))
            return verified_count, len(claims), risk_score
        except Exception as e:
            logger.error(f"Verification Error: {e}")
            return 0, len(claims), 1.0

    def analyze_precedent_chain(self, primary_case_names: list) -> dict:
        chain_data = {}
        for name in primary_case_names[:1]:
            full_text = get_full_text_for_source(name)
            if not full_text:
                continue
            citations = find_citations(full_text)
            results = []
            for cit in citations[:1]:
                chain = self.precedent_prompt | self.llm | StrOutputParser()
                out = (
                    clean_and_parse_json(
                        chain.invoke(
                            {
                                "primary_case_text": full_text[:300],
                                "cited_case_text": cit,
                            }
                        )
                    )
                    or {}
                )
                results.append(
                    {
                        "cited_case": cit,
                        "relationship": out.get("relationship", "Unknown"),
                        "justification": out.get("justification", "N/A"),
                    }
                )
            chain_data[name] = results
        return chain_data

    def ask_api(self, query_text: str, chat_history: list = []):
        start_time = time.time()

        history_context = ""
        if chat_history:
            history_context = (
                "PREVIOUS HISTORY:\n"
                + "\n".join(
                    [f"{m['role'].upper()}: {m['text']}" for m in chat_history[-3:]]
                )
                + "\n\n"
            )

        tags_res = self.classify_query(query_text)
        tags = tags_res.get("tags", []) if tags_res else []

        context, source_names = self.retrieve_with_metadata(
            f"{', '.join(tags)}: {query_text}"
        )
        precedent_data = self.analyze_precedent_chain(source_names)

        full_context = f"{history_context}LAWS:\n{context}\n\nPRECEDENT:\n{json.dumps(precedent_data)}"

        chain = self.rag_with_claims_prompt | self.llm | StrOutputParser()
        llm_out = (
            clean_and_parse_json(
                chain.invoke({"context": full_context, "question": query_text})
            )
            or {}
        )

        verified, total, risk = self.verify_claims(llm_out.get("claims", []))

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        logger.info(f"\n⚡ REQUEST COMPLETE in {duration} seconds ⚡\n")

        return {
            "answer": llm_out.get(
                "answer", "I encountered an error analyzing your request."
            ),
            "risk_score": risk,
            "verified_claims": verified,
            "total_claims": total,
            "precedent_data": precedent_data,
            "tags": tags,
            "execution_time": duration,
        }

# Initialize Engine Once
engine = None
logger.info("Loading Legal AI Engine... (This may take a minute)")
try:
    engine = QueryEngine()
    logger.info("Engine Loaded Successfully.")
except Exception as e:
    logger.critical(f"CRITICAL ERROR LOADING ENGINE: {e}")
    traceback.print_exc()

@csrf_exempt
def ask(request):
    if request.method == "POST":
        if not engine:
            logger.error("QueryEngine not initialized, cannot process ask request.")
            return JsonResponse({"error": "QueryEngine not initialized."}, status=500)

        try:
            data = json.loads(request.body)
            query_text = data.get("query")
            chat_history = data.get("history", [])

            response_data = engine.ask_api(query_text, chat_history)
            return JsonResponse(response_data)
        except json.JSONDecodeError:
            logger.error("Invalid JSON received for ask request.")
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.exception("Error processing ask request:")
            return JsonResponse({"error": str(e)}, status=500)
    logger.warning(f"Unsupported method {request.method} for ask endpoint.")
    return JsonResponse({"error": "Only POST requests are supported"}, status=405)

@csrf_exempt
def status(request):
    return JsonResponse({"status": "running", "engine_loaded": bool(engine)})