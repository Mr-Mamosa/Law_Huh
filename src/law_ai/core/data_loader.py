import pypdf
from pathlib import Path
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# --- Configuration ---
# Assuming this file is in src/law_ai/core, the project root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def extract_text_from_pdf(pdf_path):
    """Extracts text from all pages of a PDF file."""
    try:
        pdf_reader = pypdf.PdfReader(pdf_path, strict=False)
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "
"
        return full_text
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"


def process_ipc():
    """Extracts text from the Indian Penal Code PDF and saves it to a text file."""
    print("--- Processing Indian Penal Code ---")
    ipc_pdf_path = DATA_DIR / "Indian_Penal_Code_1860.pdf"
    output_text_path = DATA_DIR / "ipc_text.txt"

    if output_text_path.exists():
        print(f"{output_text_path} already exists. Skipping IPC processing.")
        return

    if not ipc_pdf_path.exists():
        print(f"Error: IPC PDF not found at {ipc_pdf_path}.")
        return

    print(f"Extracting text from {ipc_pdf_path}...")
    ipc_text = extract_text_from_pdf(ipc_pdf_path)

    if ipc_text and "Error reading" not in ipc_text:
        print(f"Successfully extracted text. Writing to {output_text_path}...")
        output_text_path.write_text(ipc_text, encoding="utf-8")
        print(
            f"IPC file created successfully. Size: {output_text_path.stat().st_size} bytes"
        )
    else:
        print(f"Failed to extract substantial text from IPC PDF. Details: {ipc_text}")


def extract_and_format_pdf(pdf_file_str, judgments_path_str):
    """
    Worker function for multiprocessing pool. Takes string paths because
    multiprocessing can struggle with passing complex Path objects directly.
    """
    pdf_file = Path(pdf_file_str)
    judgments_path = Path(judgments_path_str)
    text = extract_text_from_pdf(pdf_file)
    doc = {"source": str(pdf_file.relative_to(judgments_path)), "text": text}
    return json.dumps(doc) + "
"


def process_sc_judgments():
    """Finds all SC judgment PDFs, extracts their text, and saves them to a .jsonl file using PARALLEL processing."""
    print("
--- Processing Supreme Court Judgments (Parallelized) ---")
    judgments_path = DATA_DIR / "supreme_court_judgments"
    output_jsonl_path = DATA_DIR / "sc_judgments_text.jsonl"

    if output_jsonl_path.exists():
        print(f"{output_jsonl_path} already exists. Skipping SC judgments processing.")
        return

    if not judgments_path.exists():
        print(f"Error: Supreme Court judgments folder not found at {judgments_path}.")
        return

    print("Finding all PDF files in all year directories (case-insensitive)...")
    pdf_files = list(judgments_path.rglob("*.[pP][dD][fF]"))
    print(f"Found {len(pdf_files)} PDF files to process.")

    if not pdf_files:
        print("No PDF files found. Please check the directory and file extensions.")
        return

    args_list = [(str(pdf_file), str(judgments_path)) for pdf_file in pdf_files]

    num_processes = cpu_count()
    print(f"Starting parallel processing with {num_processes} CPU cores...")

    try:
        with Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.starmap(extract_and_format_pdf, args_list),
                    total=len(pdf_files),
                    desc="Processing SC Judgments (Parallel)",
                )
            )
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        return

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(results)

    print(f"Finished processing Supreme Court judgments.")
    print(
        f"Output saved to {output_jsonl_path}. Size: {output_jsonl_path.stat().st_size} bytes"
    )

def process_all_data():
    process_ipc()
    process_sc_judgments()

if __name__ == "__main__":
    process_all_data()
