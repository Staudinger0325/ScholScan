# ScholScan
# 🔬 Paper Replication & Dataset Repository

This repository provides all the necessary materials for replicating the experiments in our paper, including the dataset, precomputed embeddings, evaluation scripts, and inference pipelines. It supports both **main results** and **retrieval-augmented generation (RAG)** experiments across **image** and **text** modalities.

---

## 📂 Repository Structure

### `data/`
Contains all dataset-related files.

- `origin_pdf/`: Raw PDF files before conversion to images.
- `final_image.json`: Image-based inputs used in main results.
- `final_image_rag.json`: Image-based inputs for RAG experiments (includes Oracle-used page numbers).
- `final_text.json`: Text-based inputs used in main results.
- `final_text_rag.json`: Text-based inputs for RAG experiments (includes Oracle-used page numbers).

### `images/`
All PDFs rendered into images at **300 DPI**, organized by document ID and page number.

### `text_OCR_page/`
OCR outputs of the original PDFs. Each JSON file is named by ID and contains a list of strings (one per page).

---

## 🧠 Embedding Results

### `embed_results/`

- `rag_images/`: Contains subfolders for each image-based retriever model:
  - `colpali-v1.3-hf/`, `colqwen2.5-v0.2/`, `VisRAG-Ret/`
- `rag_texts/`: Contains subfolders for each text-based retriever model:
  - `bge-m3/`, `bm25/`, `contriever-msmacro/`, `nv-embed-v2/`

Each subfolder contains embedding results per-question ID. Regardless of scoring method, all models store results under the field `cos_sim` for simplicity.

---

## 📈 Evaluation Scripts

### `evaluation/`

- `evaluation_api.py`: Calls GPT-4.1 for scoring model outputs in **main results**.
- `evaluation_vrag_api.py`: GPT-4.1 evaluation script for **VRAG-RL** results in RAG experiments.
- `evaluation_cal.py`: Aggregates GPT-4.1 scores into final evaluation metrics.
- `evaluation_volc_cal.py`: Similar to above, but tailored for outputs from **VolcEngine** APIs.

---

## 🚀 Inference Pipelines

### `inference/`

- `inference_images/`: Stores image-based model outputs.
- `inference_texts_oracle/`, `inference_texts_rag/`: Stores text-based model outputs (Oracle or RAG setting).
- Other subfolders follow the same naming logic: `modality_setting`.

---

## 🔄 Batch Conversion for VolcEngine

### `volc_batch/`

- `convert_to_volcengine.py`: Converts JSON to VolcEngine batch input format (`.jsonl`), including modifying image URLs to match cloud storage paths.
- `convert_back.py`: Converts VolcEngine outputs (`.jsonl`) back to standard JSON format.

**Note:** Image files must be pre-uploaded to the appropriate VolcEngine storage bucket.

---

## 🛠️ Utilities & Conversion

- `pdf2image_mp.py`: Multi-threaded PDF-to-image converter with `dpi=300`.
- `rag_statistic_.py`: Computes retrieval performance metrics for each RAG model.
- `utils.py`: Centralized storage of all prompts and utility functions.

---

## 📌 Notes

This github anonymous repo does not contain any detailed data for storage reason, please vist https://huggingface.co/datasets/0x7ajsncjansg/ScholScan to get access the data refered frontly.

