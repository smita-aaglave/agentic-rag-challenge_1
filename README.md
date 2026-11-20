# Agentic RAG Challenge — SkillBridge Student Academic Intelligence

This repository contains the Kaggle notebook for the Agentic RAG challenge and a starter Streamlit dashboard.

## Project structure

```text
agentic-rag-challenge/
├── app/
│   └── app.py
├── notebooks/
│   └── agentic-rag-challenge.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Run the Kaggle notebook (full Agentic RAG pipeline)

The notebook expects the competition data to be available in Kaggle at:

`/kaggle/input/competitions/gen-ai-project-agentic-rag/Gen AI Project 3`

It reads the `Dataset`, `Query Files`, and `Policy Documents` folders. These challenge data files are not included in this repository.

Add the required API credentials using Kaggle Secrets (do not paste keys into the notebook):
- `GROQ_API_KEY_1` (and optional additional keys through `GROQ_API_KEY_8`)
- `GOOGLE_API_KEY`

Open `notebooks/agentic-rag-challenge.ipynb` in Kaggle, attach the competition dataset, enable the secrets, then run the notebook cells in order. The final cell generates `submission.csv`.

## 2. Run the dashboard locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

Upload the CSV files from the challenge `Dataset` folder in the sidebar. The dashboard currently supports dataset previews and student lookup/attendance counts.

**Important:** the Streamlit dashboard is a UI/data-exploration starter. The full Gemini embeddings + FAISS retrieval + Groq tool-calling agent remains in the Kaggle notebook and is not yet connected to the dashboard's question box.

## 3. GitHub upload

Extract the ZIP, open the extracted folder in VS Code, then run:

```bash
git init
git add .
git commit -m "Add Agentic RAG challenge notebook and dashboard"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your own GitHub repository URL.

## Security

Never commit API keys, Kaggle credentials, private datasets, or generated submission files. The `.gitignore` excludes common local secrets, data files, and generated outputs.
