# info290 — User Story RAG System

A RAG (Retrieval-Augmented Generation) system that ingests multimodal product research artifacts (user interviews, meeting transcripts, PRDs, Jira issues) and generates structured product outputs — user stories and pain points — grounded in real source material.

---

## Data Setup

Large data files are not committed to this repo. Download the full `data/` folder from Google Drive and place it at the root of the project:

**[Download data/ folder from Google Drive](https://drive.google.com/drive/folders/1Qhzc7-CM2NnLwGqKjpfNkv5UUMj7Iyyi)**

Your local structure should look like this after downloading:

```
data/
├── audio/
│   ├── ami/               # AMI Meeting Corpus (4.17 GB, 14 parquet files)
│   └── mediasum/          # MediaSum transcripts (1.51 GB, 3 zip files)
└── documentation/
    ├── jira/              # JOSSE Jira Dataset (193 MB, zip)
    └── user stories/      # Mendeley User Stories Dataset (22 txt files)
```

### Dataset Sources

| Dataset | Source | Size | Format |
|---|---|---|---|
| AMI Meeting Corpus | [HuggingFace — edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami) | 4.17 GB | Parquet |
| MediaSum | [HuggingFace — ccdv/mediasum](https://huggingface.co/datasets/ccdv/mediasum) | 1.51 GB | ZIP |
| JOSSE Jira Dataset | [Zenodo — record 7022735](https://zenodo.org/records/7022735) | 193 MB | ZIP (SQLite + CSV) |
| User Stories (Mendeley) | [Mendeley Data — Lucassen et al.](https://data.mendeley.com/datasets/7zbk8zsd8y/1) | ~1 MB | TXT (Connextra format) |

> **Note:** AMI train set is partial (5 of 42 shards). Full corpus is ~35 GB and can be downloaded directly from HuggingFace if needed.

---

## Project Structure

```
.
├── CLAUDE.md              # Project context for Claude Code
├── README.md
├── data/
│   ├── audio/
│   ├── documentation/
│   └── download_agent.py  # Agent script used to acquire datasets
```

---

## Setup

```bash
pip install anthropic datasets huggingface_hub
```
