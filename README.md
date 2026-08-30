# SIH26189 — AI-Powered Criminal Network Analysis System

**Smart India Hackathon 2026 | Problem Statement: Criminal Network Analysis**

An integrated investigation intelligence platform combining graph neural networks, temporal analysis, document intelligence, and case management for criminal network analysis.

> ⚠️ **Synthetic/Anonymized Demonstration Dataset**
> All data in this system is synthetically generated. No real criminal records, FIRs, or personally identifiable information is used. Scores indicate network anomaly / investigation priority only — they do not predict guilt.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INVESTIGATOR DASHBOARD                         │
│                    investigator.html (Single-Page App)                  │
├──────────┬──────────────┬───────────────────┬──────────────────────────┤
│  Graph   │  Document    │  Case Hub         │  Graph                   │
│  View    │  Intelligence│  (Case Mgmt)      │  Panel                   │
│          │              │                   │  (Entity Detail +        │
│  D3.js   │  FIR Viewer  │  Cases            │   Explainability +       │
│  Force-  │  NER/RE      │  Evidence         │   Model Info +           │
│  Directed│  BNS/IPC     │  Notes            │   Evaluation)            │
│  Graph   │  Extraction  │  Timeline         │                          │
│          │  Evaluation  │  Report Gen       │                          │
├──────────┴──────────────┴───────────────────┴──────────────────────────┤
│                        DATA LAYER (JSON)                               │
│  gnn_results.json | events.json | temporal.json | nlp_results.json    │
│  synthetic_firs.json | bns_sections.json | ipc_legacy_sections.json   │
│  case_data.json | nlp_ground_truth.json                               │
├────────────────────────────────────────────────────────────────────────┤
│                     OFFLINE PIPELINE (Python)                          │
│  Crime (1).ipynb → src/nlp/extractor.py → src/nlp/evaluate.py        │
│  GraphSAGE Training | Feature Engineering | NER/RE Extraction         │
└────────────────────────────────────────────────────────────────────────┘
```

## Features

### Phase 1 — Network Analysis & Explainability
- **Interactive force-directed graph** with 220 entities and 1,571 relationships
- **GraphSAGE-based anomaly detection** with hybrid scoring formula
- **Explainability panel** with per-entity risk breakdown
- **Entity search, relationship/community/risk-band filters**
- **Top-risk entities** ranked by anomaly score

### Phase 2 — Temporal & Relationship Investigation
- **Activity timeline** with burst detection (1.5σ threshold)
- **2-hop subgraph expansion** for entity neighborhoods
- **Shortest-path explorer** between any two entities
- **Community isolation** investigation mode
- **Case-scoped communication analysis**

### Phase 3 — Document Intelligence (NLP Pipeline)
- **12 synthetic FIR-style documents** with realistic Indian FIR structure
- **Rule-based NER extraction** (PERSON, LOCATION, DATE, LEGAL_SECTION, CASE_NUMBER)
- **Relationship extraction** (accused-of, witness-to, mentioned-together, associated-location)
- **BNS (Bharatiya Nyaya Sanhita 2023) & legacy IPC section extraction**
- **Transparent evaluation** with real calculated precision/recall/F1
- **Explainability** — every extraction shows supporting text and method

### Phase 4 — Case Management & Reporting
- **3 pre-seeded investigation cases** linked to real network entities
- **13 evidence findings** derived from actual graph/temporal/NLP analysis
- **Investigator notes** with localStorage persistence
- **Case-scoped timelines** (communication events filtered to case-relevant entities)
- **Investigation report generation** with copy-to-clipboard

### CAVIAR Network Explorer
- Real-world CAVIAR criminal dataset (107 individuals)
- Community structure and centrality analysis

---

## Methodology

### Graph Neural Network (GraphSAGE)

| Component | Details |
|-----------|---------|
| **Architecture** | GraphSAGE (Graph Sample and Aggregate) |
| **Input features** | 15 per node: degree, degree centrality, betweenness centrality, embedding, bet_c, burst_c, n_calls, n_messages, n_meetings, n_transactions, n_locations, active_days, max_daily_burst, unique_contacts, neighbour_avg_degree |
| **Hidden dimensions** | 32 → 8 (two-layer encoder) |
| **Training approach** | Reconstruction-based embedding (autoencoder paradigm) |
| **Anomaly scoring** | Hybrid formula combining embedding outlier detection, structural centrality, and temporal burst patterns |

### Hybrid Risk Scoring Formula

```
risk_score = 0.5 × embedding_outlier + 0.3 × betweenness_centrality + 0.2 × temporal_burst
```

- **Embedding outlier (0.5)**: How much the entity's learned representation deviates from the network norm
- **Betweenness centrality (0.3)**: The entity's role as an information bridge between communities
- **Temporal burst (0.2)**: Unusual spikes in communication activity

### NLP Pipeline

| Component | Approach |
|-----------|----------|
| **NER** | Rule-based regex patterns (no ML models) |
| **Relationship Extraction** | Role-based + contextual pattern matching |
| **Legal Section Extraction** | Pattern matching for BNS and IPC section references |
| **Evaluation** | Manual ground-truth annotations on 12 synthetic FIRs |

### Indian Legal Framework

- **Primary**: Bharatiya Nyaya Sanhita (BNS), 2023 — 48 commonly-referenced sections
- **Legacy**: Indian Penal Code (IPC), 1860 — 39 sections for older document compatibility
- **Source**: Official BNS text (Act No. 45 of 2023), effective 1 July 2024

---

## Evaluation

### Anomaly Detection Validation (Synthetic Planted Anomalies)

| Metric | Value |
|--------|-------|
| Total entities | 220 |
| Planted synthetic anomalies | 10 (P201–P210, Community 4) |
| Planted anomalies in top 15 ranks | **10/10** |
| Risk band assignment | All 10 assigned MODERATE (correct) |

> This validates that the GraphSAGE model successfully identifies planted anomalies through the hybrid scoring formula. This is a synthetic validation, not a real-world precision/recall measurement.

### NLP Entity Extraction (Real Calculated Metrics)

| Metric | Value |
|--------|-------|
| **Entity Precision** | 0.776 (97/125 extracted) |
| **Entity Recall** | 0.758 (97/128 ground truth) |
| **Entity F1** | 0.767 |

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| PERSON | 1.000 | 0.923 | 0.960 |
| LEGAL_SECTION | 1.000 | 1.000 | 1.000 |
| CASE_NUMBER | 1.000 | 1.000 | 1.000 |
| DATE | 0.621 | 1.000 | 0.766 |
| LOCATION | 0.320 | 0.229 | 0.267 |
| ORGANIZATION | 0.000 | 0.000 | 0.000 |

### NLP Relationship Extraction (Real Calculated Metrics)

| Metric | Value |
|--------|-------|
| **Relationship Precision** | 0.316 (24/76 extracted) |
| **Relationship Recall** | 0.750 (24/32 ground truth) |
| **Relationship F1** | 0.444 |

| Relationship Type | Precision | Recall | F1 |
|-------------------|-----------|--------|-----|
| accused-of | 1.000 | 1.000 | 1.000 |
| witness-to | 0.900 | 1.000 | 0.950 |
| mentioned-together | 0.500 | 0.333 | 0.400 |
| associated-location | 0.000 | 0.000 | 0.000 |

> Evaluation performed on synthetic demonstration data only. Does not represent real-world FIR processing performance. LOCATION and associated-location extraction are known limitations of rule-based approaches without NER models.

---

## Data Sources

| File | Contents | Size |
|------|----------|------|
| `data/gnn_results.json` | 220 entities, 1,571 relationships, anomaly scores | 192 KB |
| `data/events.json` | 2,500 communication events (60-day span) | 271 KB |
| `data/temporal.json` | Pre-computed temporal analytics, adjacency, burst detection | 283 KB |
| `data/nlp_results.json` | NLP extraction results + evaluation metrics | 148 KB |
| `data/synthetic_firs.json` | 12 synthetic FIR documents | 33 KB |
| `data/bns_sections.json` | 48 BNS sections with metadata | 19 KB |
| `data/ipc_legacy_sections.json` | 39 legacy IPC sections | 16 KB |
| `data/case_data.json` | 3 cases, 13 evidence, 6 notes, case-scoped timelines | 108 KB |
| `data/nlp_ground_truth.json` | Manual ground-truth annotations for evaluation | 38 KB |

---

## Setup

### Prerequisites
- A modern web browser (Chrome, Firefox, Edge)
- Python 3 (for running the NLP pipeline locally)

### Running the Dashboard

```bash
# Start a local server
python3 -m http.server 3000 --bind 0.0.0.0

# Open in browser
open http://localhost:3000
```

### Re-running the NLP Pipeline

```bash
# Run the extractor (generates data/nlp_results.json)
python3 src/nlp/extractor.py

# Run the evaluation (updates metrics in nlp_results.json)
python3 src/nlp/evaluate.py
```

---

## Project Structure

```
├── index.html                    # Landing page
├── investigator.html             # Main investigator dashboard (single-page app)
├── caviar_viz.html               # CAVIAR dataset explorer
├── Crime (1).ipynb               # Original Colab notebook (analysis pipeline)
├── data/
│   ├── gnn_results.json          # Graph analysis output
│   ├── events.json               # Communication events
│   ├── temporal.json             # Temporal analytics
│   ├── nlp_results.json          # NLP extraction + evaluation
│   ├── synthetic_firs.json       # Synthetic FIR documents
│   ├── bns_sections.json         # BNS legal sections
│   ├── ipc_legacy_sections.json  # Legacy IPC sections
│   ├── case_data.json            # Investigation cases + evidence
│   └── nlp_ground_truth.json     # Evaluation ground truth
├── src/nlp/
│   ├── extractor.py              # Rule-based NER + relationship extraction
│   └── evaluate.py               # Evaluation metrics calculator
└── README.md                     # This file
```

---

## Disclaimers

1. **All data is synthetic.** No real criminal records, FIRs, or personally identifiable information is used.
2. **Anomaly scores indicate investigation priority, not guilt.** A high score means the entity exhibits unusual network patterns worth investigating.
3. **NLP extraction is rule-based on synthetic data.** It does not represent real-world FIR processing capability.
4. **IL-TUR dataset** (referenced in notebook) contains annotated Indian court judgments and is not an FIR dataset. It was explored for research purposes only.
5. **BNS sections** are sourced from the Bharatiya Nyaya Sanhita, 2023 (Act No. 45 of 2023), effective 1 July 2024.

---

## Technologies

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: D3.js v7 (force-directed graph)
- **Data**: Pre-processed JSON (no runtime database)
- **NLP Pipeline**: Python 3 stdlib (regex-based, no ML dependencies)
- **Graph Analysis**: GraphSAGE (trained in Colab notebook)
- **Storage**: Browser localStorage for investigator notes

---

## License

This project is developed for Smart India Hackathon 2026 demonstration purposes.
