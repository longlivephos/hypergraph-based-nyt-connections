# Hypergraph-Based Metrics for Semantic Ambiguity and Structural Overlap in NYT Connections

## Overview

This repository contains the implementation of the framework proposed in *Hypergraph-Based Metrics for Semantic Ambiguity and Structural Overlap in NYT Connections*.

The project introduces a mathematical approach for analyzing the structural difficulty of NYT Connections puzzles through hypergraph modeling. While puzzle difficulty is often assessed subjectively by players or puzzle designers, this work aims to quantify difficulty using measurable structural properties of the puzzle itself.

Each puzzle is represented as a 4-uniform hypergraph, where words are modeled as vertices and categories are modeled as hyperedges. Based on this representation, the framework introduces two structural metrics:

* **Semantic Ambiguity (AS)**, which measures the extent to which words participate in multiple plausible groupings.
* **Structural Overlap (SOS)**, which measures the similarity between correct categories and competing candidate categories.

Together, these metrics provide an objective and interpretable way to analyze ambiguity and difficulty in NYT Connections puzzles.

---

## Methodology

1. Load NYT Connections puzzle data.
2. Represent each puzzle as a hypergraph.
3. Compute semantic similarity using WordNet path similarity.
4. Generate candidate hyperedges using a coherence threshold.
5. Construct the final hypergraph.
6. Compute Ambiguity Score (AS).
7. Compute Structural Overlap Score (SOS).
8. Export puzzle-level and category-level results.

---

## Repository Structure

```text
.
├── connections_analysis.py
├── nyt_connections_reselected_40.csv
├── connections_results.csv
└── readme.md
```

---

## Requirements

* Python 3.10+
* pandas
* nltk

Install dependencies:

```bash
pip install pandas nltk
```

---

## Usage

Run the analysis for all puzzles:

```bash
python connections_analysis.py
```

Analyze a single puzzle:

```bash
python connections_analysis.py --single 1019
```

Specify a custom coherence threshold:

```bash
python connections_analysis.py --tau 0.12
```

---

## Output

The framework produces:

* Ambiguity Score (AS)
* Structural Overlap Score (SOS)
* Category-level AS and SOS
* Candidate hyperedge statistics
* Descriptive statistics
* CSV result files


