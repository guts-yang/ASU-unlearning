# Leak-Resistant Unlearning: Data Construction

Code for constructing multi-hop evaluation datasets used in our paper on leak-resistant machine unlearning.

## Overview

We construct multi-hop reasoning questions from two sources to evaluate whether LLM unlearning methods truly erase knowledge or merely suppress surface-level recall:

1. **MQuAKE Curation**: Generates multi-hop questions from MQuAKE-CF knowledge triples using 6 formal logical reasoning types.
2. **Books Curation**: Extracts knowledge from book text and generates single-hop + multi-hop questions via LLM-based pipelines.

## Multi-Hop Logical Types

| Type | Abbreviation | Formal Rule | Example |
|------|-------------|-------------|---------|
| Hypothetical Syllogism | HS | A→B, B→C ⊢ A→C | "X is citizen of Y" + "capital of Y is Z" → "What is the capital of X's country?" |
| Modus Tollens | MT | A→B ∧ ¬B ⊢ ¬A | "X plays for team Y" → "Which player would leave if team Y disbanded?" |
| Disjunctive Syllogism | DS | P∨Q ∧ ¬P ⊢ Q | "Either X or Z holds title" + "X does not" → "Who holds the title?" |
| Leibniz's Law | LL | x=y ∧ P(x) ⊢ P(y) | "X is also known as Y" → "What is Y's nationality?" |
| Modus Ponens | MP | ∀x(S→P) ∧ S(a) ⊢ P(a) | "X works at Y" + "Y is a type of Z" → "X works at what kind of institution?" |
| Conjunction Decomposition | CD | ∧ᵢ[P(xᵢ)∧Qᵢ(xᵢ)] ⊢ ∧ᵢP(xᵢ) | "Both X and Z are citizens of Y" → "Who else shares X's nationality?" |

## Project Structure

```
├── mquake_curation/          # MQuAKE multi-hop question generation
│   ├── run_curation.py       # Main pipeline entry point
│   ├── config.py                 # Configuration (API keys, models)
│   ├── data_loader.py            # Load MQuAKE-CF data
│   ├── llm_generator.py          # LLM-based question generation
│   ├── formatter.py              # Output formatting
│   └── matchers/                 # Logical type implementations
│       ├── chain_implication.py
│       ├── contrapositive_reasoning.py
│       ├── disjunctive_elimination.py
│       ├── identity_substitution.py
│       ├── instance_subsumption.py
│       └── multi_element_intersection.py
│
└── books_curation/           # Books knowledge extraction & QA generation
    ├── extract_viewpoints_books.py   # Step 1: Extract viewpoints from text
    ├── extract_knowledge.py          # Step 2: Extract knowledge triples
    ├── clean_knowledge.py            # Step 3: Clean & deduplicate knowledge
    ├── clean_viewpoints.py           # Step 4: Clean viewpoints
    ├── generate_multihop.py          # Step 5: Generate multi-hop questions
    ├── generate_and_rewrite.py       # Step 6: Rewrite & diversify questions
    └── supplement_questions.py       # Step 7: Supplement with additional questions
```

## Usage

### MQuAKE Curation

```bash
# Set your OpenAI-compatible API key
export OPENAI_API_KEY="your-key-here"

# Run curation pipeline
python mquake_curation/run_curation.py \
    --input_file /path/to/MQuAKE-CF-3k-v2.json \
    --output_dir /path/to/output
```

### Books Curation

The books pipeline runs sequentially:

```bash
# Step 1: Extract viewpoints from book text
python books_curation/extract_viewpoints_books.py \
    --input_file /path/to/book.txt \
    --output_file viewpoints.json \
    --api_key YOUR_API_KEY

# Step 2-4: Extract and clean knowledge
python books_curation/extract_knowledge.py --input_file viewpoints.json --output_file knowledge.json --api_key YOUR_API_KEY
python books_curation/clean_knowledge.py --input_file knowledge.json --output_file knowledge_clean.json --api_key YOUR_API_KEY
python books_curation/clean_viewpoints.py --input_file viewpoints.json --output_file viewpoints_clean.json --api_key YOUR_API_KEY

# Step 5-7: Generate multi-hop questions
python books_curation/generate_multihop.py --input_file knowledge_clean.json --output_file multihop.json --api_key YOUR_API_KEY
python books_curation/generate_and_rewrite.py --input_file multihop.json --output_file multihop_rewritten.json --api_key YOUR_API_KEY
python books_curation/supplement_questions.py --input_file multihop_rewritten.json --output_file multihop_final.json --api_key YOUR_API_KEY
```

## Requirements

- Python >= 3.9
- openai
- tqdm

## Data

- **MQuAKE-CF**: Available from the [MQuAKE repository](https://github.com/princeton-nlp/MQuAKE)
- **Books (MUSE-Books)**: Available from the [MUSE benchmark](https://github.com/jaechan-repo/muse_bench)
