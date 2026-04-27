## Jailbreak Attack 

This repository contains the code, experimental results, and demonstration videos for conducting jailbreak attack evaluations on four large language models discussed in the paper.

## Overview

This pipeline classifies input instructions, generates adversarial prompts through a series of LLM-assisted steps, and evaluates responses using a two-stage judge (harm detection + relevance scoring). It targets four major LLMs: **GPT-5.3**, **Claude Sonnet 4.6**, **Gemini 3 Flash**, and **Grok 4.2**.

---

## Project Structure

```
.
├── common.py                    # Shared utilities: chat, judge, classifier, refiner
├── tutorial_pipeline.py         # Main entry point — runs the full attack pipeline
├── non_tutorial_pipeline.py     # Non-tutorial branch pipeline
├── harm.json                    # Input instructions
└── result/
    ├── claude-sonnet-4-6.json   # Jailbreak results on Claude Sonnet 4.6
    ├── gemini-3-flash.json      # Jailbreak results on Gemini 3 Flash
    ├── gpt_5.3.json             # Jailbreak results on GPT-5.3
    ├── grok-4.2.json            # Jailbreak results on Grok 4.2
├── Claude.mp4                   # Partial jailbreak demo — Claude
├── GPT.mp4                      # Partial jailbreak demo — GPT
├── Gemini.mp4                   # Partial jailbreak demo — Gemini
└── Grok.mp4                     # Partial jailbreak demo — Grok
```

---

## Getting Started

### Configuration

Edit the config section at the top of `common.py`:

```python
HELPER_MODEL    = "gpt-5.3-chat-latest"
HELPER_BASE_URL = "https://your-api-endpoint/v1"
HELPER_API_KEY  = "your-api-key"

TARGET_MODEL    = "gpt-5.3-chat-latest"
TARGET_BASE_URL = "https://your-api-endpoint/v1"
TARGET_API_KEY  = "your-api-key"

JUDGE_MODEL     = "gpt-4o"
```

### Running the Pipeline

```bash
python tutorial_pipeline.py
```


## Results

Full jailbreak results for each target model are saved in the `result/` directory:

| File | Target Model |
|------|-------------|
| `claude-sonnet-4-6.json` | Anthropic Claude Sonnet 4.6 |
| `gemini-3-flash.json` | Google Gemini 3 Flash |
| `gpt_5.3.json` | OpenAI GPT-5.3 |
| `grok-4.2.json` | xAI Grok 4.2 |

Each file follows the format:

```json
[
  {
    "index": 1,
    "original_instruction": "...",
    "Jailbreak_prompt": "...",
    "LLM_response": "..."
  }
]
```

---

## Demo Videos

The following videos are recorded demonstrations of partial jailbreak attacks on each model:

| File | Target Model |
|------|-------------|
| `Claude.mp4` | Anthropic Claude Sonnet 4.6 |
| `GPT.mp4` | OpenAI GPT-5.3 |
| `Gemini.mp4` | Google Gemini 3 Flash |
| `Grok.mp4` | xAI Grok 4.2 |

---

## Disclaimer

This repository is intended **for academic research purposes only**. All experiments should be conducted in controlled environments with appropriate authorization. The authors do not condone any misuse of this pipeline against production systems or real users.
