# Sethu

Multilingual, source-cited answers about Indian government schemes. See [IDEATION.md](IDEATION.md) for the plan.

**Status: first working prototype.** Covers 740 schemes (central plus Karnataka, Maharashtra, Tamil Nadu, Telangana, Andhra Pradesh). Tax and MSME/entrepreneurship modules are not built yet.

## Run it
```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env        # add GROQ_API_KEY (and SETHU_LLM_PROVIDER=groq), or ANTHROPIC_API_KEY

python -m sethu.adapters.myscheme data/raw/myscheme.jsonl                  # download + clean the scheme PDFs (~10 min)
python -m sethu ingest data/raw/myscheme.jsonl --states "Karnataka,Maharashtra,Tamil Nadu,Telangana,Andhra Pradesh"   # ~45 min on CPU
uvicorn sethu.api:app                                                      # web app at http://localhost:8000
python -m sethu ask "free eye check-up" --state Telangana                  # CLI
python -m sethu.evaluate eval/questions.jsonl --modes dense hybrid dense+rewrite hybrid+rewrite
pytest
```

## How it works
1. **Ingest** (`sethu/adapters/myscheme.py`, `chunking.py`, `embed.py`, `store.py`): scheme PDFs are cleaned, split by section, embedded with `multilingual-e5-base` (local, CPU) into Chroma.
2. **Retrieve** (`retrieve.py`): non-English questions are translated to an English search query by the LLM (`rewrite.py`); dense search on both versions plus BM25 on the English text, fused with reciprocal rank fusion. State filter returns that state's schemes plus central ones.
3. **Context** (`context.py`): the top two schemes are expanded to their eligibility, benefits, application and document sections, because answering from fragments produced wrong application steps.
4. **Answer** (`answer.py`): the LLM answers only from the numbered sources, in the requested language, with `[n]` citations, and says it is unsure when the sources do not cover the question.
5. **UI** (`sethu/web/index.html`): state and language pickers, cited sources, voice input (browser speech recognition; support and language coverage vary by browser) and read-aloud.
6. **Read-aloud** (`tts.py`, `POST /tts`): speech is generated on the server and played as audio, so it works on any device regardless of installed voices. The language is detected from the answer's script and spoken with a neural voice for all 8 languages. It uses the unofficial `edge-tts` route to Microsoft's voices (no key, but the answer text leaves your server); switch to Bhashini, Azure or Google Cloud TTS for a real deployment.

## Evaluation (47 answerable + 5 out-of-scope questions, `eval/questions.jsonl`)
| mode | hit@1 | hit@3 | non-English hit@1 |
|---|---|---|---|
| dense | 0.81 | 0.96 | 0.56 |
| hybrid | 0.83 | 0.96 | 0.56 |
| dense + rewrite | 0.94 | 1.00 | 0.89 |
| **hybrid + rewrite (default)** | **0.98** | **1.00** | **0.94** |

Caveats: small set written by the developer from the same corpus, so it is optimistic. The non-English questions are the developer's own translations. n is 3 to 5 per language, so per-language numbers are indicative only.

## Data provenance: development use only
The current corpus is a third-party scrape of myScheme.gov.in (Hugging Face `Navraj025/gov_myscheme`). myScheme's [Terms of Use](https://www.myscheme.gov.in/terms-of-use) say "The use of any software (e.g. bots, scraper tools) or other automatic devices to access, monitor, or copy the platform pages is prohibited unless expressly authorized by the myScheme in writing." (its robots.txt allows crawling, but the terms govern). So this data must not be published or deployed, and **no scraper of myScheme should be built without written authorization**. A public version needs data from sources whose terms allow it, each stored with its official URL, licence and retrieval date. Routes to check: written permission from myScheme/NeGD, an API Setu partner account, data.gov.in datasets under the Government Open Data License (reuse allowed with attribution), and primary documents (ministry/department guidelines, India Code, Income Tax Department) after reading each site's own terms.

## Known limitations
- **Data is an undated community scrape** of myScheme (Hugging Face `Navraj025/gov_myscheme`). The "data from" date in the UI is the upload date, not when a scheme was last updated. Schemes change; always verify on the official site. Karnataka (22) and Telangana (15) look incomplete. Official myScheme/API Setu access would fix coverage and freshness.
- **Spot-check against the live myScheme site (5 schemes, 2026-10-04) found our copy is out of date.** `source_url` is built from the PDF filename (`myscheme.gov.in/schemes/<slug>`). 2 of 5 slugs load (`klp`, `cs-tn`); 3 return "Page not found" (`kv`, `ysrrb`, `mdckra`), so those schemes were renamed or removed and our "official page" link is dead. For `klp` (Kalyana Lakshmi) the live page gives one parental income limit of ₹2,00,000, while our copy lists ₹1,50,000 for rural BC/EBC. Treat every figure as unverified until the data is refreshed from an official source.
- **Refusal is mostly the LLM's job.** Embedding distance cannot separate answerable from unrelated questions with e5 (ranges overlap), so the distance cutoff (0.25) removes only the clearly unrelated.
- **Tamil answer quality is weaker** with the current model (mixed English words, occasional garbled text). Kannada/Hindi/Telugu/Marathi looked reasonable; Malayalam and Bengali are untested by a native speaker (shown as beta).
- **Free Groq tier allows ~8,000 tokens/minute**, which is about one to two answers a minute. Fine for a demo, not for real traffic.
- **Answers vary run to run.** The same Kannada question once included the full application steps and once only the documents.
- **Voice input still depends on the browser** (Chrome and Edge support it; others may not, and Indian-language accuracy varies). A server-side speech-to-text step (for example Whisper on Groq, or Bhashini) would remove that dependency.
- Not built yet: tax calculator, MSME/entrepreneurship data, states beyond Tier 1, Gujarati/Odia/Assamese, an answer-faithfulness evaluation.

## Layout
`sethu/` package, `tests/` (36 offline tests with a fake embedder and stub LLM), `eval/` question set, `data/` (git-ignored). `tests/fixtures/schemes.jsonl` is **synthetic test data, not real schemes**.
