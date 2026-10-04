# Sethu: Ideation

*Sethu (bridge): connecting citizens to the government schemes they are entitled to.*

## 1. Problem
Millions of eligible people never claim benefits. Scheme information is spread across many portals and PDFs, written in formal English or Hindi, and hard to match to a person's own situation. People depend on middlemen or on luck.

## 2. Users
| Persona | Situation | What they need |
|---|---|---|
| Farmer, Kannada/Telugu speaker | Has 2 acres, doesn't know which schemes apply | "What can I get?" in their own language, by voice |
| Young woman, small-town Tamil Nadu | Wants a self-employment loan or scholarship | Eligibility, documents, where to apply |
| Field helper (ASHA worker, CSC operator, volunteer) | Helps many people daily | Fast, citable answers |

## 3. Core idea
A user describes their situation in their own language, by text or voice. Sethu retrieves matching scheme documents and answers with:
1. **Which schemes apply** and why (eligibility reasoning)
2. **Documents needed**
3. **How and where to apply**
4. **Source citation**: the official page or PDF the answer came from
5. **"I'm not sure"** when the sources don't support an answer

## 4. MVP scope (v1)
- Corpus: a curated set of central schemes plus one or two states, not everything
- 8 languages: English, Hindi, Kannada, Telugu, Tamil, Malayalam, Bengali, Marathi
- Web app with chat, voice input, and read-aloud
- Cited answers and a visible "last updated" date per source
- Out of scope for v1: form filling, application submission, user accounts, WhatsApp

## 5. Architecture (hybrid)
```
Official docs -> clean + chunk -> multilingual embeddings (local) -> vector store (local)
User query (any language, text/voice)
   -> embed -> retrieve top-k (+ metadata filters: state, category)
   -> Claude API: answer in the user's language, using only retrieved chunks, with citations
   -> UI: answer + sources + read-aloud
```
| Layer | Proposed choice | Why |
|---|---|---|
| Embeddings | BGE-M3 or multilingual-e5 (local) | Cross-lingual: Kannada query can match an English document |
| Vector store | Chroma or Qdrant (local) | Free, simple, supports metadata filters |
| LLM | Claude via API | Strong multilingual generation and grounded answering |
| Voice | Browser Web Speech API for MVP; Bhashini as an upgrade | Free to start; Bhashini covers Indian languages better |
| Backend | FastAPI | |
| Frontend | Next.js or plain React | |

**Key design decision:** index the official documents in their original language, embed them cross-lingually, and generate the answer in the user's language. This avoids translating the whole corpus.

## 6. Data sources (to verify)
- myScheme.gov.in (aggregates central and state schemes)
- Individual scheme portals and guideline PDFs (PM-KISAN, Ayushman Bharat, PMAY, scholarships, etc.)
- State government portals

Check each site's terms of use and robots.txt before scraping. Prefer official APIs or datasets where they exist.

## 7. Risks and guardrails
| Risk | Mitigation |
|---|---|
| Wrong eligibility advice | Answer only from retrieved text; cite; refuse when the evidence is weak |
| Stale data (schemes change) | Store a fetch date per document; show it; schedule re-ingestion |
| Weak quality in low-resource languages | Build an evaluation set per language before launch |
| Scraping or legal issues | Official sources only; respect terms |
| Users trusting it too much | Clear "verify at the office or portal" notice with the official link |
| Privacy | No accounts or personal data stored in v1; don't log raw queries by default |

## 8. Evaluation plan
- Build a set of about 100 real-style questions with known correct schemes
- Metrics: retrieval hit rate, answer faithfulness (is every claim in the sources?), citation correctness, refusal on unanswerable questions
- Run it per language to catch quality gaps early

## 9. Build phases
1. **Data:** choose 15 to 25 schemes, collect and clean the documents
2. **RAG core:** ingestion, chunking, embeddings, retrieval, grounded answers (English first)
3. **Multilingual:** add the other 7 languages and the evaluation set
4. **Web app:** chat UI, voice in and out, citations
5. **Harden:** guardrails, stale-data handling, user testing with real people

## 10. Open questions
- Which state or states for v1? (Sets the state-scheme corpus)
- Is this for a course or hackathon deadline, or a real deployment?
- Who will test the non-English answers? Native speakers are needed for quality checks.
- Name and branding: keep "Sethu"?

## 11. Expanded domains: income tax and entrepreneurship
Sethu covers three domains as separate modules on one RAG core:

| Domain | Examples | Answer type |
|---|---|---|
| Schemes | PM-KISAN, Ayushman Bharat, scholarships | Retrieval + eligibility explanation |
| Income tax | ITR form choice, deductions, deadlines, notices, regime comparison | Retrieval + deterministic calculation |
| Entrepreneurship | Udyam/MSME, GST basics, Mudra, PMEGP, Startup India, CGTMSE | Retrieval + step-by-step checklists |

Design rules:
- **Tax arithmetic is done by code, not the LLM.** Slabs, rebates and deductions live in a calculator tool that Claude calls. RAG explains rules; code computes numbers.
- **Versioned tax data.** Every tax chunk carries financial year / assessment year and law version metadata. The retriever filters on it. Re-ingest after each Budget.
- **Explain, don't advise.** Describe what the rules say; do not recommend investments or tax structuring. Show a "consult a CA" notice for complex cases.
- **Domain routing.** Classify each question (or use metadata filters) and show the user which domain answered.
- **Workflows for entrepreneurs.** Output ordered checklists (e.g. Udyam registration, then GST if the threshold applies, then loan schemes).

Suggested order: schemes (base RAG), then entrepreneurship (reuses the pipeline), then tax (needs calculator + version handling).

## 12. Decisions: coverage and languages
- **Geography:** central schemes plus the 15 most populous states (Census 2011, post-bifurcation): UP, Maharashtra, Bihar, West Bengal, MP, Tamil Nadu, Rajasthan, Karnataka, Gujarat, Andhra Pradesh, Odisha, Telangana, Kerala, Jharkhand, Assam.
- **Deadline:** none.
- **Verified languages (owner can review):** Kannada, Hindi, Tamil, Telugu, Marathi, English.
- **Unverified (ship as beta):** Malayalam, Bengali. Need an outside reviewer.
- **Gap:** Gujarat, Odisha, Assam need Gujarati, Odia, Assamese, which are not yet in scope.

Rollout tiers:
| Tier | Coverage |
|---|---|
| 0 | Central schemes, tax, MSME/entrepreneurship |
| 1 | Karnataka, Maharashtra, Tamil Nadu, Telangana, Andhra Pradesh |
| 2 | UP, Bihar, MP, Rajasthan, Jharkhand (Hindi) |
| 3 | West Bengal, Kerala, Gujarat, Odisha, Assam |

Every chunk carries `state`, `domain`, `language`, `source_url` and `fetched_on` metadata, so a new state is a data task, not a code change.
