"""Convert the Hugging Face myScheme PDF dump (Navraj025/gov_myscheme) to normalized JSONL.

Each PDF is a printed myScheme web page: sign-in boilerplate, then the state or ministry,
the scheme title, tag words, and sections whose headings are glued to the following text.
"""

import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import ftfy

from sethu.schema import CENTRAL, SchemeRecord

DATASET = "Navraj025/gov_myscheme"
SCHEME_URL = "https://www.myscheme.gov.in/schemes/{slug}"  # slug = PDF filename; unverified

STATES = sorted(
    [
        "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
        "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
        "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan",
        "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
        "West Bengal",
    ],
    key=len,
    reverse=True,
)

# (heading as printed, section key). "Sources And References" and "Feedback" end the body.
HEADINGS = [
    ("Details", "details"),
    ("Benefits", "benefits"),
    ("Eligibility", "eligibility"),
    ("Exclusions", "exclusions"),
    ("Application Process", "application_process"),
    ("Documents Required", "documents_required"),
    ("Frequently Asked Questions", "faq"),
]
END_HEADINGS = ["Sources And References", "Feedback"]

# What may follow a heading: a capital, digit, bullet, quote or rupee sign, directly or after
# one space. Lookahead only, so the first character stays part of the section text.
_START_CHAR = "[A-Z0-9•●'\"“‘(₹]"
_CONTENT_START = rf"(?={_START_CHAR}|\s{_START_CHAR})"


def clean(text: str) -> str:
    # A zero-width space after a broken quote (â€) stops ftfy from repairing it, so drop it first.
    text = ftfy.fix_text(text.replace("​", "")).replace("﻿", "")
    # A closing curly quote (UTF-8 E2 80 9D) arrives with its last byte lost, leaving "â€".
    # The other E2 80 xx punctuation marks are repaired by ftfy, so what remains is this quote.
    text = text.replace("â€", "”")
    text = re.sub(r"\s+", " ", text).strip()
    # The PDFs drop the space after a full stop ("India.The applicant"). Restoring it needs
    # a capital followed by a lowercase letter so abbreviations like "Ph.D." are left alone.
    return re.sub(r"(?<=[a-z0-9)])\.(?=[A-Z][a-z])", ". ", text)


def parse(slug: str, raw_text: str, fetched_on: date) -> tuple[SchemeRecord | None, str]:
    """Return (record, head); head is the unparsed state/ministry text, for diagnostics."""
    text = clean(raw_text)
    title = text.split("Are you sure you want to sign out?")[0].strip()

    marker = re.search(r"Check Eligibility\s*", text)
    if not title or not marker:
        return None, ""
    body = text[marker.end() :]

    state, head = CENTRAL, body[:60]
    for s in STATES:
        if body.startswith(s):
            state = s
            break

    # The page header is printed a second time after the content; everything from there is junk.
    repeat = re.search(r"Check Eligibility", body)
    if repeat:
        body = body[: repeat.start()]
        body = re.sub(r"Apply Now$", "", body)

    # Locate each heading in order. Headings are sometimes glued to the text around them and
    # sometimes separated by a space (line wrapping), so we accept either, but require the
    # next text to start like a sentence or list item, which rejects "Account Details, etc.".
    found: list[tuple[str, int, int]] = []
    pos = 0
    for heading, key in HEADINGS:
        m = re.compile(rf"{re.escape(heading)}{_CONTENT_START}").search(body, pos)
        if m:
            found.append((key, m.start(), m.end()))
            pos = m.end()
    if not found:
        return None, head

    end = len(body)
    for h in END_HEADINGS:
        m = re.compile(rf"{re.escape(h)}(?:{_CONTENT_START}|$)").search(body, pos)
        if m:
            end = min(end, m.start())

    sections = {}
    for i, (key, _, text_start) in enumerate(found):
        text_end = found[i + 1][1] if i + 1 < len(found) else end
        content = body[text_start:text_end].strip()
        if content:
            sections[key] = content

    if sum(len(v) for v in sections.values()) < 200:
        return None, head
    record = SchemeRecord(
        id=slug,
        name=title,
        state=state,
        domain="scheme",
        language="en",
        source_url=SCHEME_URL.format(slug=slug),
        fetched_on=fetched_on,
        sections=sections,
    )
    return record, head


def main(out_path: str, limit: int | None = None) -> None:
    from huggingface_hub import HfApi, snapshot_download
    from pypdf import PdfReader

    last_modified = HfApi().dataset_info(DATASET).last_modified
    fetched_on = last_modified.date() if last_modified else date.today()
    root = Path(
        snapshot_download(
            DATASET, repo_type="dataset", allow_patterns=["text_data/*.pdf"],
            ignore_patterns=["* copy.pdf"],
        )
    )
    pdfs = sorted((root / "text_data").glob("*.pdf"))[:limit]

    seen: set[str] = set()
    kept = skipped = dupes = 0
    heads_central: dict[str, int] = {}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for n, pdf in enumerate(pdfs, 1):
            try:
                raw = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
            except Exception as e:  # corrupt PDF: skip, don't abort the whole run
                print(f"skip {pdf.name}: {e}", file=sys.stderr)
                skipped += 1
                continue
            record, head = parse(pdf.stem, raw, fetched_on)
            if record is None:
                skipped += 1
                continue
            digest = hashlib.md5(json.dumps(record.sections, sort_keys=True).encode()).hexdigest()
            if digest in seen:
                dupes += 1
                continue
            seen.add(digest)
            if record.state == CENTRAL:
                key = head[:25]
                heads_central[key] = heads_central.get(key, 0) + 1
            f.write(record.model_dump_json() + "\n")
            kept += 1
            if n % 200 == 0:
                print(f"{n}/{len(pdfs)}", file=sys.stderr)
    print(f"kept={kept} skipped={skipped} duplicate_content={dupes} -> {out}")
    top = sorted(heads_central.items(), key=lambda kv: -kv[1])[:15]
    print("most common 'central' heads (check for unrecognised states):", top)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
