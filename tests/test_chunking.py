from datetime import date

from sethu.chunking import chunk_record
from sethu.schema import SchemeRecord


def _record(**sections):
    return SchemeRecord(
        id="s1",
        name="Test Scheme",
        state="Karnataka",
        source_url="https://example.invalid",
        fetched_on=date(2026, 1, 1),
        sections=sections,
    )


def test_each_chunk_carries_scheme_name_and_metadata():
    chunks = chunk_record(_record(eligibility="Farmers can apply."))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text.startswith("Test Scheme (Karnataka) - eligibility")
    assert (c.state, c.section, c.fetched_on) == ("Karnataka", "eligibility", "2026-01-01")


def test_empty_sections_are_skipped():
    assert chunk_record(_record(eligibility="  ", documents="")) == []


def test_long_section_is_split_within_limit_without_losing_text():
    sentences = [f"Sentence number {i} is here." for i in range(60)]
    chunks = chunk_record(_record(details=" ".join(sentences)), max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    body = " ".join(c.text.split("\n", 1)[1] for c in chunks)
    assert all(s in body for s in sentences)


def test_unbroken_text_longer_than_limit_is_hard_cut():
    chunks = chunk_record(_record(details="x" * 1000), max_chars=300)
    assert len(chunks) > 1
    assert sum(c.text.count("x") for c in chunks) == 1000
