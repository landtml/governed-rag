from governed_rag.chunking import chunk_markdown
from governed_rag.text import count_tokens

def toks(c): return count_tokens(c.body)

def test_empty_body_yields_no_chunks():
    assert chunk_markdown("T", "  \n", 350, 500) == []

def test_sections_never_share_a_chunk_and_paths_nest():
    cs = chunk_markdown("Handbook", "intro\n\n# A\n\na1\n\n## B\n\nb1\n\n# C\n\nc1", 350, 500)
    assert [(c.ord, c.section, c.body) for c in cs] == [
        (0, "", "intro"), (1, "A", "a1"), (2, "A > B", "b1"), (3, "C", "c1")]
    assert cs[2].index_text == "Handbook > A > B\n\nb1"
    assert cs[0].index_text == "Handbook\n\nintro"

def test_packs_paragraphs_up_to_target():
    body = "\n\n".join(f"p{i} " + "w " * 99 for i in range(7))          # 7 paragraphs of 100 tokens
    assert [toks(c) for c in chunk_markdown("T", body, 350, 500)] == [300, 300, 100]

def test_atomic_blocks_are_never_split():
    table = "\n".join(["| a | b |", "|---|---|"] + ["| x | y |"] * 200)
    code = "```\n# not a heading\n" + "line\n" * 600 + "```"
    listing = "\n".join(f"- item {i} " + "w " * 5 for i in range(100))
    cs = chunk_markdown("T", f"# S\n\n{table}\n\n{code}\n\n{listing}", 350, 500)
    assert [c.body for c in cs] == [table, code, listing]
    assert {c.section for c in cs} == {"S"}

def test_oversized_paragraph_splits_on_sentences():
    cs = chunk_markdown("T", "Alpha beta gamma delta. " * 300, 350, 500)
    assert len(cs) > 1 and all(toks(c) <= 500 for c in cs) and all(c.body.endswith(".") for c in cs)

def test_paragraph_between_target_and_max_stays_whole():
    assert len(chunk_markdown("T", "w " * 450, 350, 500)) == 1

def test_loose_list_is_one_atomic_block():
    listing = "\n\n".join(f"- item {i} " + "w " * 40 for i in range(12))
    assert [c.body for c in chunk_markdown("T", f"# S\n\n{listing}\n\nAfter.", 350, 500)] == [listing, "After."]

def test_blank_line_then_paragraph_ends_a_list():
    cs = chunk_markdown("T", "- a\n- b\n\nPara.", 350, 500)
    assert [c.body for c in cs] == ["- a\n- b\n\nPara."]
    big = "\n".join(f"- item {i} " + "w " * 40 for i in range(12))
    assert [c.body for c in chunk_markdown("T", f"{big}\n\nPara.", 350, 500)] == [big, "Para."]
