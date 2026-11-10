import pytest
from governed_rag.gates import authority_of, check, numbers
from governed_rag.types import Authority, Citation, Draft, Evidence

EV = (Evidence("E1", "kb/p", "Refund policy", Authority.CURATED,
               "Refund window is 14 days. Orders over 1,000 EUR need approval.", 0.9),
      Evidence("E2", "archive/t", "Ticket", Authority.ARCHIVE, "Customer waited 30 days.", 0.6))

def d(answer, *cites): return Draft(False, answer, tuple(Citation(e, q) for e, q in cites))
def chk(draft, m=Authority.ARCHIVE): return check(draft, EV, m)

def test_numbers_extraction():
    assert numbers("A 14-day window, 1,000 EUR, 3.5 and [E12].") == {"14", "1000", "3.5"}

def test_valid_draft_passes_despite_case_and_spacing():
    assert chk(d("The window is 14 days [E1].", ("E1", "refund window is  14 DAYS"))) == []

def test_abstention_passes():
    assert chk(Draft(abstain=True)) == []

@pytest.mark.parametrize("draft, code", [
    (d("The window is 14 days."), "G1:no_citation"),
    (d("x [E9]", ("E9", "x")), "G1:unknown_evidence:E9"),
    (d("14 days [E1] and more [E2]", ("E1", "14 days")), "G1:marker_without_citation:E2"),
    (d("14 days [E1]", ("E1", "window is fourteen days")), "G2:quote_not_found:E1"),
    (d("The window is 30 days [E1].", ("E1", "Refund window is 14 days")), "G3:number_unsourced:30"),
])
def test_each_failure_is_reported(draft, code):
    assert code in chk(draft)

def test_number_formats_normalise():                             # Review Focus 5
    quote = "Refund window is 14 days. Orders over 1,000 EUR need approval."
    assert chk(d("A 14-day window; approval above 1000 EUR [E1].", ("E1", quote))) == []

def test_authority_is_the_weakest_cited_tier():
    both = d("14 days [E1], one case 30 days [E2]", ("E1", "14 days"), ("E2", "30 days"))
    assert authority_of(both, EV) == Authority.ARCHIVE
    assert chk(both, Authority.CURATED) == ["G4:authority_below_minimum"]
    assert authority_of(d("14 days [E1]", ("E1", "14 days")), EV) == Authority.CURATED
    assert authority_of(Draft(abstain=True), EV) == Authority.NONE

def test_quote_must_align_with_token_boundaries():
    assert "G2:quote_not_found:E1" in chk(d("The window is 4 days [E1].", ("E1", "4 days")))
    assert "G2:quote_not_found:E1" in chk(d("x [E1]", ("E1", "fund window")))
    assert "G2:quote_not_found:E1" in chk(d("Over 000 EUR [E1]", ("E1", "000 EUR")))
    assert chk(d("The window is 14 days [E1].", ("E1", "14 days"))) == []
    assert chk(d("Approval above 1,000 EUR [E1].", ("E1", "over 1,000 EUR need approval."))) == []
    assert chk(d("The window is 14 days [E1].", ("E1", "is 14 days. Orders"))) == []

def test_empty_answer_fails():
    assert chk(Draft(False, "  ", (Citation("E1", "Refund window is 14 days."),)))[0] == "G1:empty_answer"
