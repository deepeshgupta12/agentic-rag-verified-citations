"""Ranking conflicting sources without deciding for the reader.

Detection answers "do these disagree?". Resolution answers "which should you
believe?" — and must answer it with reasons rather than a verdict. A pipeline
that silently picks a side produces an answer indistinguishable from one
where the sources agreed: the reader sees a confident figure and cannot know
a conflict existed or that something chose for them.
"""

from __future__ import annotations

from ragverify.contradiction import detect
from ragverify.resolution import DECISIVE_MARGIN, as_prompt_block, resolve, summarize
from ragverify.schemas import EvidenceItem, Route

FILING = ("Acme European revenue reached 2.1 billion euro in the third quarter "
          "of 2024 as filed with the commission.")
RUMOUR = ("Sources close to the matter claim Acme European revenue reached around "
          "5.8 billion euro in the third quarter of 2024.")


def web(sid: str, url: str, text: str) -> EvidenceItem:
    return EvidenceItem(source_id=sid, label=url, text=text, origin=Route.WEB, url=url)


def local(sid: str, text: str) -> EvidenceItem:
    return EvidenceItem(sid and sid or "S", label="doc", text=text, origin=Route.LOCAL,
                        source_id=sid)


class TestRanking:
    def test_authority_separates_filing_from_blog(self):
        ev = [web("S1", "https://www.sec.gov/filing", FILING),
              web("S2", "https://rumour.blogspot.com/p", RUMOUR)]
        resolutions = resolve(detect(ev), ev, "What was revenue?")

        assert resolutions
        assert resolutions[0].decisive
        assert resolutions[0].preferred.source_id == "S1"

    def test_both_sides_survive_ranking(self):
        """Ranking is not deletion — the reader must see the disagreement."""
        ev = [web("S1", "https://www.sec.gov/filing", FILING),
              web("S2", "https://rumour.blogspot.com/p", RUMOUR)]
        resolution = resolve(detect(ev), ev, "What was revenue?")[0]

        assert {s.source_id for s in resolution.ranked} == {"S1", "S2"}
        assert "retained" in resolution.describe()

    def test_reasons_are_stated(self):
        ev = [web("S1", "https://www.sec.gov/filing", FILING),
              web("S2", "https://rumour.blogspot.com/p", RUMOUR)]
        resolution = resolve(detect(ev), ev, "What was revenue?")[0]

        assert resolution.ranked[0].reasons(), "a ranking without reasons is a verdict"
        assert "publisher" in resolution.describe()

    def test_hedged_source_scores_lower_on_specificity(self):
        ev = [web("S1", "https://a.test/x", FILING),
              web("S2", "https://b.test/y", RUMOUR)]
        resolution = resolve(detect(ev), ev, "What was revenue?")[0]
        standing = {s.source_id: s.specificity for s in resolution.ranked}

        assert standing["S1"] > standing["S2"], "'claim ... around' is weaker than a filed figure"


class TestInconclusive:
    def test_similar_sources_are_not_separated(self):
        """A near-tie broken by rounding reads exactly like a decisive result."""
        ev = [web("S1", "https://a.test/x",
                  "Acme European revenue reached 2.1 billion euro in the third quarter of 2024."),
              web("S2", "https://b.test/y",
                  "Acme European revenue reached 1.6 billion euro in the third quarter of 2024.")]
        resolution = resolve(detect(ev), ev, "What was revenue?")[0]

        assert not resolution.decisive
        assert resolution.preferred is None
        assert "do not separate" in resolution.describe()

    def test_margin_threshold_is_wide_enough_to_matter(self):
        assert DECISIVE_MARGIN >= 0.1, "a hair-thin margin makes ranking arbitrary"

    def test_summary_reports_both_counts(self):
        ev = [web("S1", "https://www.sec.gov/f", FILING),
              web("S2", "https://rumour.blogspot.com/p", RUMOUR)]
        text = summarize(resolve(detect(ev), ev, "q"))

        assert "ranked by source standing" in text
        assert "Both sides are reported" in text


class TestPromptBlock:
    def test_forbids_hiding_the_conflict(self):
        ev = [web("S1", "https://www.sec.gov/f", FILING),
              web("S2", "https://rumour.blogspot.com/p", RUMOUR)]
        block = as_prompt_block(resolve(detect(ev), ev, "q"))

        # The instruction wraps across lines in the rendered block.
        flat = " ".join(block.split())
        assert "ALWAYS report the other figure" in flat
        assert "Never present a ranked conflict as though the sources agreed" in flat
        assert "[S1]" in block and "[S2]" in block

    def test_empty_input(self):
        assert as_prompt_block([]) == ""
        assert summarize([]) == ""


class TestLocalEvidence:
    def test_local_documents_get_neutral_authority(self):
        """The user chose them; ranking their publisher would be meaningless."""
        ev = [
            EvidenceItem(source_id="S1", label="a.txt", text=FILING, origin=Route.LOCAL),
            EvidenceItem(source_id="S2", label="b.txt",
                         text="Acme European revenue reached 1.6 billion euro in the "
                              "third quarter of 2024.", origin=Route.LOCAL),
        ]
        resolutions = resolve(detect(ev), ev, "What was revenue?")
        assert resolutions
        for standing in resolutions[0].ranked:
            assert standing.authority == 0.5
