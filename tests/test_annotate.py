"""Groundedness annotation harness.

The pipeline cannot grade itself: every quality number it reports comes from
the same lexical rules it uses to decide. These tests cover the machinery
that produces an independent reference — the labels themselves need humans.
"""

from __future__ import annotations

import pytest

from evals.annotate.agreement import (
    agreement_report,
    cohens_kappa,
    fleiss_kappa,
    interpret,
)
from evals.annotate.schema import Annotation, AnnotationItem, GoldItem, Label


class TestLabels:
    def test_positive_projection(self):
        """The pipeline decides binary, so scoring needs a binary view."""
        assert Label.SUPPORTED.is_positive
        assert Label.PARTIAL.is_positive, "a partially supported citation is defensibly kept"
        assert not Label.UNSUPPORTED.is_positive
        assert not Label.CONTRADICTED.is_positive
        assert not Label.UNCLEAR.is_positive

    def test_item_id_is_content_stable(self):
        a = AnnotationItem.make_id("revenue grew 34%", "S1", "European revenue grew 34%.")
        b = AnnotationItem.make_id("revenue grew 34%", "S1", "European revenue grew 34%.")
        assert a == b, "re-extraction must not renumber items"

    def test_item_id_tracks_the_passage_not_just_the_id(self):
        """The same S1 in a different run may be a different passage."""
        a = AnnotationItem.make_id("claim", "S1", "passage one")
        b = AnnotationItem.make_id("claim", "S1", "passage two")
        assert a != b


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels = ["supported", "unsupported", "partial"] * 5
        assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_chance_agreement_scores_near_zero(self):
        """Two annotators agreeing only as often as random labelling would."""
        a = ["supported", "unsupported"] * 20
        b = ["supported", "unsupported", "unsupported", "supported"] * 10
        assert abs(cohens_kappa(a, b)) < 0.2

    def test_skew_is_corrected(self):
        """Raw agreement flatters a skewed set; kappa should not."""
        a = ["supported"] * 90 + ["unsupported"] * 10
        b = ["supported"] * 95 + ["unsupported"] * 5
        raw = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)

        assert raw > 0.9
        assert cohens_kappa(a, b) < raw - 0.2, "kappa must discount the skew"

    def test_systematic_disagreement_is_negative(self):
        a = ["supported"] * 10 + ["unsupported"] * 10
        b = ["unsupported"] * 10 + ["supported"] * 10
        assert cohens_kappa(a, b) < 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cohens_kappa(["supported"], ["supported", "partial"])

    def test_empty(self):
        assert cohens_kappa([], []) == 0.0


class TestFleissKappa:
    def test_unanimous(self):
        ratings = [["supported"] * 3, ["unsupported"] * 3] * 10
        assert fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_partial_disagreement_lowers_it(self):
        unanimous = [["supported"] * 3, ["unsupported"] * 3] * 10
        mixed = [["supported", "supported", "unsupported"], ["unsupported"] * 3] * 10
        assert fleiss_kappa(mixed) < fleiss_kappa(unanimous)

    def test_uneven_rater_counts_raise(self):
        with pytest.raises(ValueError):
            fleiss_kappa([["supported", "supported"], ["supported"]])

    def test_empty(self):
        assert fleiss_kappa([]) == 0.0


class TestInterpretation:
    @pytest.mark.parametrize(
        "kappa,expected",
        [(0.9, "almost perfect"), (0.7, "substantial"), (0.5, "moderate"),
         (0.3, "fair"), (0.1, "slight")],
    )
    def test_bands(self, kappa, expected):
        assert interpret(kappa) == expected

    def test_negative_is_called_out(self):
        assert "worse than chance" in interpret(-0.2)


class TestAgreementReport:
    def test_two_annotators_use_cohen(self):
        report = agreement_report({
            "alice": {"i1": "supported", "i2": "unsupported", "i3": "partial"},
            "bob": {"i1": "supported", "i2": "unsupported", "i3": "supported"},
        })
        assert report["method"] == "cohen"
        assert report["items_compared"] == 3

    def test_three_annotators_use_fleiss(self):
        labels = {"i1": "supported", "i2": "unsupported"}
        report = agreement_report({"a": labels, "b": labels, "c": labels})
        assert report["method"] == "fleiss"

    def test_only_jointly_labelled_items_are_compared(self):
        """A partial overlap must not be silently treated as full coverage."""
        report = agreement_report({
            "alice": {"i1": "supported", "i2": "partial"},
            "bob": {"i1": "supported"},
        })
        assert report["items_compared"] == 1
        assert report["items_labelled"] == {"alice": 2, "bob": 1}

    def test_small_sample_is_flagged_unreliable(self):
        report = agreement_report({
            "alice": {"i1": "supported"}, "bob": {"i1": "supported"},
        })
        assert report["reliable"] is False, "a kappa over one item is noise"

    def test_single_annotator_reports_no_kappa(self):
        assert agreement_report({"alice": {"i1": "supported"}})["kappa"] is None

    def test_no_shared_items(self):
        report = agreement_report({"alice": {"i1": "supported"}, "bob": {"i2": "partial"}})
        assert report["kappa"] is None


class TestGoldItem:
    def test_records_how_it_was_reached(self):
        """An adjudicated label is weaker evidence than a unanimous one."""
        gold = GoldItem(
            item_id="x", label=Label.PARTIAL, n_annotators=3, agreement=0.667,
            adjudicated=True,
            labels_given=[Label.SUPPORTED, Label.PARTIAL, Label.PARTIAL],
        )
        assert gold.adjudicated
        assert gold.agreement < 1.0

    def test_annotation_records_time(self):
        """Five-second judgements on dense passages are visible in the data."""
        a = Annotation(item_id="x", annotator="alice", label=Label.SUPPORTED, seconds=31.4)
        assert a.seconds > 0
