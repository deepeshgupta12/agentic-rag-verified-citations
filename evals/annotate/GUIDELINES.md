# Groundedness annotation guidelines

You are shown a **claim** and **one passage** it cites. Judge only whether
that passage supports that claim.

This is the reference the pipeline is graded against. Every quality number it
reports about itself comes from the same rules it uses to decide, which
measures self-consistency rather than correctness — your judgement is the
only thing that breaks that circle.

---

## The one rule that matters most

**Judge against the passage, not against reality.**

If you know a claim is true but this passage does not establish it, the label
is `unsupported`. The system is being tested on whether it can tell what its
sources say — not on whether it happens to be right.

This is the rule annotators break most often, and breaking it makes the
resulting labels useless: they would measure the model's world knowledge,
which is exactly the thing the pipeline is supposed to refuse to rely on.

---

## Labels

### `s` — supported
A careful reader of this passage must conclude the claim is true. Paraphrase
is fine; wording need not match.

> **Passage:** "European revenue grew 34% year over year to €2.1 billion."
> **Claim:** "Revenue in Europe rose 34%." → `supported`

### `p` — partial
The passage supports part of the claim, or supports it with a caveat the
claim drops.

> **Passage:** "Revenue grew in most European regions."
> **Claim:** "Revenue grew across Europe." → `partial`

Use this freely. Forcing these into `supported` or `unsupported` is where
agreement collapses, and a scheme with no middle produces low κ for a reason
that has nothing to do with the annotators.

### `u` — unsupported
The passage neither establishes nor refutes the claim. Includes:

- **Right topic, wrong fact.** Passage covers revenue; claim is about headcount.
- **Attribution.** "The CEO said profits rose" does not establish that profits rose.
- **Modality and tense.** "expects to launch" does not establish "launched".
- **Scope.** A passage about the Berlin office does not establish a claim about the company.

### `c` — contradicted
The passage states something incompatible with the claim.

> **Passage:** "Revenue grew 34%."
> **Claim:** "Revenue grew 47%." → `contradicted`

Contradicted is stronger than unsupported and worth distinguishing: no
further retrieval fixes a claim that asserts the opposite of its source.

### `?` — unclear
You genuinely cannot judge — ambiguous wording, missing context, or domain
knowledge you do not have. **Use it rather than guessing.** A guess is
indistinguishable from a judgement in the data and quietly corrupts the
reference; `unclear` items are separated out during adjudication.

---

## Edge cases, decided in advance

Consistency matters more than which way each is decided, so these are settled
here rather than left to individual judgement:

| Situation | Label | Why |
|---|---|---|
| Claim says "most", passage says "many" | `partial` | Overlapping but not equivalent |
| Claim rounds 34.2% to 34% | `supported` | Rounding is not fabrication |
| Claim says 34%, passage says "about a third" | `partial` | Compatible, less precise |
| Passage is a table row the claim reads correctly | `supported` | Layout is not content |
| Claim combines two facts, passage has one | `partial` | Half-supported is partial |
| Claim is true but this passage is silent | `unsupported` | **The core rule** |
| Passage contradicts itself | `?` | Adjudication decides |
| Claim is a gap statement ("the sources do not give X") | `?` | Absence is unverifiable from one passage |

---

## Practical notes

- **Do not look at other passages.** You are judging this pair. Another
  passage supporting the claim is irrelevant here.
- **Do not rush the long ones.** Time per item is recorded; a run of
  five-second judgements on dense passages is visible in the data.
- **Take breaks.** Aim for 50–80 items per sitting. Agreement degrades with
  fatigue and the degradation is not random — late items skew toward whichever
  label is easiest to reach for.
- Roughly 30 seconds per item is normal. 300–500 items gives a stable κ.

## Targets

| κ | Reading |
|---|---|
| > 0.8 | Almost perfect — guidelines are working |
| 0.6–0.8 | Substantial — usable |
| 0.4–0.6 | Moderate — **fix the guidelines before labelling more** |
| < 0.4 | The task is underspecified, not the annotators |

Below 0.6, stop and look at the disagreements as a group. The usual cause is
a category the guidelines do not cover, and adding a rule for it is far
cheaper than labelling more items at low agreement.
