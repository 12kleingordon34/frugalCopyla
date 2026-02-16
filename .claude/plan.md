# Current Plan

**Goal:** Paper Revisions - Bernoulli Submission Preparation (56→40 pages)
**Started:** 2026-02-15
**Status:** In Progress - Review Complete, Edits in Approval Phase
**Branch:** inversion (parent), main (submodule)

## Overview

Comprehensive paper review targeting Bernoulli conference submission. Working through Top 10 priority actions in page order with individual user approval before implementation.

**Current Paper State:** 56 pages (main + 6 appendices)
**Target:** ~40-45 pages
**Estimated Reduction:** 10-15 pages via consolidation and appendix moves

---

## Top 10 Priority Actions (From Paper Review)

### Phase 1: Introduction & Background Edits (In Progress)

- [x] **Action 1/10:** Review complete paper and generate priority list
- [x] **Action 2/10:** Trim introduction lines 1-17 (BN/copula preamble) — v3 approved by user
- [x] **Action 3/10:** Deduplicate Lin/Seaman in introduction (lines 26-28) — v4 approved by user
- [x] **Action 4/10:** Create Related Work subsection at end of Section 1 — approved
- [x] **Action 5/10:** Move notation paragraph to start of Section 2 — approved
- [ ] **Action 6/10:** Trim PCC Uniqueness subsection (Section 2.5) — pending presentation
- [ ] **Action 7/10:** Move Theorems 5.2/5.3 proofs to appendix — pending presentation

### Phase 2: Section 6 Consolidation (Pending)

- [ ] **Action 8/10:** Consolidate Section 6 experiments (keep one static + longitudinal)
  - Drop one static model validation
  - Replace p-value histograms with summary table
  - Move dropped content to supplementary material

### Phase 3: Conclusion & Appendices (Pending)

- [ ] **Action 9/10:** Expand conclusion from 7 to 25-30 lines
  - Recap contributions
  - Practical significance
  - Limitations
  - Future directions

- [ ] **Action 10/10:** Delete Appendices B and F
  - App B (Vine Sampling): Too brief, fold into Section 2 or delete
  - App F (CI p-values): Redundant with Section 6 tables

### Phase 4: Cross-Cutting Polish (Pending)

- [ ] Delete all commented-out text across all sections
- [ ] Add intuition for Conditions 3-4 in Section 5
- [ ] Add sensitivity discussion to Section 6

---

## Active Edit Progress

### Edit 1: Trim Introduction Lines 1-17 (APPROVED v3)

**Before:** ~230 words across two paragraphs
**After:** ~100 words in single paragraph
**Status:** User approved replacement text, awaiting implementation

**Replacement text:**
```latex
Bayesian networks (BNs) and copula models offer complementary approaches to parameterizing multivariate distributions. BNs decompose the joint into conditional factors and naturally encode conditional-independence constraints, making them a standard tool in causal reasoning~\citep{koller2009probabilistic}. Copulas instead separate marginal distributions from the dependence structure~\citep{sklar1959,joe2014dependence}; for high-dimensional settings, pair-copula constructions (PCCs) decompose the joint dependence into a sequence of bivariate copulas, offering considerable modelling flexibility~\citep{bedford2002vines,bauer2012paircopula}.
```

### Edit 2: Deduplicate Lin/Seaman (APPROVED v4)

**Before:** ~180 words — detailed comparison
**After:** ~60 words — brief positioning + forward reference
**Status:** User approved replacement text, awaiting implementation

**Replacement text:**
```latex
Related works target different objects and encode dependence in different ways. \citet{lin2025exactsimulationlongitudinaldata} extend frugal models to longitudinal MSMs via pair-copula constructions, while \citet{seaman2023simulating} target survival-time MSMs by coupling a confounder risk score to a latent failure mechanism via a copula. A detailed comparison of how each approach handles conditional-independence constraints appears in \Cref{sec:background}.
```

**Additional cleanup:** Delete commented-out lines 30-32

### Edit 3: Create Related Work Subsection (APPROVED)

**Action:** Create `\subsection{Related Work}` at end of Section 1
**Content to collect:**
- Young/Keogh/Havercroft prior work from background
- Lin/Seaman detailed comparison from background lines 43-49
- Markov narrative

**Status:** Approved in principle, awaiting detailed before/after

### Edit 4: Move Notation Paragraph (APPROVED)

**From:** background.tex after copulas intro (current lines 53-55)
**To:** Start of Section 2 (before subsections)
**Rationale:** Notation should precede first use, not appear mid-section
**Status:** Approved, awaiting implementation

---

## Completed Steps (Previous Sessions)

### 2026-02-14: Section 2.1 MSM Rewrite - DONE

- [x] Apply initial 4 edits to strengthen MSM motivation (commit: b36d60c, b3d4c4e)
- [x] Restructure narrative flow based on user feedback (commit: 685d1d1, ecb0ddc)
- [x] Pull user's remote edits (commits: 9d8a4e5, 2eb23e4)

### 2026-02-14: Section 2 Restructuring - DONE

- [x] Fix integral_pcc appendix references (commit: 01248ee)
- [x] Polish Section 2 flow (commit: 0f29a07)
- [x] Reorder subsections: MSMs → Copulas → PCCs → Uniqueness → BNs (uncommitted from earlier)

### 2026-02-10-13: Supervisor Feedback - DONE

- [x] Phase 1: Quick Fixes (commit: c2cab3a)
- [x] Phase 2: Corollary 2.3 Rewrite (commit: 3bb4380)
- [x] Phase 3-5: Background notation, Condition 3, Appendix modularization

---

## Next Steps

1. **Await user signal to proceed with implementation** of edits 1-4
2. Present action 6/10 (PCC Uniqueness trimming) for approval
3. Present action 7/10 (move proofs) for approval
4. Continue through remaining actions with individual approval

---

## Blockers

**No active blockers.** User approving edits individually before implementation (by design).

---

## Notes

- **Branch:** inversion (parent), main (submodule at commit a4cc32b)
- **Workflow:** Plan mode active — approval before implementation
- **User preferences:**
  - Before/after comparison for every change
  - Individual approval per suggestion
  - Narrative flow over terseness
  - No em dashes
  - Related Work at end of Section 1 (not Section 2)

- **Uncommitted changes from previous sessions:**
  - Section 2 subsection reordering (from 2026-02-14, may need verification)

- **Paper review location:**
  - Full review stored in `/Users/danielmanela/.claude/plans/inherited-leaping-hickey.md`

---

# Previous Plans (Completed)

## Plan: Subsection 2.1 Rewrite - COMPLETE (2026-02-14)

**Goal:** Strengthen MSM motivation and improve narrative flow
**Status:** Complete (commits: b36d60c, 685d1d1, 9d8a4e5, 2eb23e4)

## Plan: Supervisor Feedback Quick Fixes - COMPLETE (2026-02-10)

**Goal:** Address RJE annotations
**Status:** Complete (commits: c2cab3a, 3bb4380)

## Plan: Section 6 Validation - COMPLETE (2026-02-05)

**Goal:** Complete all three validation experiments
**Status:** Complete (commit: ddadae4, pushed to origin/inversion)
