# Project Learnings

> Distilled insights that persist across sessions. Updated when archiving old scratchpad sessions.

## Codebase Insights
- frugalCopyla is a Python package for copula-based causal inference
- Uses JAX/NumPyro for probabilistic computation
- Has companion R scripts in nonparanormal/ for experiments
- Paper (Hybrid-Frugal-Paper) is submission-ready, focuses on R implementation

## Recurring Issues
- JAX/JAXlib require manual installation due to version conflicts
- GCM tests show OpenMP warnings on macOS (platform-specific, not a bug)
- VineCopula R-vine matrices must have sequential diagonal
- `topoOrder` default includes outcome Y (usually wrong for frugal models)

## Effective Approaches
- YAML-based experiment framework enables reproducible validation
- DAG-aware rank unconditioning critical for longitudinal models
- BN parameterization (independent ranks + conditional CDFs) works best for Bayesian network structures
- Modular R package structure (8 focused modules) better than monolithic scripts

## Key Decisions Summary
- Use BN parameterization for validation experiments (not vine copula for Z-Z)
- Copula used ONLY for Y-Z dependence, not Z-Z dependence in BN structures
- Paper Section 6 focuses on R implementation (Python ~40% complete, deprioritized)
- Context management with rotation keeps files lean (rotate at 5+ sessions)

---

## Update: 2026-02-15 (Paper Review Session)

### From session 2026-02-15 (Bernoulli submission preparation):

**Iterative Approval Workflow for Paper Editing:**
- User prefers to approve each edit individually before implementation
- Requires before/after comparisons for every change, even small ones
- Will iterate on suggestions (v2, v3, v4) until satisfied
- Implementation happens in batch or incrementally as directed
- This is slower but ensures alignment on narrative and style decisions

**Paper Structure Patterns:**
- Introduction overlapping with background is common issue (Lin/Seaman comparison appeared 3 times)
- "Related Work" subsections work better at end of Section 1 than in Section 2
- Notation paragraphs should precede first use (start of Section 2, not mid-section)
- Detailed technical comparisons belong in background, not introduction
- Introduction should give 1-2 sentence positioning; background gives full comparison

**Conference Submission Checklist (Bernoulli):**
- Target length: ~40-45 pages for main text + critical appendices
- Conclusion must be substantial (25-30 lines minimum, not 7)
- Move proof details to appendix (keep Thm 5.1 as template, move 5.2/5.3)
- Consolidate redundant experiments (3 experiments showing same thing → 2)
- Delete expendable appendices (too-brief ones like App B, redundant ones like App F)
- Delete ALL commented-out text before submission

**Effective Review Process:**
1. Read all sections and appendices completely
2. Identify cross-cutting issues (redundancy, notation inconsistency)
3. Generate prioritized action list (Top 10)
4. Work through in page order (not importance order) for easier navigation
5. Store full review in plan file for reference

**Narrative vs Terseness:**
- User explicitly prefers narrative/story flow over terse academic writing
- "Problem → early solutions → general framework" structure works well
- Detailed technical exposition should follow motivation, not precede it
- Concrete examples (like Figure 1 in intro) are more valuable than abstract descriptions

---

## Update: 2026-02-05 (Session Rotation)

### From archived sessions (Feb 3-4):

**Validation Experiment Design:**
- Created three validation experiments: static, dynamic, longitudinal
- All designed to test whether nonparanormal approximation preserves p(Y|do(X))
- Key insight: X does NOT enter copula structure — affects Y only through causal margin

**DAG-Aware Rank Unconditioning:**
- Fixed assumption that variables conditioned on ALL previous (D-vine)
- Correct: variables conditioned only on their DAG parents
- Required new DAG utility functions (dag.R with 47 tests)
- Critical for longitudinal models with temporal structure

**Two Approaches to Nonparanormal:**
- Approach A (Regenerate): Fresh ranks from Gaussian copula → exact Markov/CI, approximate tail dependence
- Approach B (Preserve): Keep original ranks → preserve tail dependence, ~0.02 Markov violations
- Decision: Use Approach A for causal inference (exact Markov critical)
- Paper should explicitly state which approach used

---

## CRITICAL: Sampling from Gaussian Copula BN with Conditional Marginals

### The Canonical Algorithm (Markov-Preserving)

**Goal:** Sample from a DAG Z_1 → Z_2 → Z_3 where each variable has a conditional marginal F_{k|pa(k)} but the overall copula structure is Gaussian.

**Full Algorithm (conceptually clean):**
```
For each node k in topological order:
  1. Draw Z_k | Z_{pa(k)} ~ N(μ_k(Z_pa), σ_k²)    # From Gaussian BN
  2. U_k = Φ((Z_k - μ_k) / σ_k)                   # Conditional rank (INDEPENDENT of parents)
  3. X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})      # Transform through conditional marginal
```

**Shortcut (mathematically equivalent, simpler):**
```
For each node k in topological order:
  1. U_k ~ Unif(0,1)                              # Independent by definition
  2. X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})      # Transform through conditional marginal
```

### Why These Are Equivalent

In the full algorithm, step 2 computes:
- U_k = Φ((Z_k - μ_k) / σ_k)
- Where (Z_k - μ_k) ~ N(0, σ_k²) is independent of Z_{pa(k)} by construction
- So U_k ~ Unif(0,1) and U_k ⊥ Z_{pa(k)}

This is exactly what `U_k ~ Unif(0,1)` gives directly!

### Why the Markov Property Holds

The data are generated from a recursive SEM:
```
X_1 = g_1(U_1)
X_2 = g_2(X_1, U_2)
X_3 = g_3(X_2, U_3)
```
with independent innovations (U_1, U_2, U_3). This directly yields X_1 ⊥ X_3 | X_2.

### When to Use Each Version

**Use the full Gaussian intermediate when:**
- Fitting copula correlations from empirical data
- You need a unified latent Gaussian representation
- For theoretical clarity or proofs

**Use the shortcut when:**
- Conditional marginals are specified directly (simulation)
- Simplicity is preferred
- You control the DGP

### Common Mistake (What Broke Our Code)

**WRONG approach that causes Markov violations:**
```
1. Fit a joint copula to ALL variables
2. Extract conditional ranks U_{k|pa} from this copula  ← These ARE independent of parents!
3. Transform through conditional marginals F^{-1}_{k|pa}
```

**Why this fails:**
- Step 2 extracts ranks that are ALREADY independent of parents
- Step 3 then transforms through conditional CDFs
- But conditional CDFs need dependence to come from SOMEWHERE
- Result: No dependence created → violates intended DAG structure

**The fix:** Either use the canonical algorithm (independent ranks + conditional marginals) OR use correlated ranks with MARGINAL marginals. Never mix conditional ranks with conditional marginals from different sources.

### R Implementation Pattern

```r
# CORRECT: Independent ranks + conditional marginals
generate_bn_sample <- function(n) {
  # Root node
  U_1 <- runif(n)
  X_1 <- qgamma(U_1, shape = SHAPE_1, scale = SCALE_1)

  # Child node: parameters depend on parent VALUE
  U_2 <- runif(n)  # INDEPENDENT
  shape_2 <- BASE_SHAPE_2 + EFFECT * X_1  # Parameter depends on X_1
  X_2 <- qgamma(U_2, shape = shape_2, scale = SCALE_2)

  return(list(X_1 = X_1, X_2 = X_2))
}
```

### Reference

This algorithm is the "conditional ranks through conditional marginals" approach for Gaussian copula BNs. The key insight is that dependence comes from the conditional CDF parameters (which depend on parent values), NOT from correlated ranks.

---

## Update: 2026-02-05 (Complete Section 6 Validation)

### Session Completed: Complete Paper Section 6 Validation

**Major achievement:** Completed all 4 phases of comprehensive Paper Section 6 update with all three validation experiments showing consistent unbiased ATE estimates.

### Key Results Summary

**All Three Validation Experiments Pass:**

| Model | Markov Test | IPW Bias | AIPW Bias | Status |
|-------|-------------|----------|-----------|--------|
| Static | N/A | 0.002 (p=0.295) | 0.001 (p=0.786) | ✅ Unbiased |
| Dynamic | KS p=0.485 | 0.001 (p=0.547) | 0.000 (p=0.827) | ✅ Unbiased |
| Longitudinal | KS p=0.928/0.618 | 0.0004 | 0.002 | ✅ Unbiased |

### Paper Text Updates

**Added Remark 1 after Algorithm 1:**
- Clarifies distinction between BN factorization (for Z-Z dependencies) and copula (for Y-Z dependencies)
- Explains that Gaussian copula imposed ONLY for Y-Z dependence
- Critical for understanding what is being approximated

**Section 6.5 Clarification:**
- Replaced vague "Gaussian copula approximation" with explicit BN/copula terminology
- Added "Sampling Mechanism" paragraph explaining independent ranks + conditional CDFs
- Makes the BN parameterization approach explicit

**Dynamic Model Validation Added:**
- Complete subsubsection with DAG, Markov test results, ATE table
- Complements static and longitudinal validation
- Shows consistency across temporal structures

**Appendix Fix:**
- Changed "GCM test" → "partial correlation test" in Section app:ci-pvalues
- Note: Section 6.3 kept GCM references (those were actual GCM tests, not partial correlation)

### Code Upgrade

**Dynamic validation script upgraded:**
- Uses `ppcor::pcor.test` (matching longitudinal pattern)
- Integrated Markov test into main simulation loop
- Added KS uniformity test for p-values
- Added p-value histogram output

### Key Implementation Pattern

**For validation experiments, use BN parameterization:**

```r
# Z variables: Independent ranks + conditional CDFs
U_Z2 <- runif(n)  # Independent!
shape_Z2 <- BASE_SHAPE + BETA * Z1  # Parameter depends on parent VALUE
Z2 <- qgamma(U_Z2, shape = shape_Z2, scale = SCALE)

# Y variable: Gaussian copula for dependence on Z
# (after Z is fully generated with marginal ranks)
```

**Critical distinction:**
- Z-Z dependence: Encoded in conditional CDF parameters (BN approach)
- Y-Z dependence: Encoded in Gaussian copula (copula approach)
- NEVER extract conditional ranks from copula and use with conditional CDFs

### Validation Confirms Theory

**What was validated:**
1. ✅ Nonparanormal approximation preserves causal margins p(Y|do(X))
2. ✅ BN parameterization preserves Markov properties (KS tests pass)
3. ✅ IPW and AIPW estimators are unbiased across all temporal structures
4. ✅ Results consistent across static, dynamic, and longitudinal models

**Paper contribution confirmed:**
- Frugal parameterization allows exact specification of p(Y|do(X))
- Nonparanormal approximation is viable for feasible implementation
- Both structural (Markov) and causal (ATE) quantities preserved

### Files Successfully Modified

**Paper:**
- `Hybrid-Frugal-Paper/sections/nonparanormal.tex` — All text updates
- `Hybrid-Frugal-Paper/sections/appendix.tex` — GCM→partial correlation fix
- Updated histogram plots

**Code:**
- `nonparanormal/causal_validation_dynamic.R` — ppcor upgrade

**Documentation:**
- `.claude/scratchpad.md` — Rotated (8→5 sessions)
- `.claude/plan.md` — Marked complete
- `.claude/handoff.md` — Complete session summary
- `.claude/archive/scratchpad-2026-02.md` — Archive updated

### Commit
- **ddadae4** — All changes committed and pushed to origin/inversion
- Paper Section 6 now submission-ready

### Context Management

**Rotation performed:**
- Scratchpad had 8 sessions (> 5 threshold)
- Kept last 5 sessions in active scratchpad.md
- Moved older sessions to archive/scratchpad-2026-02.md
- Extracted and documented key learnings above

**Other tracking files:**
- Errors.md: Still empty (no errors to log)
- Decisions.md: 4 entries (< 50 threshold, no rotation needed)
- References.md: Not checked (likely low, no rotation needed)

### Next Steps (Optional)

1. Fix 7 pre-existing minor paper issues (out of scope for this session)
2. Final submission preparation
3. Consider methodological note comparing BN vs vine approaches

### Lessons Learned This Session

**Paper Writing:**
- Explicit terminology prevents confusion (BN factorization vs copula)
- Remarks after algorithms are valuable for clarifying design choices
- Dynamic validation complements static and longitudinal nicely

**Experimental Design:**
- Consistency across experiments builds confidence
- KS tests for p-value uniformity are good validation
- Having three temporal structures (static/dynamic/longitudinal) shows generality

**Context Management:**
- Rotation at 8 sessions keeps scratchpad manageable
- Extracting learnings before archiving preserves knowledge
- Handoff documents should be immediately actionable

**Last Updated:** 2026-02-05 16:30

---

## Update: 2026-02-11 (Supervisor Feedback Work)

### From sessions 2026-02-10 and 2026-02-11:

**Batch Editing Workflow for Paper Feedback:**
- Processed 50+ supervisor annotations in two commit batches (c2cab3a, 3bb4380)
- Grouped by type: typos, grammar, British→American spelling, notation fixes
- Efficient to batch similar changes together, commit in logical groups
- Separate commits for conceptual changes (e.g., Corollary→Remark rewrite)

**When to Defer vs Fix Immediately:**
- **Fix immediately:** Typos, clear grammar errors, objective corrections
- **Defer:** Conceptual questions, ambiguous comments, changes requiring context
- **Ask supervisor:** When comment intent is unclear or change seems questionable

**Handling Conceptual Passages:**
- Don't guess supervisor intent on abstract concepts (e.g., uniqueness of natural PCCs)
- Analyze the passage to identify the core conceptual question
- Document specific alternatives (e.g., keep/simplify/remove Gamma notation)
- Present analysis to supervisor with clear options

**Notation Consistency Patterns:**
- c_{Y|Z} vs c_{Y|Z*} — be precise about conditioning sets
- "edge and Y" vs "edge to Y" — prepositions matter in graph theory
- British vs American spelling — pick one standard and apply consistently

**Managing Uncommitted Changes:**
- Keep related fixes together (e.g., both notation fixes in uniqueness passage)
- Don't commit until the full logical unit is complete
- Use git diff to review before committing

**Tracking Unresolved Items:**
- Explicitly list items that need supervisor clarification
- Distinguish "blocked" from "deferred" — blocked items stop progress
- Document WHY each item is unresolved (unclear reference, needs conceptual decision, etc.)

**Context Management:**
- Rotate scratchpad at 5 sessions, activity.log at 500 lines
- Extract learnings BEFORE archiving (they're not in the old content)
- Handoff should specify EXACTLY how to resume, including what NOT to do

**Last Updated:** 2026-02-14 17:00

---

## Update: 2026-02-14 (Paper Restructuring Work)

### From session 2026-02-14:

**Pedagogical Flow in Technical Papers:**
- Lead with motivation (WHY) before tools (HOW)
- Section 2 restructuring: MSMs → Copulas → PCCs is more natural than Copulas → MSMs
- Readers need context for why technical machinery is necessary
- Notation should appear just before first use, not as isolated preamble

**Cross-Reference Management:**
- When moving content to appendix, grep for all references: `\Cref`, `\ref`, `\eqref`
- Fix references before committing (prevents broken builds)
- Descriptive label names help: `app:integral-pcc` better than `sec:counterexample`

**Appendix Organization:**
- Order appendices by first citation in main text
- integral_pcc cited in Section 2 → becomes Appendix A
- Modular appendices (6 files) easier to navigate than monolithic file

**Commit Granularity:**
- Separate commits for: (1) reference fixes, (2) polish/flow, (3) major restructuring
- Makes git history more useful for tracking logical changes
- Easy to revert specific changes if needed

**Duplicate Content Detection:**
- Check for duplicates when moving content between sections and appendix
- Example: topological ordering example was in both main text and integral_pcc.tex
- Removing duplicates reduces paper length without losing content

**Bridging Sentences:**
- Add forward references when introducing new subsections: "Before addressing this question in \Cref{sec:method},"
- Helps readers understand narrative flow
- Especially important after major restructuring

**Last Updated:** 2026-02-14 21:00

---

## Update: 2026-02-14 (Subsection 2.1 Rewrite Session)

### From session 2026-02-14 (second session today):

**Collaborative Editing Workflow:**
- When both user and agent edit same files, coordinate frequently
- Pull remote changes before continuing work
- User may manually clean up commented-out LaTeX — this is normal
- Git submodule workflow: commit in submodule first, then update parent repo pointer

**Narrative Structure Preferences:**
- User prefers "problem → early solutions → general framework" structure
- For Section 2.1: MSM need → prior work → Evans framework → recent extensions → limitations
- This creates clear intellectual progression

**Multi-commit Workflow:**
- Initial implementation may get replaced after user review
- Be prepared to restructure even after committing (use rebase if needed)
- User feedback can completely change narrative flow — this is normal iteration

**Managing External Edits:**
- When pulling remote changes, note what changed (git log, git diff)
- User may fix things we missed (commented-out text cleanup)
- Pull twice if needed — user may push multiple times

**Last Updated:** 2026-02-15 16:30

---
