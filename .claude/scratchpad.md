# Scratchpad

## Session: 2026-02-16 - Route B: Replace Empirical PIT with Gaussian BN SEM Propagation

### Current Task
Implemented Route B across both R code and paper. Replaces empirical rank-based PIT (`rank(z)/(n+1)`) with DAG-constrained linear-Gaussian BN propagation on the latent scale.

### Key Accomplishments

**Code Changes (B1-B7):**
1. **B1:** Added `fit_reference_gaussian_bn()` to `gaussian_copula_dag.R` — convenience wrapper for one-time calibration
2. **B4:** Modified `simulateConditionalOutcomeSamples()` — accepts `gaussian_bn_fit` and `parents`, auto-fits DAG-constrained BN, backward compatible
3. **B2:** Modified `generate_frugal_outcome_approach_A()` — uses DAG-constrained R via `fit_reference_gaussian_bn()` instead of `fitMVGaussianCopula`
4. **B3:** Modified `generate_frugal_outcome()` — added `gaussian_bn_fit` and `dag_parents` params, backward compatible
5. **B5:** Modified `simulateMarginalOutcomeSamples()` — accepts `gaussian_bn_fit` as optional override
6. **B6:** Mirrored all changes into legacy `nonparanormal.R` (added `fit_gaussian_copula_dag`, helpers, `fit_reference_gaussian_bn`, updated samplers)
7. **B7:** Updated `experiment_framework.R` — added `extract_dag_parents()` helper, experiment runner now auto-fits DAG-constrained Gaussian BN

**Paper Changes (A1-A5):**
8. **A3:** Updated Algorithm 1 `\Require` block — added Gaussian BN $(B, D)$ to inputs
9. **A2:** Replaced Algorithm 1 Block 2 — empirical ranks → SEM propagation with $\varepsilon_d$, $q_d^{\text{raw}}$
10. **A1:** Updated prose paragraph — describes Route B propagation method
11. **A5:** Added explanatory paragraph after Algorithm 1 — explains Gaussian BN fitting (OLS regression, cov2cor)
12. **A4:** Updated Remark 3 — replaced "compute marginal pseudo-observations" with "propagating BN conditional PITs"

### Test Results
All tests pass:
- gaussian_copula_dag: 55/55 pass
- gaussian_copula_dag_sample: 27/27 pass
- outcome_generation: 18/18 pass
- rank_transform: 18/18 pass
- copula_fit: 26/26 pass
- correlation: 23/23 pass
- dag: 47/47 pass
- integration: 16/17 pass (1 pre-existing failure in vine reparameterization, unrelated)

### Files Modified
- `nonparanormal/R/gaussian_copula_dag.R` — Added `fit_reference_gaussian_bn()`
- `nonparanormal/R/outcome_generation.R` — Modified all 3 sampling functions
- `nonparanormal/R/generate_frugal_outcome.R` — Modified both outcome generation functions
- `nonparanormal/R/experiment_framework.R` — Added `extract_dag_parents()`, updated `run_experiment()`
- `nonparanormal/nonparanormal.R` — Mirrored all code changes into legacy monolith
- `Hybrid-Frugal-Paper/sections/nonparanormal.tex` — All paper changes (Algorithm 1, prose, Remark 3)

### Design Decisions
- All functions are backward compatible (new params default to NULL, old behavior preserved)
- When `parents` provided without `gaussian_bn_fit`, auto-fits from data
- `extract_dag_parents()` infers DAG from config shape_formula references
- Legacy `nonparanormal.R` gets full copies of `fit_gaussian_copula_dag` and helpers (no cross-file sourcing)

---

## Session: 2026-02-15 - Paper Review and Introduction Editing (Plan Mode)

### Current Task
Comprehensive paper review targeting Bernoulli conference submission. Working through page-order suggestions with user approval on each change.

### Key Accomplishments

**Paper Review Completed:**
- Full review of all sections (abstract, intro, background, parameterizing, hybrid_frugal, survival, nonparanormal, conclusion)
- Reviewed all appendices (A-F)
- Generated Top 10 priority actions to reduce from 56 to ~40-45 pages
- Review stored in plan file: inherited-leaping-hickey.md

**Edits Accepted (pending implementation, in plan mode):**
1. ✅ **Suggestion 1/10:** Trim introduction preamble (lines 1-17) — compress BN/copula description from ~230 words to ~100 words (v3 approved)
2. ✅ **Suggestion 2/10:** Deduplicate Lin/Seaman in introduction (lines 26-28) — from ~180 words to ~60 words, forward reference to background (v4 approved)
3. ✅ **Suggestion 3/10:** Create `\subsection{Related Work}` at end of Section 1 — collect Young/Keogh/Havercroft + Lin/Seaman + Markov narrative there
4. ✅ **Suggestion 4/10:** Move notation paragraph from after Copulas intro to before it (start of Section 2)

**Edits Pending (not yet presented):**
- Suggestion 5/10: Trim PCC Uniqueness subsection (Markov chain example)
- Suggestion 6/10: Move Thm 5.2/5.3 proofs to appendix
- Suggestion 7/10: Consolidate Section 6 experiments
- Suggestion 8/10: Expand conclusion
- Suggestion 9/10: Delete Appendices B and F
- Suggestion 10/10: Add intuition for Conditions 3-4 in Section 5

### User Preferences Documented
- Wants before/after comparisons for every change
- Wants to approve each change individually before implementation
- Prefers narrative/story over terse academic writing
- No em dashes
- Related Work subsection goes at end of Section 1 (not Section 2)

### Files Reviewed
- `Hybrid-Frugal-Paper/sections/introduction.tex` — 4 accepted changes pending
- `Hybrid-Frugal-Paper/sections/background.tex` — 2 accepted changes pending
- `Hybrid-Frugal-Paper/sections/parameterizing.tex` — reviewed
- `Hybrid-Frugal-Paper/sections/hybrid_frugal.tex` — reviewed
- `Hybrid-Frugal-Paper/sections/survival.tex` — reviewed
- `Hybrid-Frugal-Paper/sections/nonparanormal.tex` — reviewed
- `Hybrid-Frugal-Paper/sections/conclusion.tex` — reviewed
- All appendices A-F reviewed

### Key Findings
- Paper is solid theoretically but needs tightening for Bernoulli submission
- Introduction overlaps heavily with background.tex (Lin/Seaman comparison appears 3 times)
- Section 5 has all proof details in main text (should move proofs of Thm 5.2/5.3 to appendix)
- Section 6 has three validation experiments showing same thing (could consolidate)
- Conclusion is only 7 lines (needs expansion to 25-30 lines)
- Appendices B and F are expendable
- Estimated reduction: 10-15 pages

---

## Session: 2026-02-14 - Section 2.1 MSM Subsection Rewrite

### Current Task
Rewrote Subsection 2.1 (MSM introduction) in background.tex to strengthen motivation and improve narrative flow per user feedback.

### Key Accomplishments

**Initial implementation (commits b36d60c, b3d4c4e):**
- Applied 4 planned edits to strengthen MSM motivation:
  1. Added motivating paragraph: longitudinal confounding → MSMs → simulation difficulty
  2. Added Seaman & Keogh (2024) to prior-work paragraph
  3. Changed "addresses this" → "provides a more general framework by directly parameterizing"
  4. Added copula/CI foreshadowing to dependency-measure paragraph

**Restructuring after feedback (commits 685d1d1, ecb0ddc):**
- User didn't like narrative flow
- Restructured 2.1 with new order:
  1. Start with MSMs and their need + early prior work (Young, Keogh, Havercroft)
  2. Introduce frugal parameterisation (Evans)
  3. Then Lin et al. and Seaman & Keogh
  4. Then hint at Markov assumption limitations
- Much clearer progression from problem → early solutions → general framework

**Pulled remote changes:**
- User made manual edits on remote
- First pull (9d8a4e5): 8 insertions, 13 deletions to background.tex — cleaned up commented-out text
- Second pull (2eb23e4): Changes to background.tex and hybrid_frugal.tex

### Files Modified
- `Hybrid-Frugal-Paper/sections/background.tex` — Complete Subsection 2.1 rewrite
- `Hybrid-Frugal-Paper/sections/hybrid_frugal.tex` — Pulled remote changes

### Commits This Session
**Submodule (Hybrid-Frugal-Paper):**
- `b36d60c` — Initial MSM motivation edits
- `685d1d1` — Restructured 2.1 narrative (rebased from b36d60c)
- `9d8a4e5` — User's remote edits (pulled)
- `2eb23e4` — User's remote edits (pulled)

**Parent repo:**
- `b3d4c4e` — Updated submodule pointer (initial)
- `ecb0ddc` — Updated submodule pointer (after restructure)

### Key Findings
- User has strong preference for "problem → early solutions → general framework" narrative structure
- Commented-out LaTeX should be removed once edits are finalized (user cleaned this up remotely)
- Frequent coordination needed when both user and agent edit same files

---

## Session: 2026-02-14 - Section 2 Restructuring Complete

### Current Task
Complete Section 2 restructuring begun in previous session. Reorder subsections to lead with frugal model motivation (MSMs) before copula tools.

### Key Accomplishments
1. **Fixed integral_pcc appendix references** (committed 01248ee):
   - Fixed `\Cref{fig:pcc-counterexample}` → `\Cref{app:integral-pcc}` in nonparanormal.tex
   - Removed duplicate topological ordering example from integral_pcc.tex
   - Added `\input{appendices/integral_pcc}` to appendix.tex (as App A)

2. **Polished Section 2 flow** (committed 0f29a07):
   - Fixed "Consider again" → "Consider" (no "again" since figure now in appendix)
   - Fixed "reverses this operation" → "reverses the h-function" (clearer antecedent)
   - Added bridging sentence before h-function Manipulation subsection

3. **Reordered Section 2 subsections** (uncommitted):
   - New order: 2.1 MSMs → 2.2 Copulas → 2.3 PCCs → 2.4 Uniqueness → 2.5 Integrating BNs
   - Moved notation paragraph to end of 2.1 (before copulas section)
   - Rationale: Motivation (frugal models) before tools (copulas)

### Files Modified
- `Hybrid-Frugal-Paper/sections/background.tex` — Major subsection reordering + notation placement
- `Hybrid-Frugal-Paper/sections/appendix.tex` — Added integral_pcc input (committed)
- `Hybrid-Frugal-Paper/appendices/integral_pcc.tex` — Removed duplicate content (committed)
- `Hybrid-Frugal-Paper/appendices/nonparanormal.tex` — Fixed cross-reference (committed)

### Commits This Session
- `01248ee` — refactor: Restructure Section 2 background for improved flow
- `0f29a07` — fix: Polish flow after Section 2 restructuring

### Open Questions
- Should we commit the subsection reordering or wait for other changes?

---

## Session: 2026-02-11 - Handoff Update (Documentation Only)

### Current Task
Updated tracking files for handoff. No code changes made this session — purely documentation.

### Key State Summary
- **Branch:** main (changed from inversion)
- **Latest commits:** 3bb4380 (Corollary→Remark), c2cab3a (50+ fixes)
- **Uncommitted:** 2 notation fixes in background.tex (lines 160, 162)
- **Active work:** Uniqueness passage in background.tex (lines 160-168)

---

## Session: 2026-02-10 - Supervisor Feedback Quick Fixes (RJE Annotations) - COMPLETE

### Current Task
Address supervisor feedback from RJE annotations on Hybrid-Frugal-Paper. Completed 45+ quick fixes across two batches.

### Key Accomplishments

**Batch 1 - Committed (commit: 51e50c8, later merged into c2cab3a on main):**
1. Removed "a later section" (pg 9) - background.tex
2. Changed "Premise" → "Condition" (pg 26, 2 instances) - survival.tex
3. Removed "again" (pg 13) - parameterizing.tex
4-18. Additional 15 quick fixes (typos, grammar, notation)

**Batch 2 - Also committed (merged into c2cab3a on main):**
19-49. Additional 25+ fixes including: British→American spelling (15+ instances), typos, grammar, duplicate paragraphs, notation fixes across 8 files.

**Corollary 2.3 Rewrite (commit: 3bb4380):**
- Rewrote as Remark with corrected PCC formula
- Added Remark 2.4 for |Pa(V_i)| bivariate copula explanation

### Deferred Items
- "Topologically contiguous" order dependence clarification (pg 24) - user unsure
- "Parents of the children of Q" scope clarification (pg 24) - user unsure
- "covariates for which" editorial decision (pg 3)

### Commits
- **c2cab3a** - Batch of 50+ fixes (on main)
- **3bb4380** - Corollary→Remark rewrite (on main)

---
