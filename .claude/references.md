# References & Sources

> ⚠️ IMPORTANT: Every factual claim, technique, or recommendation should be traceable to a source listed here. If no source exists, clearly state it's based on reasoning/inference.

---

## Quick Reference Index

| ID | Title | Type | Topic | Verified |
|----|-------|------|-------|----------|
| REF-001 | Hybrid-Frugal-Paper manuscript | Paper | Frugal causal modeling, copulas | ✅ Verified |

---

## References by Topic

### Frugal Causal Modeling

#### [REF-001] Hybridized Conditional Copula Networks Parameterization for Frugal Causal Simulation
- **Authors:** Daniel de Vassimon Manela, Xi Lin, Chase Mathis, Robin J. Evans
- **Location:** `/Users/danielmanela/Library/CloudStorage/GoogleDrive-danielmanela@gmail.com/My Drive/work/Oxford/frugalCopyla/Hybrid-Frugal-Paper/main.tex`
- **Date Accessed:** 2026-02-02
- **Type:** Academic manuscript (submission draft for JRSS-B)
- **Verified:** ✅ Verified (project's own paper)
- **Key Information Extracted:**
  - Definitions of frugal, natural, feasible parameterizations
  - Graphical conditions (1-4) for feasibility
  - h-function tricks (Lemmas 3.5-3.6)
  - Nonparanormal approximation algorithm (Algorithm 1, Section 6)
  - Theorems 4.1-4.3 on consequences of incorrect CDF specification
  - Theorems 5.1-5.3 on feasibility conditions
- **Context:** Primary source for all project terminology, mathematical definitions, and implementation requirements. All code should implement concepts from this paper.
- **Sections Referenced:**
  - §2: Background on copulas, PCCs, h-functions
  - §3: Natural and feasible definitions
  - §4: Non-feasible models and error bounds
  - §5: Feasibility conditions
  - §6: Nonparanormal approximation + experiments

---

## Unverified Claims

> Claims made without a source that should be verified later:

### Code Quality Assessment
- **Claim:** "Python package is ~40% complete"
  - **Basis:** Line count, TODO comments, missing functions analysis
  - **Status:** Should verify with user what "complete" means for their goals

- **Claim:** "R package is ~90% complete"
  - **Basis:** Full roxygen docs, all functions implemented, works in experiments
  - **Status:** Should verify if there are missing features user needs

### Technical Implementation
- **Claim:** "causl is R-based, so R integration is easier"
  - **Basis:** Inference from package name and common R causal inference ecosystem
  - **Status:** Should verify with user or check causl documentation
  - **Action:** Need to search for causl documentation if user confirms this is important

---
