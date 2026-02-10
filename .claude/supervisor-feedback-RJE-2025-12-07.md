# Supervisor Feedback Log: RJE 2025-12-07

> Annotations from PDF: `~/Downloads/Hybrid_Frugal_Paper_RJE_2025-12-07.pdf`
> Compiled: 2026-02-10

---

## General Comments (from email)

- Remove bracketing (somewhat!)
- Consistent use of American spelling for the entire document

---

## Page-by-Page Feedback

### Page 3
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "covariates for which" | "why remove this section?" | Clarify or restore removed content |

### Page 4
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "which may require some form of numerical integration and struggle to be sampled from" | Didn't like the brackets (general bracketing issue) | Remove unnecessary brackets |

### Page 7
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| Copula notation `cX1X2(FX1,FX3)` | "odd notation" - something not rendered correctly | Fix notation/rendering issue |
| "If we let W = Z" | "seems unusual to do this, just say you're taking it to be topological" | Simplify explanation |
| "valid factorization of this model" | Out of touch, doesn't need to be mentioned | Consider removing |

### Page 8
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "if we choose σ to be the identity permutation" | "Always true" | Remove or clarify - may be redundant |
| "those that correctly model the parameterize the conditional dependency Y\|Z but with the minimal number of bivariate copulas required" | "How do you know this is unique?" | Address uniqueness question |
| Corollary 2.3 | "A corollary of what?" | Add reference to parent theorem |
| "is parameterized by \|V^Pa(i)\| bivariate copulas" | "where does this come from?" | Add derivation or reference |
| "member of this set" | Notation clarity | Clarify |

### Page 9
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "a later section" | "I don't think we do this in the paper" | Remove vague reference |
| "Lemma 2" | "this was already proven. look it up!" - likely in Lin et al. paper | ✅ DONE: Now cites Lin et al. (2025), Lemma 3.6 directly |

### Page 10
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "Proof. See Appendix C.2" | Unclear what comment is | Investigate |

### Page 11
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "(e.g. ..." | Overuse of brackets | Remove brackets |
| "leverage" | Should be "generally leverage" | Add "generally" |
| "(as well as the conditional distribution of treatment given the covariates)" | Remove brackets | Remove parentheses, integrate into sentence |

### Page 12
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "This violation" | "How is this a violation?" | Clarify why it's a violation |

### Page 13
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| `pXZ = pX|Z · pZ` / `pY|do(X)` | Variables should be in topological order (Z, X, Y) | Reorder to `pZX = pZ · pX|Z` etc. |
| "i=2" | "bit irregular to see this in math notation. just collapse into one factor?" | Simplify notation |
| Definition 3.1 language | "clunky" | Rewrite for clarity |
| "However" and "again" | Remove these words | Delete |

### Page 15
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| Lemma 3.3 / `pYt|do(Xt)` | Not sure what's wrong | Investigate |

### Page 16
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| Comma | Remove comma | Delete comma |
| Theorem 4.1 - V notation | "problem with notation of V being duplicated. Maybe remove the first V?" | Fix duplicate V notation |

### Page 17
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| Variable ordering and spelling | Check ordering and spelling | Review and fix |

### Page 19
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "approximate the marginal..." | "didn't you say Z before?" | Check consistency with earlier Z reference |
| CDF notation `F̃(X) = gX(FX(X))` | Related to above | Verify notation consistency |

### Page 20
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "For example" | Remove | Delete phrase |
| "needed for frugal parameterization" | "doesn't make much sense" | Rewrite or remove |

### Page 21
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "Z = an(Y)" | General cleanup needed | Review entire page |
| "ηG = (η1,...,ηD) ∈ V^D" | Part of general cleanup | Clarify notation |
| Entire page | "Maybe we need to give a general clean to pg 21" | Full page review |

### Page 22
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "de(Vi)" and "V = {V1,...,VI}" | Notation issues | Clarify |
| Omega (Ω) | "comes out of nowhere" | Introduce Ω properly before use |

### Page 23
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| Proof language | "seems unnatural where he's highlighted. how can we do better?" | Rewrite proof language |

### Page 24
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| T | "We've lost track of T at this stage. Unclear what it is" | Reintroduce/clarify T |
| "a topologically contiguous" | "is this subject to variations based on order?" | Address order dependence |
| "parents of the children of Q which are not in Q or T*" | "but between i and l, right?" | Clarify scope (between i and l) |
| "and l" | Unclear reference | Fix reference |

### Page 26
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "Premise 2(i)" and "Premise 1" | Should these be "Condition"? | Standardize terminology |

### Page 28
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "Figure 8" | Incorrect figure reference | Change to correct figure number (Figure 6?) |

### Page 29
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "(2i)" and "4(2i)" | Condition numbering consistency | Standardize |
| "However Condition 4 is not satisfied" | "There's a messiness in these paragraphs that need to be sorted" | Rewrite paragraphs for clarity |

### Page 41
| Highlighted Text | Comment | Action |
|------------------|---------|--------|
| "Replicate of Figure 8" | Same figure reference issue | Fix figure number |

---

## Summary by Category

### Quick Fixes (Mechanical)
- [ ] Remove unnecessary brackets throughout
- [ ] American spelling consistency check
- [ ] Fix figure references (Figure 8 → correct number)
- [ ] Remove "However", "again", "For example" where noted
- [ ] Remove specific commas
- [ ] Standardize "Premise" vs "Condition" terminology

### Notation/Clarity
- [ ] Fix copula notation rendering (pg 7)
- [ ] Fix duplicate V in Theorem 4.1 (pg 16)
- [ ] Introduce Ω before use (pg 22)
- [ ] Clarify T definition (pg 24)
- [ ] Topological ordering of variables in joint densities

### Substantive Revisions
- [ ] Address uniqueness question for minimal copulas (pg 8)
- [ ] Add parent theorem reference for Corollary 2.3 (pg 8)
- [x] Find and cite Lin et al. proof for Lemma 2 (pg 9) ✅
- [ ] Explain why "violation" (pg 12)
- [ ] Rewrite Definition 3.1 (pg 13)
- [ ] General cleanup of page 21
- [ ] Rewrite proof language (pg 23)
- [ ] Fix messy paragraphs around Condition 4 (pg 29)

### To Investigate
- [ ] Page 10 comment unclear
- [ ] Page 15 Lemma 3.3 issue unclear
- [ ] "approximate the marginal" vs Z consistency (pg 19)

---

## Next Steps

1. ~~**Find Lin et al. proof** for Lemma 2 (pg 9)~~ ✅ DONE
2. **Quick fixes pass** - brackets, spelling, figure refs
3. **Notation fixes** - V duplication, Ω introduction, T clarification
4. **Substantive rewrites** - Definition 3.1, pg 21, pg 23 proof, pg 29 paragraphs

---

*Last updated: 2026-02-10*
