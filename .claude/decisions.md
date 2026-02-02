# Decisions Log

> Document architecture, model, statistical, and design decisions with rationale.

---

## 2026-02-02 - Initialize Context Management
**Type:** Architecture
**Status:** Accepted

### Context
Setting up a new project with comprehensive tracking for frugalCopyla - a research project with both paper and code components.

### Decision
Use the standard context management system with scratchpad, plan, handoff, errors, decisions, codebase-map, and references files.

### Rationale
Enables session continuity, traceable decisions, and verifiable sources. Critical for managing dual Python/R codebase aligned with academic paper.

---

## 2026-02-02 - Create CLAUDE.md for Paper-Code Context
**Type:** Documentation
**Status:** Accepted

### Context
Project has a comprehensive academic paper (Hybrid-Frugal-Paper) that needs to be tightly linked to the codebase. Standard codebase-map.md insufficient for capturing mathematical definitions, theorems, and paper structure.

### Decision
Create comprehensive CLAUDE.md file (500+ lines) containing:
- Complete paper summary with key contributions
- Full terminology glossary (frugal, feasible, natural, h-functions, vines)
- Mathematical notation reference
- Code-paper mapping showing which files implement which sections
- Section-by-section breakdown with key theorems

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| Just use codebase-map.md | Simpler | Too limited for academic context |
| Separate paper-summary.md | Clean separation | Harder to find code-paper links |
| Inline in code comments | Close to implementation | Fragmented, hard to get overview |

### Rationale
Need single source of truth that connects abstract mathematical concepts in paper with concrete implementations in code. Future sessions need to quickly understand what "feasible" or "h-function" means without re-reading the entire paper.

### Consequences
- CLAUDE.md is now required reading for any code changes
- Must keep CLAUDE.md updated if paper changes
- Larger context overhead but massive time savings on understanding

### Related Files
- `/Users/danielmanela/Library/CloudStorage/GoogleDrive-danielmanela@gmail.com/My Drive/work/Oxford/frugalCopyla/CLAUDE.md` — The context file
- `/Users/danielmanela/Library/CloudStorage/GoogleDrive-danielmanela@gmail.com/My Drive/work/Oxford/frugalCopyla/Hybrid-Frugal-Paper/main.tex` — Paper source

---
