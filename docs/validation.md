# Validation record

This document separates observed behavior from intended behavior. It is a
compact review aid, not a substitute for rerunning the notebooks with valid API
credentials.

## API-backed reference run

The last full API-backed execution is preserved in Git commit
[`9cdecde`](https://github.com/trzhang-ai/game-research-agent/blob/9cdecde3ff3c6b5352e344367dcfd48c5277abe7/Udaplay_02_solution_project.ipynb).
It completed six routes without notebook execution errors:

| Check | Expected decision | Observed path | Result |
| --- | --- | --- | --- |
| Favorite FPS from stored context | Long-term memory | `classify_request → search_memory` | Answered from the matching memory fragment |
| Pokémon Gold and Silver release year | Local evidence sufficient | `classify_request → retrieve_game → evaluate_retrieval` | Returned 1999 and Game Boy Color |
| Platform follow-up in the same session | Existing conversation sufficient | `classify_request` | Reused prior evidence without new retrieval |
| First 3D Super Mario platformer | Local evidence insufficient for a superlative | `classify_request → retrieve_game → evaluate_retrieval → game_web_search` | Produced a source-linked web-backed answer |
| Mortal Kombat X on PlayStation 5 | Required game absent locally | `classify_request → retrieve_game → evaluate_retrieval → game_web_search` | Distinguished native release from backward compatibility and cited sources |
| Egg-boiling follow-up | Out of scope | `classify_request` | Declined without invoking game research |

That revision also contains a successful one-result semantic search over the
15-record local dataset in Part 1.

The current notebooks intentionally have cleared outputs. The portfolio refactor
changed retrieval plumbing, source-search depth, and supporting library code;
leaving the earlier outputs attached would imply that revised code had been
executed when it had not. To reproduce the integration scenarios, run
`01_game_index.ipynb` followed by `02_research_agent.ipynb` with the configured
project kernel. These cells send query and memory content to OpenAI and Tavily
and incur usage charges.

## Deterministic checks

The standard-library test suite covers behavior that does not require paid or
networked services:

- JSON-schema generation for nested tool parameters and route enums;
- session isolation and defensive copying in short-term memory;
- deterministic long-term-memory IDs and idempotent upsert behavior;
- explicit vector-store create, retrieve, and delete outcomes;
- state transitions after routing and retrieval evaluation;
- Unicode-preserving tool-result serialization; and
- notebook JSON integrity, Python syntax, and saved-output error checks.

Run the checks with:

```bash
uv run python -m unittest discover -s tests -v
```

## What remains unproven

- No benchmark dataset is used to estimate retrieval precision, routing
  accuracy, judge agreement, factual accuracy, latency, or cost.
- The LLM judge has not been calibrated against human labels.
- Web-source quality is assessed at answer time rather than by a deterministic
  allowlist or independent citation-verification service.
- The current revision has not yet received a fresh API-backed notebook run;
  its live behavior should be reconfirmed before a release claim is made.
- The current memory design demonstrates retrieval mechanics, not production
  privacy, authorization, retention, or deletion controls.
- The implementation has not been load-tested or deployed as a service.

These limits are deliberate: the repository demonstrates evidence-aware agent
orchestration without presenting a learning prototype as production software.

