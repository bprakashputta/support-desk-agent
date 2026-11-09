# 004: BM25 over the help center, with citations checked in code

**Status:** accepted

## Context
Customers ask policy questions ("where does my refund go?") as well as asking for actions. The help center here has 7 short articles, 22 passages, written from the tau-bench policy.

## Decision
- Retrieval is an in-process BM25 index over `##` sections, with stable passage ids (`returns#where-the-refund-goes`). There are no embeddings, no vector store and no second API key. At this corpus size, lexical retrieval is effectively exact and takes microseconds.
- The `search_help_center` tool returns passages tagged `[kb:<id>]`, and the prompt asks the model to cite them.
- `validate_citations` keeps a citation only if that id was actually returned by a search in this conversation. Anything else is stripped from the customer-facing reply and counted as `dropped_citations` in the trace. The stored model message is left untouched: the thinking blocks are bound to it (see ADR 006).

## Evidence
A test checks six paraphrased questions; each one retrieves the correct article at rank 1. The citation validator is unit-tested. There is no public ground-truth set for this help center, so there is no retrieval metric to report beyond that, and none is claimed.

## When to switch
At hundreds of articles, or with heavy paraphrase, move to hybrid retrieval (BM25 + embeddings + rerank), as in the author's `placewise` project. The tool interface and the citation check would not change.
