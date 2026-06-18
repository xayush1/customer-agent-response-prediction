
## Interpreting your results

**OVERALL: cosine=0.893, judge=5.00/5** — this is excellent.

| Case | Cosine | Judge | Interpretation |
|---|---|---|---|
| Chat #1054 (French) | 0.998 | 5/5 | Near-identical |
| memcyco.com SOC | 1.000 | 5/5 | Exact match retrieved |
| neighbourhood.com | **0.791** | 5/5 | Wording differed but intent correct — cosine is the weak signal here |
| Hiver Workflow | 1.000 | 5/5 | Exact match |
| Hiver Support depth=7 | **0.585** | 5/5 | Lowest cosine — actual reply was 1 line ("Looking forward to working with you"), predicted was longer but still correct |
| Upgrade to Elite | 1.000 | 5/5 | Exact retrieval |
| WooCommerce depth=11 | 0.773 | 5/5 | Added extra detail, still correct |
| Chat #529 | 1.000 | 5/5 | Exact match |

