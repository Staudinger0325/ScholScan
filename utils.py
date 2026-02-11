intra_generate_prompt="""
You will receive a high-quality, already accepted scientific paper as a PDF. Working only with the PDF itself (and any appendix embedded in the same PDF), edit specific textual spans to inject one or more errors chosen only from the taxonomy below, such that the errors are hard yet clearly identifiable by a professional reviewer reading the PDF alone.

Error Type (fixed):
Research Question & Definitions (A)
    Definition: The core construct/hypothesis/variable is insufficiently or inconsistently defined (conceptual vs operational), leaving the estimand ambiguous.
    Example: Methods define the construct as time-on-task (dwell time), Eq.(1) optimizes a loss over clicks/session (proxy), Table 2 header reports “engagement (clicks/session),” Figure 3 caption interprets coefficients per click, and Notation table reuses E for two different constructs without reconciliation.
Design & Identifiability (B)
    Definition: Given a clear estimand, the design violates structural identification conditions so the effect is not identifiable even with infinite data and perfect measurement.
    Example: Methods adjust for a post-treatment mediator and set exposure at first prescription while eligibility requires survival to day 30 (immortal time); Assumption A2 (exclusion for IV / continuity for RD / parallel trends for DID) is weakened or removed in Appendix A; Theorem 1 claims “ATE identified” yet Results/Abstract retain strong causal wording without reporting first-stage/continuity/trend diagnostics.
Sampling & Generalizability (C)
    Definition: The sampling frame/process/composition or cluster/power setup does not support valid or stable sample→population claims.
    Example: Keep “two universities” as the sample but change the claim to “effective in the population”; or reduce “12 schools (6/arm)” to “4 schools (2/arm).”
Measurement & Operationalization (D)
    Definition: Measures/manipulations lack feasibility/reliability/validity/timing, so observed variables systematically diverge from the intended construct/treatment.
    Example: Measures replace a validated 10-item scale (α=0.86) with a single-item proxy, manipulation check p=0.08 is labeled “adequate” in Results, the “long-term” outcome is assessed immediately post-intervention, and Appendix C drops reliability/validity metrics (e.g., CFA/ICC), while Discussion still infers durable construct change.
Data Handling & Preprocessing (E)
    Definition: Pipeline choices in missing handling, joins/keys, temporal splitting, feature construction, or partitioning introduce bias (incl. leakage or unit/scale conflicts).
    Example: Methods flip to standardize-then-split and perform global feature selection using all data; Validation switches to random k-fold for a forecasting task; Table S2 shows N increasing by 2–5% vs Table 1 with no flow; Results still report a single held-out test score and SOTA comparison.
Computation & Formulae (F)
    Definition: Arithmetic/algebra/notation errors (totals/ratios, unit conversion, CI vs point estimate, p-value vs label, symbol reuse, undefined variables, dimension mismatch).
    Example: Methods state a natural-log transform (ln), Eq.(1) uses \log_{10}, Table 2 still reports Δ% = exp(β) − 1, Figure 3 labels log10(response), and the Notation table defines log as base-10, creating a base–inverse mismatch across formula/table/axis; Or: change concentration units to μg/L, Eq.(3) drops the ×1000 factor (C_{μg/L} = C_{mg/L}), Table 2 header shows μg/L but totals remain unchanged, Figure 4 axis uses μg/L, and Results compare to a 10 mg/L guideline, producing a systematic scale inconsistency.
Inference & Conclusions (G)
    Definition: Interpretations or causal statements exceed what methods/data support, or contradict the shown statistics/tables/captions.
    Example: Discussion upgrades “associated with” to “caused by,” Abstract claims durable OOD benefit and policy relevance, while Methods are observational (no randomization/IV/RD) and Results report only short-term in-distribution metrics.
Internal Consistency (H)
    Definition: Contradictions about the same quantity/term across text, tables, captions, or appendix within the paper.
    Example: Abstract N=312, Methods N=321, Table 1 N=320; the same metric is labeled “decrease” in a caption but “increase” in Results.
Language & Expression (I)
    Definition: Terminology/capitalization/grammar ambiguities that affect meaning or domain-critical term consistency (not cosmetic typos).
    Example: The same method appears as “Bootstrap/BootStrap/boot strap” in Methods/Results/Caption.

Global constraints (must comply)
1. Each error must map to exactly one primary category in the taxonomy. Do not mix causes.
2. Each error must involve more than 2 micro-edits (each edit ≤ 20 English words) spread across distinct pages or paragraphs.
3. If an edit would create an immediate contradiction in the same sentence/paragraph/caption, you may add shadow patch(es) for the same error to keep the text natural (still counted as edit locations).
4. Independence across errors (per-copy generation)
    Generate each error on a separate copy of the original PDF. Different errors must be logically and operationally independent:
    No progression or variant relations: an error must not be a stricter/looser version, superset/subset, or minor wording variant of another error.
    No anchor reuse: do not target the same sentence/caption/table cell or reuse the same old_str (or a near-duplicate paraphrase) across different errors.
    Applying any single error in isolation to the original PDF must still yield a detectable, clearly categorizable error according to the taxonomy.
5. Every error must be supportable using text inside the PDF. Do not rely on external supplementary files or prior knowledge.
6. Design as difficult as possible but clean errors. Prefer edits that force cross-checking between two spots (e.g., Methods vs Results). Avoid trivialities. Edits must remain locally plausible and not advertise themselves via obviously artificial phrases (e.g., avoid contrived tokens purely added to be detectable).
7. “No cosmetic issues” applies except for I (Language & Expression). For I, edits must affect meaning or domain-critical terminology (e.g., ambiguous phrasing, inconsistent technical terms). Pure typos, punctuation tweaks, or layout nits are not allowed.
8. Do not edit titles, author lists, bibliography entries, equation numbering, figure images, or add new figures/tables/references.
9. Frame each question as a neutral imperative that asks for a decision about a specific condition, using (but not limited to) Decide/Determine/Judge/Evaluate/Assess whether …. Do not presuppose an outcome or use suggestive intensifiers (e.g., clearly/obviously/likely/suspicious as examples).
10. Output English-only and strictly follow the JSON schema below. Do not include any additional text outside the JSON:

[
  {
    "id":"1-based integer as string",
    "modify":[
      {
        "location":"Page number + short unique nearby quote (≤15 tokens).",
        "old_str":"Exact original text from the PDF (verbatim).",
        "new_str":"Edited text after your change."
      }
      /* Add 1–2 more locations; each location ≤ 20 words changed.
         Shadow patches for local coherence count as locations. */
    ],
    "question":"One neutral audit-style task (1–25 words).",
    "explanation":"Explain in 2–4 sentences why a reviewer can detect this error from the edited PDF alone.",
    "Type":"Name the primary category (e.g., Inference & Conclusions).",
  }
  /* More Errors */
]
"""

intra_sample_prompt="""
You will receive a paper PDF and the weaknesses mentioned in its peer-review comments. Your task is, based only on the content of that PDF, to sample from the review comments and verify possible errors related to the categories below, and for each confirmed or highly plausible error, generate one question and one explanation.
Error definitions:
Visualization & Presentation Bias
Definition: Visual design choices (e.g., scale, axis truncation/breaks, dual axes, colormap, ordering, facet ranges, chart type) systematically mislead interpretation or exaggerate/attenuate effects.
Example: Use a non-zero y-axis baseline or an axis break that magnifies small differences; or compare panels that use inconsistent y-ranges/colormaps, inflating perceived gains across subplots.
Design & Identifiability
Definition: Given a clear estimand, the design violates structural identification conditions so the effect is not identifiable even with infinite data and perfect measurement.
Example: Add “adherence during follow-up” to adjusted covariates; or require survival to day 30 while follow-up starts at treatment (immortal time).
Sampling & Generalizability
Definition: The sampling frame/process/composition or cluster/power setup does not support valid or stable sample→population claims.
Example: Keep “two universities” as the sample but change the claim to “effective in the population”; or reduce “12 schools (6/arm)” to “4 schools (2/arm).”
Measurement & Operationalization
Definition: Measures/manipulations lack feasibility/reliability/validity/timing, so observed variables systematically diverge from the intended construct/treatment.
Example: Replace “validated 10-item scale (α=0.86)” with “single-item proxy”; or assess a “long-term” outcome immediately post-intervention.
Data Handling & Preprocessing
Definition: Pipeline choices in missing handling, joins/keys, temporal splitting, feature construction, or partitioning introduce bias (incl. leakage or unit/scale conflicts).
Example: “Standardize the dataset, then split into train/test”; or use random k-fold CV for a forecasting task.

Global constraints (must comply)
Output only the specified categories; even if other error types appear in the reviews, do not output them.
Sample first, then verify: extract candidates from the review comments, then confirm them in the PDF. If you cannot locate supporting anchors in the PDF (page number plus phrase/label), do not output that candidate.
Questions must be neutral and non-leading: use an “audit task + decision” style, avoiding yes/no bias.
Independence: each question must target a different figure or different textual anchor; no minor variants of the same issue.
Evidence first: the explanation must cite locatable anchors in the PDF (page number + original phrase/caption). You may mention a key short phrase from the review as a clue, but write the question and explanation in your own words.
Language & format: both question and explanation must be in English; output JSON only, with no extra text.
Quantity: sort by evidence strength and output up to 5 items; if none qualify, output an empty array [].
Example output
[
  {
    "id": "1",
    "question": "Audit y-axis baselines and possible axis breaks in Figure 2; decide presence/absence and cite evidence.",
    "explanation": "The review flags possible exaggeration in Fig.2. In the PDF (p.6, caption 'Performance vs baseline'), the y-axis starts at 0.85 with a break, magnifying small differences; panels use different ranges."
        "Type":"Visualization & Presentation Bias"
  }
]
"""
extractor_prompt="""
You will receive three inputs:
Q: the open-ended question;
E: the gold explanation (describes exactly one error; extra details still belong to the same single error);
A: the model’s answer to be evaluated.
Your job is to extract counts only and output a single JSON object with the exact schema below. Do not compute any scores. Do not add fields.

Core selection rule (multiple errors in A)
1. Parse E into a single gold error (the “target error”).
2. From A, identify how many distinct error claims are made. Cluster together mentions that support the same error (multiple locations for one error are still one error).
3. Existence decision (binary correctness only):
Let the gold existence be 1 if E asserts an error exists, else 0.
Let the predicted existence be 1 if A asserts any error, else 0 (e.g., states no error).
Set existance = 1 if predicted existence equals gold existence; otherwise set existance = 0.
4. If existance = 0: set contains_target_error = 0; set all location and reasoning counts to 0; and set unrelated_errors to the total number of distinct error claims in A. Then output the JSON.
5. If existance = 1:
If the gold existence is 1: determine whether A contains the target error (match by the main error idea in E: category/intent/scope; treat E’s subpoints as the same error).
    If yes, set contains_target_error = 1 and compute location and reasoning only for the target error. Count all other error claims in A as unrelated_errors.
    If no, set contains_target_error = 0; set all location and reasoning counts to 0; set unrelated_errors to the total number of distinct error claims in A.
If the gold existence is 0: set contains_target_error = 0; set all location and reasoning counts to 0; set unrelated_errors to the total number of distinct error claims in A. (These negative items are for binary accuracy only; they are not used for detailed scoring.)
6. If E doesn't contain anything, set existance = 0 and follow the Rule 4.

Matching guidance (A error ↔ target error): match by the main error idea in E (category/intent/scope), not by wording. Treat E’s subpoints as part of the same single error. Prefer the best-matching cluster in A; if ties, choose the one with stronger alignment to E’s core claim.

Counting rules
Location (for the target error only when existance=1 and contains_target_error=1):
gold_steps: number of unique error locations described in E (after normalization and deduplication).
hit_steps: number of predicted locations in A that match any gold location for the target error.
extra_steps: number of predicted locations in A for the target error that do not match any gold location.

Reasoning (for the target error only when existance=1 and contains_target_error=1):
Convert E into a canonical set or ordered chain of reasoning steps for the target error.
gold_steps: total number of such steps.
reached_steps:
    single-chain tasks: length of the longest valid prefix of A along the gold chain;
    multi-path/parallel tasks: size of the intersection between A’s steps and the gold step set (or the maximum across gold paths if multiple are defined).
missing_steps: gold_steps - reached_steps (non-negative integer).
Unrelated errors:
unrelated_errors: number of distinct error claims in A that are not the target error (0 if none).
Output schema (return exactly this JSON; integers only)
{
  "existance": 0,
  "contains_target_error": 0,
  "location": {
    "gold_steps": 0,
    "hit_steps": 0,
    "extra_steps": 0
  },
  "reasoning": {
    "gold_steps": 0,
    "reached_steps": 0,
    "missing_steps": 0
  },
  "unrelated_errors": 0
}
"""
system_prompt="""
You are a neutral, careful academic reviewer. You will receive an open-ended question and the paper content. The paper may or may not have issues related to the question. Do not assume there are errors. If the question is about citations, you will be given a citing paper and a cited paper; evaluate only the citing paper for possible issues and use the cited paper only as the reference for comparison. Write in natural prose with no fixed template.

Rules:
Speak only when sure. State an error only if you are confident it is a real error (not a mere weakness).
Stay on scope. Discuss only what the question asks about.
Evidence completeness. For every error you state, list all distinct evidence cues you are confident about from the PDF. Include plain identifiers (figure/table/section/equation/citation) or quotes. Avoid redundant repeats of the exact same instance; include all distinct locations needed to support the error.
Be clear and brief. Use short, direct sentences. No metaphors. No fancy wording.
No guesses or outside sources. Do not invent figures, tables, equations, citations, or results.
Report as many distinct, well-supported errors as you can within scope. If none are clear, write exactly: “No clear issue relevant to the question.” and nothing else.
"""
Definition_A="""
Definition of "Research Question & Definitions":
The core construct/hypothesis/variable is insufficiently or inconsistently defined (conceptual vs operational), leaving the estimand ambiguous.
"""
Definition_B="""
Definition of "Design & Identifiability":
Given a clear estimand, the design violates structural identification conditions so the effect is not identifiable even with infinite data and perfect measurement.
"""
Definition_C="""
Definition of "Sampling & Generalizability":
The sampling frame/process/composition or cluster/power setup does not support valid or stable sample→population claims.
"""
Definition_D="""
Definition of "Measurement & Operationalization":
Measures/manipulations lack feasibility/reliability/validity/timing, so observed variables systematically diverge from the intended construct/treatment.
"""
Definition_E="""
Definition of "Data Handling & Preprocessing":
Pipeline choices in missing handling, joins/keys, temporal splitting, feature construction, or partitioning introduce bias (incl. leakage or unit/scale conflicts).
"""
Definition_F="""
Definition of "Computation & Formulae":
Arithmetic/algebra/notation errors (totals/ratios, unit conversion, CI vs point estimate, p-value vs label, symbol reuse, undefined variables, dimension mismatch).
"""
Definition_G="""
Definition of "Inference & Conclusions":
Interpretations or causal statements exceed what methods/data support, or contradict the shown statistics/tables/captions.
"""
Definition_H="""
Definition of "Internal Consistency":
Contradictions about the same quantity/term across text, tables, captions, or appendix within the paper.
"""
Definition_I="""
Definition of "Language & Expression":
Terminology/capitalization/grammar ambiguities that affect meaning or domain-critical term consistency (not cosmetic typos).
"""