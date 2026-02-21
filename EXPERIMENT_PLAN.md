# Decoder‑Patching Experiment Plan (V‑PatchScope)

## 0. Purpose & Scope

Design a series of **language‑decoder hidden‑state patching experiments** that reveal how visual and textual evidence interact inside Vision–Language Models (VLMs). All experiments follow the two‑pass causal activation patching protocol (source ↔ target) and swap only **selected token segments** while matching total sequence length.

---

## 1. Shared Methodology Template

1. **Source inference**: ⟨_Is, Ts_⟩ → cache hidden states *H*ℓs at decoder layer ℓs.
2. **Target inference**: ⟨_It, Tt_⟩; at decoder layer ℓt, replace a chosen segment *S* in *H*ℓt with the cached segment, then continue forward pass.
3. **Layer grid**: ℓs, ℓt ∈ {early, mid, late} or full sweep.
4. **Segments (S)**: visual tokens \[0 … Nv‑1], single text token _j_, or masked span.
5. **Metrics**: flip rate, accuracy, hallucination suppression, sense switch, etc.

---

## 2. Experiment Catalogue

### 2.1 Visual Negation Patching

_Why?_ **Negation is a core linguistic operation that current VLMs routinely mishandle**—for example, answering "yes" to "Is there no dog?" even when the image is empty. By causally transplanting visual evidence of _presence_ into a _negation_ scenario, we pinpoint the decoder depth where the "object‑absent" signal lives—or where it gets overwritten by bias. Understanding this boundary is critical for fixing hallucinations tied to missing‑object reasoning.

| Item                | Details                                                         |
| ------------------- | --------------------------------------------------------------- |
| **Objective**       | Locate where the decoder encodes object **absence** (negation). |
| **Source**          | _Is‑pos_ (object present), _Ts‑pos_: "There is a **X**."        |
| **Target**          | _It‑neg_ (object removed), _Tt‑q_: "Is there a X? yes/no"       |
| **Patched segment** | Visual tokens (0…Nv‑1)                                          |
| **Eval**            | Does answer flip to **yes**? Record earliest (ℓs, ℓt).          |
| **Insight**         | Depth at which negation is grounded vs. overwritten by bias.    |

### 2.2 Polysemy / Pronoun Disambiguation

_Why?_ **Ambiguous words and pronouns require visual grounding**. Whether "bat" means an animal or sports gear—or whether "his" refers to a male or female doctor—depends on the image. Swapping just the ambiguous‑token state lets us localise the decoder layer where vision resolves linguistic uncertainty, revealing if grounding happens early (robust) or late (fragile).

| Item                | Details                                                            |
| ------------------- | ------------------------------------------------------------------ |
| **Objective**       | Identify layer where visual context resolves ambiguous word sense. |
| **Source**          | _Is‑A_ (baseball bat), _Ts_: "A **bat** lies …"                    |
| **Target**          | _It‑B_ (animal bat), _Tt_ identical                                |
| **Patched segment** | Single text token at index _j_ ("bat")                             |
| **Eval**            | Sense switch detected via keyword classifier.                      |
| **Insight**         | Whether fusion for semantics happens early or late.                |

### 2.3 Spatial-Relation Patching

_Why?_ **Left/right and other spatial prepositions remain a notorious weakness**—high‑accuracy VLMs often guess at chance. By patching only the mask token that should encode "left" vs. "right," we measure where (or if) the decoder truly captures spatial layouts, informing targeted improvements for spatial reasoning.

| Item                | Details                                                    |
| ------------------- | ---------------------------------------------------------- |
| **Objective**       | Map layers encoding "left‑of / right‑of" spatial relation. |
| **Source**          | _Is‑L_ (cat left), sentence with blank mask.               |
| **Target**          | _It‑R_ (cat right), same sentence.                         |
| **Patched segment** | Mask token at index _k_.                                   |
| **Eval**            | Blank output flips (left ↔ right).                         |
| **Insight**         | Strength & depth of spatial grounding in decoder.          |

### 2.4 Language-Bias Hallucination Patching

_Why?_ **Models frequently hallucinate objects suggested by language priors, ignoring the image.** Injecting real visual tokens into a no‑image run shows how deep visual evidence must travel before it suppresses hallucination. This quantifies the decoder's resilience to language bias and highlights layers to fortify.

| Item                | Details                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| **Objective**       | Discover depth where visual evidence suppresses textual hallucination. |
| **Source**          | _Is_ (empty table), _Ts_: generic caption.                             |
| **Target**          | _It‑Ø_ (blank image), _Tt_ same.                                       |
| **Patched segment** | Visual tokens (0…Nv‑1).                                                |
| **Eval**            | Presence/absence of hallucinated object (regex).                       |
| **Insight**         | Earliest layer that counteracts language prior.                        |

### 2.5 Image-Text Conflict (Counting / Label)

_Why?_ **Safety‑critical scenarios demand that visual truth overrules contradictory captions**, yet VLMs can still echo the text. Swapping numeral or label tokens between truthful and misleading prompts creates a tug‑of‑war; the layer where truth prevails (or fails) maps the decoder's modality‑dominance frontier and guides robustness training.

| Item                | Details                                                  |
| ------------------- | -------------------------------------------------------- |
| **Objective**       | Trace modality dominance when image contradicts caption. |
| **Source**          | Truthful caption ("five apples"), same image.            |
| **Target**          | Misleading caption ("three apples"), same image.         |
| **Patched segment** | Text token(s) of numeral ("five"/"three").               |
| **Eval**            | Answer changes to truthful count.                        |
| **Insight**         | Layer‑wise conflict resolution; robustness hotspot.      |

---

## 3. Implementation Checklist

- Align token indices across prompts with padding.
- Slice cached hidden states to minimise VRAM.
- Automate grid search over (ℓs, ℓt) with progress logging.
- Post‑process outputs with lightweight regex / classifiers.

## 4. Tentative Timeline (per experiment)

1 day dataset crafting → 2 days grid run → ½ day analysis & figures.

## 5. Expected Deliverables

- Heat‑map plots of flip‑rate over (ℓs, ℓt).
- Sample qualitative generations.
- One‑paragraph insight summary for each experiment.

---

_Prepared 15 Jun 2025 — Revise as experiments progress._
