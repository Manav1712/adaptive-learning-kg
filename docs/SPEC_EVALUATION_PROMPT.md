# Product Spec Section Evaluation Prompt

Use this prompt to evaluate any section of `PRODUCT_SPEC.md`. Replace `[SECTION_TO_EVALUATE]` with the actual section content.

---

## Evaluation Prompt

```
You are an expert software engineer reviewing a product specification document. Your task is to evaluate the following section for completeness, clarity, implementability, and technical accuracy.

**Section to Evaluate:**

[SECTION_TO_EVALUATE]

**Context:** This is part of a Jupyter Notebook Pipeline Specification for building an adaptive learning tutoring system. The spec guides an engineer to implement the system by modifying a single Jupyter notebook cell step-by-step. The system includes:
- Coach agent (routing, plan state management)
- Retriever (embedding-based search with text + image)
- Planner (generates simplified plan JSON)
- Tutor bot (teaches from plan, handles off-plan requests)

**Evaluation Criteria:**

1. **Completeness**
   - Are all required inputs, outputs, and processes clearly defined?
   - Are there any missing steps or gaps in the logic flow?
   - Are edge cases or error conditions addressed?
   - Are all referenced functions, variables, or data structures defined elsewhere in the spec?

2. **Clarity & Readability**
   - Can an engineer understand what to do without external context?
   - Are technical terms explained or linked to definitions?
   - Is the language precise and unambiguous?
   - Are code examples syntactically correct and complete?

3. **Implementation Feasibility**
   - Can an engineer implement this section following only the instructions here?
   - Are all required dependencies, imports, or setup steps mentioned?
   - Are data formats, schemas, or contracts specified exactly?
   - Are there any implicit assumptions that should be made explicit?

4. **Technical Accuracy**
   - Are the technical details correct (algorithms, data structures, API calls)?
   - Do code snippets match the described behavior?
   - Are version numbers, model names, or configuration values accurate?
   - Are mathematical formulas or algorithms correctly stated?

5. **Examples & Concreteness**
   - Are there concrete examples with real data (not placeholders like "...")?
   - Do examples show the complete flow from input to output?
   - Are example outputs realistic and match the described format?
   - Would examples help an engineer validate their implementation?

6. **Consistency**
   - Does this section align with other sections of the spec?
   - Are naming conventions consistent (function names, variable names, JSON keys)?
   - Do data contracts match between sections (e.g., plan JSON schema)?
   - Are model names, paths, or constants consistent throughout?

7. **Missing Information**
   - What critical information is missing that would block implementation?
   - What questions would an engineer have after reading this section?
   - Are there any "TODO" or "TBD" items that need resolution?
   - Are error handling or validation steps missing?

**Your Response Format:**

Provide a structured evaluation with:

1. **Overall Assessment** (1-2 sentences)
   - Is this section ready for implementation, or does it need work?

2. **Strengths** (bullet points)
   - What does this section do well?

3. **Critical Issues** (bullet points, prioritized)
   - What must be fixed before this is implementable?
   - Include specific line numbers or code snippets if possible

4. **Suggestions for Improvement** (bullet points)
   - What would make this section clearer or more complete?
   - Include concrete examples of how to improve

5. **Missing Information** (bullet points)
   - What information is absent that should be added?

6. **Specific Recommendations** (if applicable)
   - Concrete edits, additions, or clarifications with examples

**Be specific and actionable.** Point to exact lines, provide example fixes, and prioritize issues by severity (blocking vs. nice-to-have).
```

---

## Usage Instructions

1. **Copy the evaluation prompt above**
2. **Replace `[SECTION_TO_EVALUATE]`** with the actual section content from `PRODUCT_SPEC.md`
3. **Paste into your AI assistant** (Claude, ChatGPT, etc.)
4. **Review the structured evaluation** and address critical issues first

## Example Usage

**For evaluating Step 5 (Runtime Retrieval):**

1. Copy the section starting with `## Step 5 — Runtime retrieval...` through the end of that step
2. Replace `[SECTION_TO_EVALUATE]` with that content
3. Run the prompt through your AI assistant
4. Review the evaluation and update the spec accordingly

**For evaluating Step 9 (Examples):**

1. Copy the entire `## Step 9 — Fully worked example...` section
2. Replace `[SECTION_TO_EVALUATE]` with that content
3. The AI will check if examples are detailed enough, have real data, and show complete flows

---

## Evaluation Checklist (Quick Reference)

Before submitting a section for evaluation, quickly check:

- [ ] All code snippets are syntactically valid Python
- [ ] All JSON examples are valid JSON (no `"..."` placeholders)
- [ ] All function signatures include parameter types and return types
- [ ] All file paths are clearly specified
- [ ] All environment variables are documented
- [ ] All error cases are handled or explicitly deferred
- [ ] Examples show complete input → output flows
- [ ] Technical terms are defined or linked
- [ ] Dependencies are listed with versions
- [ ] Data schemas match across sections

