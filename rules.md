# AI Agent Rules and Operational Contract

Purpose: This document defines concise, actionable rules for AI agents working on the ds_task_lab project. The rules convert the project README into clear constraints, input/output contracts, safety requirements, and coding guidelines so agents can act autonomously and safely.

## 1. Scope & Role
- Agents will assist in implementing modules described in the project: data preparation / backend, OCR & scraping, CNN training, and frontend integration.
- Agents may create code, tests, documentation, and short demonstration artifacts (reports, short videos metadata). Agents must not publish or exfiltrate secrets, credentials, or PII.

## 2. High-level rules (always follow)
1. Follow the explicit contracts for each endpoint (see Section 4). Produce outputs in the specified JSON-like structure.
2. Prefer class-based, modular implementations for backend services. Keep single responsibility per class.
3. Ensure data integrity: database operations must follow ACID principles where applicable.
4. Do not use pre-trained models for the CNN task — train from scratch using only the data in `CNN_Model_Train_Data.csv` and the scraped images for the `products` listed there.
5. Filter and redact personally identifiable information (PII) from any scraped or uploaded data before storing or returning results.
6. Respect scraping policies and robots.txt; avoid scraping private or restricted data. If legality is unclear, stop and request human instruction.

## 3. Security & Privacy
- Never include API keys, passwords, or other secrets in code or responses. If credentials are necessary, instruct maintainers to provide them via environment variables or secure vaults.
- For user-uploaded images or text, detect and redact sensitive content from outputs.

## 4. Endpoint Contracts (strict)
All endpoints should return JSON with a stable schema. If an endpoint cannot fulfill the request, return an error object with a clear code and user-facing message.

- Product Recommendation Endpoint (text queries)
  - Input: { "query": string }
  - Output: {
      "product_matches": [ { "id": string, "score": number, "metadata": {...} } ],
      "natural_language_response": string
    }
  - Requirements: Validate input length and safety; never reveal internal vector DB details or expose raw vectors.

- OCR-Based Query Endpoint (image -> text -> recommendation)
  - Input: { "image": <binary/file> }
  - Output: {
      "ocr_text": string,
      "product_matches": [...],
      "natural_language_response": string
    }
  - Requirements: Return OCR text along with recommendations. If OCR fails, return a descriptive error.

- Image-Based Product Detection Endpoint (CNN)
  - Input: { "image": <binary/file> }
  - Output: {
      "class": string,            # predicted class from CNN
      "product_description": string,
      "product_matches": [...]
    }
  - Requirements: Return predicted `class` and match via vector DB. Ensure consistent formatting with other endpoints.

## 5. Data Preparation Rules
- Clean and normalize product data: remove duplicates, fill or document missing values, standardize categories and text fields.
- When creating vectors, persist metadata (product id, name, source URL) with each vector for traceability.
- Log and report the sampling strategy and any pruning performed on the dataset.

## 6. OCR / Scraping Rules
- Use Tesseract or a similar open OCR library for text extraction; tune preprocessing (grayscale, thresholding) to improve accuracy.
- For web scraping: limit request rate, follow robots.txt, and store images with a deterministic directory structure. Document the scraping pipeline and sample sizes.

## 7. CNN Training Rules
- Train the CNN from scratch using only the `products` listed in `CNN_Model_Train_Data.csv` and scraped images tied to those classes.
- Do not use transfer learning or pre-trained weights unless explicitly authorized in writing.
- Maintain training logs: hyperparameters, dataset splits, number of epochs, validation metrics, final model size, and class mapping.

## 8. Frontend Integration Rules
- Provide three frontend pages with the features described in README: text query, image query (handwritten OCR), and product image upload.
- API responses shown on frontends must use the canonical output schema above.

## 9. Reporting & Video Documentation
- For each module, produce a concise incremental report with: title page, brief intro, high-level flow/diagrams, key decisions, challenges & solutions, and conclusion.
- Produce two videos per module: a functional demo (<5 minutes) and a code explanation (<10 minutes). Supply upload instructions (preferred format: MP4, unlisted link) and a short text summary for reviewers.

## 10. Coding & Testing Rules
- Use Flask for the backend. Follow project convention and package structure.
- Keep notebooks in the `notebook` directory with clear names. Use `requirements.txt` for dependencies.
- Write unit tests for core logic (data cleaning, vectorization, OCR wrapper, CNN model input/output conversion) and at least one integration/smoke test per endpoint.

## 11. Error Handling and Observability
- Return structured error objects with codes and human-friendly messages.
- Log errors and key events (ingestion, training, endpoint calls) with timestamps and minimal necessary context; avoid logging sensitive data.

## 12. Deliverables & Acceptance Criteria
- Each module is accepted when:
  1. Code passes unit tests and a local smoke test for endpoints.
  2. Incremental report and videos are uploaded and linked.
  3. The model and vector DB are reproducible with the provided scripts and documented hyperparameters.

## 13. When to escalate to humans
- If scraping legality is ambiguous, if requested to use external paid services or credentials not provided, or if data contains PII requiring special handling, stop and request human input.

## 14. Quick enforcement checklist for agents
1. Validate input and sanitize outputs.
2. Use class-based modules and keep logic testable.
3. Follow endpoint output schemas exactly.
4. Document and log decisions and metrics.
5. Redact secrets and PII; escalate on legal/ethical uncertainty.

---
Notes: This file is derived from the project `README.md` and formatted as a ruleset for autonomous AI agents. If you need narrower or more permissive rules (for example, allowing transfer learning or different hosting), add an explicit human-signed exception section.
