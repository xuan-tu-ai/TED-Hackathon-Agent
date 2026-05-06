---
name: implement-atomic-task
description: Use it to understand how to implement an atomic task. 
---

# Implement Atomic Task Skill

## Purpose
This skill dictates the standard operational procedure for implementing a single atomic task. It ensures that every code change is focused, thoroughly tested, and well-documented through artifact summaries.

## Workflow

### 1. Code Implementation
- Focus entirely on the single concern defined by the atomic task.
- Make the necessary code modifications without exceeding the scope of the task.
- Adhere to project guidelines (e.g., relative imports, execution via uv).

### 2. Test Creation
- Write an automated test (or update an existing one) to verify the new functionality.
- Ensure the test passes locally by running the appropriate test runner (e.g., uv run pytest tests/<test_file.py> -v).

### 3. Artifact Generation
- Once the code and tests are complete, create a task summary artifact at artifacts/task_<id>_summary.md.
- This ensures traceability and serves as a verified record of the completed work.
- See the template: [_ARTIFACT_SUMMARY_TEMPLATE.md](./references/_ARTIFACT_SUMMARY_TEMPLATE.md)

### 4. Completion Verification
- Mark the task as done [x] in the active Sprint Plan.
- Only after this review process should you propose committing the code.
