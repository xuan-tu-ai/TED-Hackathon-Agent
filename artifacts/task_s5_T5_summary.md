# Task Summary: T5 — Update index.css with Bubble/Chip/HITL Styles

**Sprint:** Sprint 5
**Task:** T5

## Summary of Work
Added all new bubble/chip/hitl-inline CSS classes to `index.css` and removed old CollapsibleStep/HITLCard CSS classes. Deduplicated `@keyframes pulse`.

## Files Modified
- `frontend/src/styles/index.css` — Major CSS update
  - Added new classes: `.chat-step`, `.bubble-thinking`, `.bubble-thinking--running`, `.bubble-thinking-header`, `.bubble-thinking-pulse`, `.chip-tool`, `.chip-tool--running`, `.chip-tool-header`, `.chip-tool-icon`, `.chip-tool-label`, `.chip-tool-args`, `.chip-tool-status`, `.chip-tool-details`, `.chip-tool-summary`, `.chip-tool-result`, `.chip-tool-result--empty`, `.hitl-inline`, `.hitl-inline-header`, `.hitl-inline-body`, `.hitl-inline-row`, `.hitl-inline-label`, `.hitl-inline-action`, `.hitl-inline-severity`, `.hitl-inline-severity.severity--{low,medium,high,critical}`, `.hitl-inline-reasoning`, `.hitl-inline-actions`, `.hitl-inline-btn`, `.hitl-inline-btn--approve`, `.hitl-inline-btn--reject`
  - Removed old classes: `.chat-msg`, `.step-header`, `.step-marker`, `.step-action-badge`, `.step-action-icon`, `.step-action-input`, `.step-status`, `.step-status--running`, `.step-status--done`, `.step-details`, `.step-reasoning-summary`, `.step-observation-summary`, `.step-loading`, `.reasoning-box`, `.action-call`, `.observation-details`, `.observation-summary`, `.observation-content`, `.step-thought-container`, `.thought-header`, `.thought-content`, `.tool-call-container`, `.tool-call-block`, `.tool-icon`, `.tool-name`, `.tool-args`, `.result-container`, `.result-details`, `.result-summary`, `.result-content`, `.hitl-card`, `.hitl-header`, `.hitl-details`, `.hitl-field`, `.hitl-label`, `.hitl-value`, `.severity-high`, `.severity-medium`, `.severity-low`, `.hitl-reasoning`, `.hitl-actions`, `.btn-hitl`, `.btn-approve`, `.btn-reject`
  - Fixed `:contains('Done')` pseudo-class (not standard CSS) → `.chip-tool:not(.chip-tool--running) .chip-tool-status`
  - Preserved `@keyframes pulse-border` (used by `.incident-item.hitl_wait`) and `@keyframes pulse` (used by `.robot-cell.processing`)

## Testing
No frontend test framework — verification is by visual inspection in the running dev server.
