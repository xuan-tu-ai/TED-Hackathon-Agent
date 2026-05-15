# Sprint 5 Plan

## Sprint Goal
Replace the flat `CollapsibleStep` card list in `ChatPanel` with a bubble-based chat UI — differentiated message roles (thinking / tool call / result / HITL) — so the diagnostic feed looks and feels like a modern AI assistant (Claude / ChatGPT style).

## Dependencies
- Sprint 2 (ChatPanel, CollapsibleStep, HITLCard, index.css)
- Sprint 4 T11 (InlineHITLCard replaces HITLCard; HITLCard.jsx will be deleted)

## Affected Files
```
frontend/src/components/
  ThinkingBubble.jsx        ← NEW
  ToolCallChip.jsx          ← NEW
  InlineHITLCard.jsx        ← NEW (replaces HITLCard.jsx)
  ChatPanel.jsx             ← REWRITE (drop CollapsibleStep, compose new components)
  CollapsibleStep.jsx       ← DELETE (replaced by composition of new components)
  HITLCard.jsx              ← DELETE (replaced by InlineHITLCard)
frontend/src/styles/
  index.css                 ← ADD new bubble/chip/hitl-inline CSS classes
```

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: New Components
- [ ] T1: Create `ThinkingBubble.jsx` — left-aligned gray bubble component.
  - Props: `reasoning: string`, `isRunning: bool` (true when `observation === null`)
  - Layout: header row `💭 Thinking` (+ pulse dot when `isRunning`), body with `reasoning` text
  - CSS class: `.bubble-thinking`

- [ ] T2: Create `ToolCallChip.jsx` — compact inline tool-call + expandable result component.
  - Props: `toolName: string`, `toolInput: string`, `observation: string | null`
  - Layout:
    - Top row (always visible): `{icon} {toolName}  ({truncatedInput, max 80 chars})` + status badge (`⏳ Running` or `✓ Done`)
    - Expandable result section below: shows `observation` text when available; collapsed by default if `observation.length > 300`, expanded if short
    - When `observation === null`: show `⏳ Processing…` placeholder inside result area
  - CSS classes: `.chip-tool`, `.chip-tool--running`, `.chip-tool-icon`, `.chip-tool-result`
  - Reuse `TOOL_ICONS` map from the deleted `CollapsibleStep.jsx`

- [ ] T3: Create `InlineHITLCard.jsx` — richer in-chat HITL approval card.
  - Props: `hitl: { queueId, action, severity, reasoning }`, `onResolve: fn`
  - Layout:
    - Header: `⚠️ Human Review Required`
    - Row: `Action` → action text in monospace/code style
    - Row: `Severity` → colored badge (`HIGH`=amber, `CRITICAL`=red, `LOW`=green, `MEDIUM`=blue)
    - Reasoning block: full text, scrollable if > 5 lines
    - Actions: `✓ Approve` (green, prominent) + `✗ Reject` (red, prominent)
  - CSS class: `.hitl-inline`, `.hitl-inline-header`, `.hitl-inline-severity--{low,medium,high,critical}`, `.hitl-inline-actions`
  - This replaces the existing `HITLCard.jsx`

### Phase 2: Rewrite ChatPanel
- [ ] T4: Rewrite `ChatPanel.jsx` — compose new components, remove old ones.
  - Remove imports of `CollapsibleStep` and `HITLCard`
  - Import `ThinkingBubble`, `ToolCallChip`, `InlineHITLCard`
  - Per `agentStep` in the feed: render `<ThinkingBubble>` then `<ToolCallChip>` inside a `.chat-step` wrapper div
  - Per `hitlItem`: render `<InlineHITLCard>`
  - Keep `<PlanTracker>`, `<AgentHeader>`, auto-scroll ref, and empty-state unchanged

### Phase 3: CSS Update
- [ ] T5: Add bubble + chip + HITL-inline CSS to `index.css`.
  - New classes: `.chat-step`, `.bubble-thinking`, `.bubble-thinking--running`, `.bubble-thinking-header`, `.bubble-thinking-pulse`, `.chip-tool`, `.chip-tool--running`, `.chip-tool-icon`, `.chip-tool-label`, `.chip-tool-status`, `.chip-tool-result`, `.chip-tool-result--empty`, `.hitl-inline`, `.hitl-inline-header`, `.hitl-inline-row`, `.hitl-inline-severity--low`, `.hitl-inline-severity--medium`, `.hitl-inline-severity--high`, `.hitl-inline-severity--critical`, `.hitl-inline-reasoning`, `.hitl-inline-actions`
  - Remove old classes: `.chat-msg`, `.step-header`, `.step-marker`, `.step-status`, `.step-status--running`, `.step-status--done`, `.step-thought-container`, `.thought-header`, `.thought-content`, `.tool-call-container`, `.tool-call-block`, `.tool-icon`, `.tool-name`, `.tool-args`, `.result-container`, `.result-details`, `.result-summary`, `.result-content`, `.step-loading`, `.hitl-card`, `.hitl-header`, `.hitl-details`, `.hitl-field`, `.hitl-label`, `.hitl-value`, `.hitl-reasoning`, `.hitl-actions`

### Phase 4: Cleanup
- [ ] T6: Delete `frontend/src/components/CollapsibleStep.jsx` — fully replaced by `ThinkingBubble` + `ToolCallChip` composition in `ChatPanel`.
- [ ] T7: Delete `frontend/src/components/HITLCard.jsx` — fully replaced by `InlineHITLCard`. Confirm no remaining imports.

## Notes / Blockers
- No frontend test framework — verification is by visual inspection in the running dev server.
- `TOOL_ICONS` map currently in `CollapsibleStep.jsx` must be copied to `ToolCallChip.jsx` before deletion (T6 depends on T2).
- The `observation` prop in `ToolCallChip` is `null` while the step is running (set to a string by `agent_step_update` SSE event). The chip must handle both states gracefully.
- `InlineHITLCard` receives `onResolve(queueId, action)` — same signature as current `HITLCard`, so `ChatPanel` wiring is unchanged.
- `ThinkingBubble` is purely presentational — no store interaction.
- Task order constraint: T6 must happen after T2 (TOOL_ICONS moved), T7 must happen after T4 (no more HITLCard import in ChatPanel).
