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
- [x] T1: Create `ThinkingBubble.jsx` — left-aligned gray bubble component.
- [x] T2: Create `ToolCallChip.jsx` — compact inline tool-call + expandable result component.
- [x] T3: Create `InlineHITLCard.jsx` — richer in-chat HITL approval card.
- [x] T4: Rewrite `ChatPanel.jsx` — compose new components, remove old ones.
- [x] T5: Add bubble + chip + HITL-inline CSS to `index.css`.
- [x] T6: Delete `frontend/src/components/CollapsibleStep.jsx` — fully replaced by `ThinkingBubble` + `ToolCallChip` composition in `ChatPanel`.
- [x] T7: Delete `frontend/src/components/HITLCard.jsx` — fully replaced by `InlineHITLCard`. Confirm no remaining imports.

## Notes / Blockers
- No frontend test framework — verification is by visual inspection in the running dev server.
- `TOOL_ICONS` map currently in `CollapsibleStep.jsx` must be copied to `ToolCallChip.jsx` before deletion (T6 depends on T2).
- The `observation` prop in `ToolCallChip` is `null` while the step is running (set to a string by `agent_step_update` SSE event). The chip must handle both states gracefully.
- `InlineHITLCard` receives `onResolve(queueId, action)` — same signature as current `HITLCard`, so `ChatPanel` wiring is unchanged.
- `ThinkingBubble` is purely presentational — no store interaction.
- Task order constraint: T6 must happen after T2 (TOOL_ICONS moved), T7 must happen after T4 (no more HITLCard import in ChatPanel).
