# Task Summary: T11 — Fix Double HITL Approval UI

**Sprint:** Sprint 4 (Phase 6)
**Task:** T11

## Summary of Work
When a `hitl_request` SSE event fires, `App.jsx` stores the item in both `globalHitlQueue` (driving `HITLModal`) and the incident's `hitlItems` (driving `HITLCard` in `ChatPanel`). After the user approved/rejected in the modal, only `removeGlobalHitl` was called — the `HITLCard` remained in the chat panel, forcing a second click.

Fixed by importing `removeHitlItem` from the Zustand store in `HITLModal` and calling it with `(current.incident_id, queueId)` immediately after `removeGlobalHitl` in both `handleApprove` and `handleReject`. The call is inside the success path (after `response.ok` is confirmed), so no change on failure — the modal stays open if the API call fails.

## Files Modified
* [frontend/src/components/HITLModal.jsx](../frontend/src/components/HITLModal.jsx) — destructure `removeHitlItem` from store; call it after approve and after reject

## Testing
* **Test File:** No automated frontend test framework (no Jest/Vitest in `package.json`).
* **Verification:** Code-review — `removeHitlItem(current.incident_id, queueId)` is only reachable after `response.ok` is true, exactly mirroring the `removeGlobalHitl` call added in the same PR. The store's `removeHitlItem` filters by `h.queueId !== queueId`, which matches the camelCase key used by `addHitlItem`.
* **Status:** Verified by inspection.

## Additional Notes
`HITLCard` in ChatPanel remains as a visual history item between the `hitl_request` event and the user's decision. After resolution it is now immediately removed, matching the modal's behaviour.
