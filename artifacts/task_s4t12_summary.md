# Task Summary: T12 — Fix SSE ETIMEDOUT on First Click

**Sprint:** Sprint 4 (Phase 6)
**Task:** T12

## Summary of Work
`handleStart()` in `App.jsx` previously called `createSSEConnection().connect()` as its first action, before the `POST /api/sim/start` request was sent. If the backend was not yet running, the Vite proxy immediately surfaced an `ETIMEDOUT` error on `/api/sim/stream`, which appeared in the client error panel and led users to believe the whole start had failed — even though the retry loop for `/start` was still in progress.

Fixed by moving the `createSSEConnection` + `.connect()` call to **inside the success path** of the retry loop, immediately after `response.ok` is confirmed for `/api/sim/start`. The `EventSource` is now only created when the backend has already confirmed it is running, eliminating the spurious ETIMEDOUT.

## Files Modified
* [frontend/src/App.jsx](../frontend/src/App.jsx) — moved `sseRef.current = createSSEConnection(...)` and `.connect()` into the `if (result.ok)` success branch; removed the `sseRef.current.disconnect()` call from the exhausted-retries error path (no connection to clean up at that point).

## Testing
* **Test File:** No automated frontend test framework.
* **Verification:** Code-review — on failure paths (network error, HTTP error) the `catch` block retries without touching `sseRef`; the `sseRef` is still cleaned up correctly in `handleStop` and the `useEffect` cleanup. The SSE reconnect logic in `sse.js` is unchanged.
* **Status:** Verified by inspection.

## Additional Notes
There is a theoretical race where the simulation engine emits the very first tick events before the SSE `EventSource` handshake completes (milliseconds after the `/start` response). In practice the tick cadence is throttled via `speed_factor` and the first observable events arrive well after the handshake. If this ever matters, a small server-sent "hello" ping can be added as the first event in `broadcaster.py`.
