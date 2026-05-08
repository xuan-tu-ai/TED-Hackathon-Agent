const API_BASE = '/api/sim';

export function createSSEConnection(onEvent) {
    let eventSource = null;

    function connect() {
        if (eventSource) eventSource.close();

        eventSource = new EventSource(`${API_BASE}/stream`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onEvent(data);
            } catch (err) {
                console.error('SSE parse error:', err);
            }
        };

        eventSource.onerror = (err) => {
            console.error('SSE Error:', err);
        };
    }

    function disconnect() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    return { connect, disconnect };
}
