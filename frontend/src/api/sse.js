const API_BASE = '/api/sim';

export function createSSEConnection(onEvent, onConnectionChange) {
    let eventSource = null;
    let reconnectAttempts = 0;
    let reconnectTimeout = null;
    let isIntentionallyDisconnected = false;

    const MAX_RECONNECT_DELAY = 30000;
    const INITIAL_RECONNECT_DELAY = 1000;

    function getReconnectDelay() {
        const delay = Math.min(
            INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttempts),
            MAX_RECONNECT_DELAY
        );
        return delay;
    }

    function setConnectionStatus(status) {
        if (onConnectionChange) {
            onConnectionChange(status);
        }
    }

    function connect() {
        if (eventSource) eventSource.close();

        isIntentionallyDisconnected = false;
        setConnectionStatus('connecting');

        eventSource = new EventSource(`${API_BASE}/stream`);

        eventSource.onopen = () => {
            reconnectAttempts = 0;
            setConnectionStatus('connected');
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onEvent(data);
            } catch (err) {
                console.error('SSE parse error:', err);
            }
        };

        eventSource.onerror = (err) => {
            if (isIntentionallyDisconnected) return;

            setConnectionStatus('disconnected');
            console.error('SSE Error:', err);

            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
            }

            const delay = getReconnectDelay();
            reconnectAttempts++;

            console.warn(`SSE connection lost. Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);

            reconnectTimeout = setTimeout(() => {
                if (!isIntentionallyDisconnected) {
                    connect();
                }
            }, delay);
        };
    }

    function disconnect() {
        isIntentionallyDisconnected = true;

        if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
        }

        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        setConnectionStatus('disconnected');
    }

    return { connect, disconnect };
}
