import React from 'react';

export default function AlertFeed({ alerts }) {
    if (!alerts || alerts.length === 0) {
        return (
            <div className="alert-feed">
                <div className="alert-feed-header">
                    <span>Alert Feed</span>
                    <span className="alert-count">0 alerts</span>
                </div>
                <div className="alert-feed-empty">No alerts yet</div>
            </div>
        );
    }

    const formatTime = (date) => {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <div className="alert-feed">
            <div className="alert-feed-header">
                <span>Alert Feed</span>
                <span className="alert-count">{alerts.length} alerts</span>
            </div>
            <div className="alert-feed-items">
                {[...alerts].reverse().map((alert, idx) => (
                    <div key={idx} className={`alert-item ${alert.type || 'info'}`}>
                        <div className="alert-time">{formatTime(new Date(alert.time))}</div>
                        <div className="alert-msg">
                            <strong>{alert.robotId}</strong>: {alert.message}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
