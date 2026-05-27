import React from 'react';

const STATUS_LABELS = {
    ALERT: 'Alert',
    QUEUED: 'Queued',
    PROCESSING: 'Processing',
    HITL_WAIT: 'HITL',
    AUTO_RESOLVED: 'Auto',
    COMPLETED: 'Done'
};

const STATUS_ORDER = ['ALERT', 'QUEUED', 'PROCESSING', 'HITL_WAIT', 'AUTO_RESOLVED', 'COMPLETED'];

export default function IncidentSidebar({ incidents, activeIncidentId, onSelectIncident }) {
    const sortedIncidents = [...incidents].sort((a, b) => {
        const orderA = STATUS_ORDER.indexOf(a.status);
        const orderB = STATUS_ORDER.indexOf(b.status);
        return orderA - orderB;
    });

    if (sortedIncidents.length === 0) {
        return (
            <div className="incident-sidebar">
                <div className="incident-sidebar-header">
                    <span>Incidents</span>
                    <span className="incident-count">0</span>
                </div>
                <div className="incident-sidebar-empty">No active incidents</div>
            </div>
        );
    }

    return (
        <div className="incident-sidebar">
            <div className="incident-sidebar-header">
                <span>Incidents</span>
                <span className="incident-count">{sortedIncidents.length}</span>
            </div>
            <div className="incident-list">
                {sortedIncidents.map(incident => (
                    <div
                        key={incident.incidentId}
                        id={`incident-${incident.incidentId}`}
                        className={`incident-item ${incident.status.toLowerCase()} ${incident.incidentId === activeIncidentId ? 'active' : ''}`}
                        onClick={() => onSelectIncident(incident.incidentId)}
                    >
                        <div className="incident-item-header">
                            <span className="incident-item-id">{incident.incidentId}</span>
                            <span className={`incident-item-status ${incident.status.toLowerCase()}`}>
                                {STATUS_LABELS[incident.status] || incident.status}
                            </span>
                        </div>
                        <div className="incident-item-meta">
                            {incident.robotId} · {incident.type || 'UNKNOWN'}
                        </div>
                        {incident.queuePosition && (
                            <span className="queue-position">#{incident.queuePosition}</span>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
