import React from 'react';

export default function HITLCard({ hitl, onResolve }) {
    const { queueId, action, severity, reasoning } = hitl;

    const handleApprove = () => onResolve(queueId, 'approve');
    const handleReject = () => onResolve(queueId, 'reject');

    return (
        <div className="hitl-card">
            <div className="hitl-header">⚠️ Human Approval Required</div>
            <div className="hitl-details">
                <div className="hitl-field">
                    <span className="hitl-label">Action:</span>
                    <span className="hitl-value">{action}</span>
                </div>
                <div className="hitl-field">
                    <span className="hitl-label">Severity:</span>
                    <span className={`hitl-value severity-${severity?.toLowerCase()}`}>{severity}</span>
                </div>
                <p className="hitl-reasoning">{reasoning}</p>
            </div>
            <div className="hitl-actions">
                <button className="btn-hitl btn-approve" onClick={handleApprove}>
                    Approve
                </button>
                <button className="btn-hitl btn-reject" onClick={handleReject}>
                    Reject
                </button>
            </div>
        </div>
    );
}
