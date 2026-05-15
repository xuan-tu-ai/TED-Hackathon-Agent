import React from 'react';

const SEVERITY_CLASSES = {
    LOW: 'severity--low',
    MEDIUM: 'severity--medium',
    HIGH: 'severity--high',
    CRITICAL: 'severity--critical',
};

export default function InlineHITLCard({ hitl, onResolve }) {
    const { queueId, action, severity, reasoning } = hitl;
    const severityClass = SEVERITY_CLASSES[severity] || 'severity--medium';

    const handleApprove = () => onResolve(queueId, 'approve');
    const handleReject = () => onResolve(queueId, 'reject');

    return (
        <div className="hitl-inline">
            <div className="hitl-inline-header">
                {'\u26A0\uFE0F'} Human Review Required
            </div>
            <div className="hitl-inline-body">
                <div className="hitl-inline-row">
                    <span className="hitl-inline-label">Action</span>
                    <code className="hitl-inline-action">{action}</code>
                </div>
                <div className="hitl-inline-row">
                    <span className="hitl-inline-label">Severity</span>
                    <span className={`hitl-inline-severity ${severityClass}`}>{severity}</span>
                </div>
                <div className="hitl-inline-reasoning">{reasoning}</div>
            </div>
            <div className="hitl-inline-actions">
                <button className="hitl-inline-btn hitl-inline-btn--approve" onClick={handleApprove}>
                    {'\u2713'} Approve
                </button>
                <button className="hitl-inline-btn hitl-inline-btn--reject" onClick={handleReject}>
                    {'\u2717'} Reject
                </button>
            </div>
        </div>
    );
}
