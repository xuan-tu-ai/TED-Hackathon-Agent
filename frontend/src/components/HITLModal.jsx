import React from 'react';
import { useSimulationStore } from '../stores/useSimulationStore.js';
import { API_BASE } from '../api/config.js';

export default function HITLModal({ onViewIncident, onError }) {
    const { globalHitlQueue, selectIncident, removeGlobalHitl, removeHitlItem } = useSimulationStore();
    const [currentIndex, setCurrentIndex] = React.useState(0);

    if (globalHitlQueue.length === 0) return null;

    const current = globalHitlQueue[currentIndex];
    const total = globalHitlQueue.length;

    const handleApprove = async (queueId) => {
        try {
            const response = await fetch(`${API_BASE}/sim/hitl/${queueId}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ queue_id: queueId, status: 'APPROVED', resolved_by: 'Supervisor' })
            });
            if (!response.ok) {
                const text = await response.text();
                onError && onError('hitl/approve', new Error(`HTTP ${response.status}`), text);
                return;
            }
            removeGlobalHitl(queueId);
            removeHitlItem(current.incident_id, queueId);
            if (currentIndex >= total - 1) setCurrentIndex(Math.max(0, currentIndex - 1));
        } catch (err) {
            console.error('Failed to approve HITL:', err);
            onError && onError('hitl/approve', err);
        }
    };

    const handleReject = async (queueId) => {
        try {
            const response = await fetch(`${API_BASE}/sim/hitl/${queueId}/reject`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ queue_id: queueId, status: 'REJECTED', resolved_by: 'Supervisor' })
            });
            if (!response.ok) {
                const text = await response.text();
                onError && onError('hitl/reject', new Error(`HTTP ${response.status}`), text);
                return;
            }
            removeGlobalHitl(queueId);
            removeHitlItem(current.incident_id, queueId);
            if (currentIndex >= total - 1) setCurrentIndex(Math.max(0, currentIndex - 1));
        } catch (err) {
            console.error('Failed to reject HITL:', err);
            onError && onError('hitl/reject', err);
        }
    };

    const handleViewIncident = () => {
        selectIncident(current.incident_id);
        onViewIncident && onViewIncident(current.incident_id);
    };

    const handlePrev = () => setCurrentIndex(i => Math.max(0, i - 1));
    const handleNext = () => setCurrentIndex(i => Math.min(total - 1, i + 1));

    const ctx = current.context || {};

    return (
        <div className="hitl-modal-overlay">
            <div className="hitl-modal-backdrop" />
            <div className="hitl-modal-card">
                <div className="hitl-modal-header">
                    <div className="hitl-modal-pulse">⚠️</div>
                    <div className="hitl-modal-title">Human Approval Required</div>
                    {total > 1 && (
                        <div className="hitl-modal-counter">({currentIndex + 1}/{total})</div>
                    )}
                </div>

                <div className="hitl-modal-body">
                    <div className="hitl-modal-row">
                        <span className="hitl-modal-label">Incident:</span>
                        <span className="hitl-modal-value">{current.incident_id}</span>
                    </div>
                    <div className="hitl-modal-row">
                        <span className="hitl-modal-label">Robot:</span>
                        <span className="hitl-modal-value">{current.robot_code}</span>
                    </div>
                    <div className="hitl-modal-row">
                        <span className="hitl-modal-label">Action:</span>
                        <span className="hitl-modal-value hitl-modal-action">{current.action}</span>
                    </div>
                    <div className="hitl-modal-row">
                        <span className="hitl-modal-label">Severity:</span>
                        <span className={`hitl-modal-value severity-${current.severity?.toLowerCase()}`}>
                            {current.severity}
                        </span>
                    </div>

                    {ctx.step_count !== undefined && (
                        <div className="hitl-modal-row">
                            <span className="hitl-modal-label">Steps Taken:</span>
                            <span className="hitl-modal-value">{ctx.step_count}</span>
                        </div>
                    )}
                    {ctx.skill_in_use && (
                        <div className="hitl-modal-row">
                            <span className="hitl-modal-label">Skill Active:</span>
                            <span className="hitl-modal-value">{ctx.skill_in_use}</span>
                        </div>
                    )}
                    {ctx.incident_summary && (
                        <div className="hitl-modal-row">
                            <span className="hitl-modal-label">Summary:</span>
                            <span className="hitl-modal-value">{ctx.incident_summary}</span>
                        </div>
                    )}

                    <div className="hitl-modal-reasoning">
                        <div className="hitl-modal-label">Reasoning:</div>
                        <div className="hitl-modal-reasoning-text">{current.reasoning}</div>
                    </div>
                </div>

                <div className="hitl-modal-actions">
                    <button
                        className="btn-hitl btn-approve"
                        onClick={() => handleApprove(current.queue_id)}
                    >
                        Approve
                    </button>
                    <button
                        className="btn-hitl btn-reject"
                        onClick={() => handleReject(current.queue_id)}
                    >
                        Reject
                    </button>
                    <button
                        className="btn-hitl btn-view-incident"
                        onClick={handleViewIncident}
                    >
                        View Incident
                    </button>
                </div>

                {total > 1 && (
                    <div className="hitl-modal-nav">
                        <button onClick={handlePrev} disabled={currentIndex === 0}>◀ Prev</button>
                        <button onClick={handleNext} disabled={currentIndex === total - 1}>Next ▶</button>
                    </div>
                )}
            </div>
        </div>
    );
}
