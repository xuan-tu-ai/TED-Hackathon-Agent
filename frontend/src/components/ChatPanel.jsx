import React from 'react';
import HITLCard from './HITLCard.jsx';
import PlanTracker from './PlanTracker.jsx';

function CollapsibleStep({ step, stepNumber }) {
    const { reasoning, action, observation } = step;

    return (
        <div className="chat-msg">
            <div className="step-marker">Step {stepNumber}</div>
            <div className="reasoning-box">{reasoning}</div>
            {action !== 'complete' ? (
                <div className="action-call">
                    <span>🔧</span> {action}
                </div>
            ) : null}
            {observation && (
                <details className="observation-details">
                    <summary className="observation-summary">
                        📊 Observation: {observation.length > 50 ? observation.substring(0, 50) + '...' : observation}
                    </summary>
                    <div className="observation-content">{observation}</div>
                </details>
            )}
        </div>
    );
}

function EmptyState() {
    return (
        <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div className="empty-state-text">Select an incident to view agent diagnostics</div>
        </div>
    );
}

function AgentHeader({ incidentId, robotId, type, status }) {
    return (
        <div className="agent-panel-header">
            <div className="agent-incident-id">{incidentId}</div>
            <div className="agent-incident-meta">{robotId} · {type} · {status}</div>
        </div>
    );
}

export default function ChatPanel({ activeIncident, agentSteps, hitlItems, onResolveHitl }) {
    if (!activeIncident) {
        return (
            <div className="chat-panel">
                <EmptyState />
            </div>
        );
    }

    return (
        <div className="chat-panel">
            <AgentHeader
                incidentId={activeIncident.incidentId}
                robotId={activeIncident.robotId}
                type={activeIncident.type || 'UNKNOWN'}
                status={activeIncident.status}
            />
            {activeIncident.plan && (
                <PlanTracker plan={activeIncident.plan} />
            )}
            <div className="chat-container">
                {agentSteps.map((step, idx) => (
                    <CollapsibleStep
                        key={step.stepId || idx}
                        step={step}
                        stepNumber={idx + 1}
                    />
                ))}
                {hitlItems?.map((hitl, idx) => (
                    <HITLCard
                        key={`hitl-${idx}`}
                        hitl={hitl}
                        onResolve={onResolveHitl}
                    />
                ))}
            </div>
        </div>
    );
}
