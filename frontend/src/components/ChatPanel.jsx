import React from 'react';
import CollapsibleStep from './CollapsibleStep.jsx';
import HITLCard from './HITLCard.jsx';
import PlanTracker from './PlanTracker.jsx';

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
    const chatBottomRef = React.useRef(null);

    // Auto-scroll to bottom when new steps arrive
    React.useEffect(() => {
        if (chatBottomRef.current) {
            chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [agentSteps?.length, hitlItems?.length]);

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
                        key={`hitl-${hitl.queueId || idx}`}
                        hitl={hitl}
                        onResolve={onResolveHitl}
                    />
                ))}
                <div ref={chatBottomRef} />
            </div>
        </div>
    );
}
