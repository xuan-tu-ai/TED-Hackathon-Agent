import React from 'react';
import ThinkingBubble from './ThinkingBubble.jsx';
import ToolCallChip from './ToolCallChip.jsx';
import InlineHITLCard from './InlineHITLCard.jsx';
import PlanTracker from './PlanTracker.jsx';
import SummaryReport from './SummaryReport.jsx';

function EmptyState() {
    return (
        <div className="empty-state">
            <div className="empty-state-icon">{'📋'}</div>
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
    const traceScrollRef = React.useRef(null);
    const [traceVisible, setTraceVisible] = React.useState(true);

    const intermediateSteps = agentSteps?.filter(s => s.toolName !== 'complete') || [];
    const completeStep = agentSteps?.find(s => s.toolName === 'complete');

    // Auto-scroll trace area when new steps arrive
    React.useEffect(() => {
        if (traceScrollRef.current && traceVisible) {
            traceScrollRef.current.scrollTop = traceScrollRef.current.scrollHeight;
        }
    }, [agentSteps?.length, hitlItems?.length, traceVisible]);

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

            {/* Section 1 — Thinking Trace */}
            <div className={`trace-section ${traceVisible ? 'trace-section--open' : 'trace-section--closed'}`}>
                <div className="trace-section-header">
                    <span className="trace-section-title">💭 Thinking Trace</span>
                    <button
                        className="trace-toggle-btn"
                        onClick={() => setTraceVisible(v => !v)}
                    >
                        {traceVisible ? 'Hide ▲' : 'Show ▼'}
                    </button>
                </div>
                {traceVisible && (
                    <div className="trace-scroll-area" ref={traceScrollRef}>
                        {intermediateSteps.length === 0 && !hitlItems?.length && (
                            <div className="trace-empty">Agent is thinking…</div>
                        )}
                        {intermediateSteps.map((step, idx) => (
                            <div className="chat-step" key={step.stepId || idx}>
                                <ThinkingBubble
                                    reasoning={step.reasoning}
                                    isRunning={step.observation === null}
                                />
                                <ToolCallChip
                                    toolName={step.toolName || (step.action ? step.action.split('(')[0] : 'unknown')}
                                    toolInput={step.toolInput || (step.action && step.action.includes('(') ? step.action.split('(')[1].replace(')', '') : '')}
                                    observation={step.observation}
                                />
                            </div>
                        ))}
                        {hitlItems?.map((hitl, idx) => (
                            <InlineHITLCard
                                key={`hitl-${hitl.queueId || idx}`}
                                hitl={hitl}
                                onResolve={onResolveHitl}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Section 2 — Diagnostic Plan */}
            {activeIncident.plan && (
                <div className="plan-section">
                    <PlanTracker plan={activeIncident.plan} />
                </div>
            )}

            {/* Section 3 — Final Report */}
            {completeStep && (
                <div className="report-section">
                    <SummaryReport report={completeStep.observation} />
                </div>
            )}
        </div>
    );
}
