import React from 'react';
import ThinkingBubble from './ThinkingBubble.jsx';
import ToolCallChip from './ToolCallChip.jsx';
import InlineHITLCard from './InlineHITLCard.jsx';
import PlanTracker from './PlanTracker.jsx';
import SummaryReport from './SummaryReport.jsx';

function EmptyState() {
    return (
        <div className="empty-state">
            <div className="empty-state-icon">{'\uD83D\uDCCB'}</div>
            <div className="empty-state-text">Select an incident to view agent diagnostics</div>
        </div>
    );
}

function AgentHeader({ incidentId, robotId, type, status }) {
    return (
        <div className="agent-panel-header">
            <div className="agent-incident-id">{incidentId}</div>
            <div className="agent-incident-meta">{robotId} \u00B7 {type} \u00B7 {status}</div>
        </div>
    );
}

export default function ChatPanel({ activeIncident, agentSteps, hitlItems, onResolveHitl }) {
    const chatBottomRef = React.useRef(null);
    const [traceVisible, setTraceVisible] = React.useState(true);
    const hasReport = agentSteps?.some(s => s.toolName === 'complete');

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
                {agentSteps.map((step, idx) => {
                    const isRunning = step.observation === null;
                    const isComplete = step.toolName === 'complete';
                    return (
                        <div className="chat-step" key={step.stepId || idx}>
                            {/* Thinking trace — collapsible when report is present */}
                            {(!isComplete || traceVisible) && (
                                <>
                                    <ThinkingBubble
                                        reasoning={step.reasoning}
                                        isRunning={isRunning}
                                    />
                                    {!isComplete && (
                                        <ToolCallChip
                                            toolName={step.toolName || (step.action ? step.action.split('(')[0] : 'unknown')}
                                            toolInput={step.toolInput || (step.action && step.action.includes('(') ? step.action.split('(')[1].replace(')', '') : '')}
                                            observation={step.observation}
                                        />
                                    )}
                                </>
                            )}
                            {isComplete && (
                                <SummaryReport
                                    report={step.observation}
                                    showToggle={hasReport}
                                    traceVisible={traceVisible}
                                    onToggleTrace={() => setTraceVisible(v => !v)}
                                />
                            )}
                        </div>
                    );
                })}
                {hitlItems?.map((hitl, idx) => (
                    <InlineHITLCard
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
