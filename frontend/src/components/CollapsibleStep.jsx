import React from 'react';

const TOOL_ICONS = {
    load_skill: '📚',
    query_pm_cards: '🔍',
    query_db: '🗄️',
    hitl_check: '🔔',
    complete: '✅',
    verify_surface_defect: '🔬',
    error: '❌',
};

function getToolIcon(toolName) {
    return TOOL_ICONS[toolName] || '🔧';
}

function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

export default function CollapsibleStep({ step, stepNumber }) {
    const { reasoning, action, observation, toolName, toolInput } = step;

    const displayTool = toolName || (action ? action.split('(')[0] : 'unknown');
    const displayInput = toolInput || (action && action.includes('(') ? action.split('(')[1].replace(')', '') : '');
    const icon = getToolIcon(displayTool);

    return (
        <div className="chat-msg">
            <div className="step-header">
                <span className="step-marker">Step {stepNumber}</span>
                <span className="step-action-badge">
                    <span className="step-action-icon">{icon}</span>
                    <span className="step-action-name">{displayTool}</span>
                    {displayInput && (
                        <span className="step-action-input">({truncate(displayInput, 20)})</span>
                    )}
                </span>
                {observation === null ? (
                    <span className="step-status step-status--running">⏳ Running</span>
                ) : observation ? (
                    <span className="step-status step-status--done">✅ Done</span>
                ) : null}
            </div>

            <details className="step-details">
                <summary className="step-reasoning-summary">💭 Thinking</summary>
                <div className="reasoning-box">{reasoning}</div>
            </details>

            <details className="step-details">
                <summary className="step-observation-summary">
                    📊 Result
                    {observation === null && <span className="step-loading">⏳ Running...</span>}
                </summary>
                <div className="observation-content">
                    {observation !== null ? observation : <span className="step-loading">⏳ Running...</span>}
                </div>
            </details>
        </div>
    );
}
