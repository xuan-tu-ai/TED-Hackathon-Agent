import React from 'react';

const TOOL_ICONS = {
    load_skill: '📚',
    query_pm_cards: '🔍',
    query_db: '🗄️',
    hitl_check: '🔔',
    update_task_status: '✅',
    create_plan: '📝',
    complete: '🏁',
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
                {observation === null ? (
                    <span className="step-status step-status--running">⏳ Running</span>
                ) : (
                    <span className="step-status step-status--done">Done</span>
                )}
            </div>

            <div className="step-thought-container">
                <div className="thought-header">
                    <span>💭 Thought</span>
                </div>
                <div className="thought-content">{reasoning}</div>
            </div>

            <div className="tool-call-container">
                <div className="tool-call-block">
                    <span className="tool-icon">{icon}</span>
                    <span className="tool-name">{displayTool}</span>
                    <span className="tool-args">({truncate(displayInput, 100)})</span>
                </div>
            </div>

            <div className="result-container">
                <details className="result-details" open={observation !== null && observation.length < 200}>
                    <summary className="result-summary">
                        <span>Result</span>
                    </summary>
                    <div className="result-content">
                        {observation !== null ? observation : <span className="step-loading">⏳ Processing...</span>}
                    </div>
                </details>
            </div>
        </div>
    );
}
