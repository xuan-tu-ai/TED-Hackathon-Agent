import React from 'react';

const TOOL_ICONS = {
    load_skill: '\u{1F4DA}',
    query_pm_cards: '\u{1F50D}',
    query_db: '\u{1F5C4}\uFE0F',
    hitl_check: '\u{1F514}',
    update_task_status: '\u2705',
    create_plan: '\u{1F4DD}',
    complete: '\u{1F3C1}',
    verify_surface_defect: '\u{1F52C}',
    error: '\u274C',
};

function getToolIcon(toolName) {
    return TOOL_ICONS[toolName] || '\u{1F527}';
}

function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

export default function ToolCallChip({ toolName, toolInput, observation }) {
    const isRunning = observation === null;
    const icon = getToolIcon(toolName);

    return (
        <div className={`chip-tool${isRunning ? ' chip-tool--running' : ''}`}>
            <div className="chip-tool-header">
                <span className="chip-tool-icon">{icon}</span>
                <span className="chip-tool-label">{toolName}</span>
                <span className="chip-tool-args">({truncate(toolInput, 80)})</span>
                <span className="chip-tool-status">
                    {isRunning ? '\u23F3 Running' : '\u2713 Done'}
                </span>
            </div>
            {observation !== null && (
                <details className="chip-tool-details" open={observation.length < 300}>
                    <summary className="chip-tool-summary">Result</summary>
                    <div className="chip-tool-result">{observation}</div>
                </details>
            )}
            {isRunning && (
                <div className="chip-tool-result chip-tool-result--empty">
                    {'\u23F3 Processing\u2026'}
                </div>
            )}
        </div>
    );
}
