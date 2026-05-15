import React from 'react';

export default function ThinkingBubble({ reasoning, isRunning }) {
    return (
        <div className={`bubble-thinking${isRunning ? ' bubble-thinking--running' : ''}`}>
            <div className="bubble-thinking-header">
                <span>{'\u{1F4AD}'} Thinking</span>
                {isRunning && <span className="bubble-thinking-pulse" />}
            </div>
            <div className="bubble-thinking-body">{reasoning}</div>
        </div>
    );
}
