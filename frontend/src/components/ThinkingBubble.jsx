import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ThinkingBubble({ reasoning, isRunning }) {
    return (
        <div className={`bubble-thinking${isRunning ? ' bubble-thinking--running' : ''}`}>
            <div className="bubble-thinking-header">
                <span>{'\u{1F4AD}'} Thinking</span>
                {isRunning && <span className="bubble-thinking-pulse" />}
            </div>
            <div className="bubble-thinking-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {reasoning || ''}
                </ReactMarkdown>
            </div>
        </div>
    );
}
