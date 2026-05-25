import React from 'react';
import ReactMarkdown from 'react-markdown';

const TASK_ICONS = {
    pending: '⬜',
    running: '🔄',
    completed: '✅',
    waiting_hitl: '🔔',
    failed: '❌',
};

function MarkdownInline({ children }) {
    return (
        <ReactMarkdown
            components={{
                p: ({ children }) => <>{children}</>,
            }}
        >
            {children}
        </ReactMarkdown>
    );
}

function TaskItem({ task, onTaskClick }) {
    const icon = TASK_ICONS[task.status] || '⬜';
    const isRunning = task.status === 'running';
    const isCompleted = task.status === 'completed';
    const isWaitingHitl = task.status === 'waiting_hitl';

    return (
        <div
            className={`plan-task plan-task--${task.status}`}
            onClick={() => onTaskClick && onTaskClick(task)}
        >
            <span className={`plan-task-icon ${isRunning ? 'plan-task-icon--spinning' : ''}`}>
                {icon}
            </span>
            <span className={`plan-task-name ${isCompleted ? 'plan-task-name--completed' : ''}`}>
                <MarkdownInline>{task.name}</MarkdownInline>
            </span>
            {isWaitingHitl && (
                <span className="plan-task-hitl-badge">HITL</span>
            )}
        </div>
    );
}

function EmptyPlanState() {
    return (
        <div className="plan-tracker-empty">
            <div className="plan-tracker-empty-icon">📋</div>
            <div className="plan-tracker-empty-text">No plan yet</div>
        </div>
    );
}

export default function PlanTracker({ plan, onTaskClick }) {
    if (!plan || !plan.tasks || plan.tasks.length === 0) return null;

    const completedCount = plan.tasks.filter(t => t.status === 'completed').length;
    const progress = Math.round((completedCount / plan.tasks.length) * 100);

    return (
        <div className="plan-tracker">
            <div className="plan-tracker-header">
                <span className="plan-tracker-title">Diagnostic Roadmap</span>
                <span className="plan-tracker-count">
                    {completedCount}/{plan.tasks.length} ({progress}%)
                </span>
            </div>
            <div className="plan-tracker-tasks">
                {plan.tasks.map(task => (
                    <TaskItem
                        key={task.id}
                        task={task}
                        onTaskClick={onTaskClick}
                    />
                ))}
            </div>
        </div>
    );
}
