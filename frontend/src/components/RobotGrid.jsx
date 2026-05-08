import React from 'react';

const ROBOT_COUNT = 47;

function RobotCell({ robot }) {
    const getClassName = () => {
        const base = 'robot-cell';
        switch (robot.state) {
            case 'alert-l1': return `${base} alert-l1`;
            case 'alert-l2': return `${base} alert-l2`;
            case 'processing': return `${base} processing`;
            case 'done': return `${base} done`;
            default: return `${base} normal`;
        }
    };

    return (
        <div id={`robot-${robot.id}`} className={getClassName()}>
            <span>{robot.id}</span>
        </div>
    );
}

export default function RobotGrid({ robots }) {
    if (!robots || robots.length === 0) {
        const initRobots = Array.from({ length: ROBOT_COUNT }, (_, i) => {
            const id = `R-${(i + 1).toString().padStart(2, '0')}`;
            return { id, state: 'normal' };
        });
        return (
            <div className="robot-grid">
                {initRobots.map(robot => (
                    <RobotCell key={robot.id} robot={robot} />
                ))}
            </div>
        );
    }

    return (
        <div className="robot-grid">
            {robots.map(robot => (
                <RobotCell key={robot.id} robot={robot} />
            ))}
        </div>
    );
}
