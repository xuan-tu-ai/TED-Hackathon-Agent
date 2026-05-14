import React from 'react';
import { useSimulationStore } from './stores/useSimulationStore.js';
import { createSSEConnection } from './api/sse.js';
import RobotGrid from './components/RobotGrid.jsx';
import AlertFeed from './components/AlertFeed.jsx';
import IncidentSidebar from './components/IncidentSidebar.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import HITLModal from './components/HITLModal.jsx';

const API_BASE = '/api/sim';

function StartScreen({ onStart, speed, onSpeedChange }) {
    return (
        <div className="start-screen">
            <div className="start-screen-content">
                <h1>🎨 Painting Robot Diagnostics</h1>
                <p>Multi-Agent Diagnostic System for Paint Shop Anomaly Detection</p>
                <div className="start-screen-actions">
                    <button className="btn-start" onClick={onStart}>
                        ▶ Start Simulation
                    </button>
                </div>
                <div className="speed-selector">
                    <span>Speed:</span>
                    <button
                        className={`speed-btn ${speed === 1 ? 'active' : ''}`}
                        onClick={() => onSpeedChange(1)}
                    >
                        1h=1s
                    </button>
                    <button
                        className={`speed-btn ${speed === 2 ? 'active' : ''}`}
                        onClick={() => onSpeedChange(2)}
                    >
                        2x
                    </button>
                    <button
                        className={`speed-btn ${speed === 5 ? 'active' : ''}`}
                        onClick={() => onSpeedChange(5)}
                    >
                        5x
                    </button>
                </div>
            </div>
        </div>
    );
}

function DashboardHeader({ tick, totalTicks, onStop, onSkip }) {
    const progress = totalTicks > 0 ? (tick / totalTicks) * 100 : 0;
    return (
        <div className="dashboard-header">
            <div className="sim-time">
                <span className="sim-label">Simulation</span>
                <span className="sim-tick">Tick: {tick}</span>
            </div>
            <div className="sim-progress">
                <div className="sim-progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
            <div className="dashboard-actions">
                <button className="btn-skip" onClick={onSkip} title="Fast-forward to first alert">⏭ Skip to Alert</button>
                <button className="btn-stop" onClick={onStop}>⏹ Stop</button>
            </div>
        </div>
    );
}

function App() {
    const [isRunning, setIsRunning] = React.useState(false);
    const [totalTicks, setTotalTicks] = React.useState(0);
    const sseRef = React.useRef(null);

    const {
        robots,
        incidents,
        activeIncidentId,
        alerts,
        tick,
        updateRobot,
        addIncident,
        updateIncidentStatus,
        addAgentStep,
        updateAgentStepObservation,
        addHitlItem,
        removeHitlItem,
        selectIncident,
        addAlert,
        setTick,
        setPlan,
        updatePlanTask,
        addGlobalHitl,
        removeGlobalHitl,
        globalHitlQueue,
        speed,
        setSpeed,
        reset
    } = useSimulationStore();

    const incidentsList = Object.values(incidents);
    const activeIncident = activeIncidentId ? incidents[activeIncidentId] : null;

    const handleStart = async () => {
        try {
            const response = await fetch(`${API_BASE}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speed_factor: speed })
            });
            if (!response.ok) throw new Error('Failed to start simulation');
            const data = await response.json();
            setTotalTicks(data.total_ticks || 0);
            setIsRunning(true);

            sseRef.current = createSSEConnection(handleSimEvent);
            sseRef.current.connect();
        } catch (err) {
            console.error('Failed to start simulation:', err);
            alert('Failed to start simulation: ' + err.message);
        }
    };

    const handleStop = async () => {
        try {
            await fetch(`${API_BASE}/stop`, { method: 'POST' });
        } catch (err) {
            console.error('Failed to stop simulation:', err);
        }
        if (sseRef.current) {
            sseRef.current.disconnect();
            sseRef.current = null;
        }
        setIsRunning(false);
        reset();
    };

    React.useEffect(() => {
        return () => {
            if (sseRef.current) {
                sseRef.current.disconnect();
            }
        };
    }, []);

    function handleSimEvent(event) {
        switch (event.type) {
            case 'tick':
                setTick(event.tick_index);
                break;
            case 'alert':
                addAlert({ robotId: event.robot_id, message: `${event.alert_type} detected on ${event.robot_id}`, type: event.alert_type.startsWith('L1') ? 'l1' : 'l2' });
                updateRobot(event.robot_id, event.alert_type.startsWith('L1') ? 'alert-l1' : 'alert-l2');
                addIncident({
                    incidentId: event.incident_id,
                    robotId: event.robot_id,
                    type: event.alert_type,
                    status: 'ALERT'
                });
                // Use getState() to get the most recent value and avoid stale closure issues
                const currentActiveId = useSimulationStore.getState().activeIncidentId;
                if (!currentActiveId) {
                    selectIncident(event.incident_id);
                }
                break;
            case 'incident_queued':
                updateIncidentStatus(event.incident_id, 'QUEUED', { queuePosition: event.queue_position });
                break;
            case 'auto_resolved':
                updateIncidentStatus(event.incident_id, 'AUTO_RESOLVED');
                break;
            case 'agent_step':
                addAgentStep(event.incident_id, {
                    stepId: event.step_id,
                    reasoning: event.reasoning,
                    action: event.action,
                    toolName: event.tool_name,
                    toolInput: event.tool_input,
                    observation: event.observation
                });
                updateIncidentStatus(event.incident_id, 'PROCESSING');
                break;
            case 'agent_step_update':
                updateAgentStepObservation(event.incident_id, event.step_id, event.observation);
                break;
            case 'plan_created':
                setPlan(event.incident_id, event.tasks);
                break;
            case 'plan_task_started':
                updatePlanTask(event.incident_id, event.task_id, 'running');
                break;
            case 'plan_task_completed':
                updatePlanTask(event.incident_id, event.task_id, 'completed');
                break;
            case 'hitl_request':
                addHitlItem(event.incident_id, {
                    queueId: event.queue_id,
                    action: event.action,
                    severity: event.severity,
                    reasoning: event.reasoning,
                    context: event.context
                });
                addGlobalHitl(event);
                updateIncidentStatus(event.incident_id, 'HITL_WAIT');
                break;
            case 'agent_done':
                updateIncidentStatus(event.incident_id, 'COMPLETED');
                break;
            case 'alert_storm_detected':
                addAlert({ robotId: 'SYSTEM', message: `⚠️ ALERT STORM: ${event.alert_count} robots triggered!`, type: 'agent' });
                break;
            case 'sim_finished':
                addAlert({ robotId: 'SYSTEM', message: 'Simulation Finished', type: 'info' });
                break;
        }
    }

    const handleResolveHitl = async (queueId, action) => {
        try {
            await fetch(`${API_BASE}/hitl/${queueId}/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    queue_id: queueId,
                    status: action === 'approve' ? 'APPROVED' : 'REJECTED',
                    resolved_by: 'Supervisor'
                })
            });
            removeHitlItem(activeIncidentId, queueId);
            removeGlobalHitl(queueId);
        } catch (err) {
            console.error('Failed to resolve HITL:', err);
        }
    };

    const handleSkip = async () => {
        try {
            await fetch(`${API_BASE}/skip_to_alert`, { method: 'POST' });
        } catch (err) {
            console.error('Failed to skip to alert:', err);
        }
    };

    const handleSpeedChange = (newSpeed) => {
        setSpeed(newSpeed);
    };

    if (!isRunning) {
        return (
            <div className="app">
                <StartScreen
                    onStart={handleStart}
                    speed={speed}
                    onSpeedChange={handleSpeedChange}
                />
            </div>
        );
    }

    return (
        <div className="app">
            <HITLModal onViewIncident={selectIncident} />
            <header className="app-header">
                <h1>🎨 Painting Robot Diagnostics</h1>
            </header>
            <main className="app-main">
                <DashboardHeader
                    tick={tick}
                    totalTicks={totalTicks}
                    onStop={handleStop}
                    onSkip={handleSkip}
                />
                <div className="dashboard">
                    <div className="dashboard-top">
                        <div className="robot-grid-container">
                            <RobotGrid robots={robots} />
                        </div>
                        <div className="alert-feed-container">
                            <AlertFeed alerts={alerts} />
                        </div>
                    </div>
                    <div className="dashboard-bottom">
                        <IncidentSidebar
                            incidents={incidentsList}
                            activeIncidentId={activeIncidentId}
                            onSelectIncident={selectIncident}
                        />
                        <ChatPanel
                            key={activeIncidentId || 'no-incident'}
                            activeIncident={activeIncident}
                            agentSteps={activeIncident?.steps || []}
                            hitlItems={activeIncident?.hitlItems || []}
                            onResolveHitl={handleResolveHitl}
                        />
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;
