import React from 'react';
import { useSimulationStore } from './stores/useSimulationStore.js';
import { createSSEConnection } from './api/sse.js';
import RobotGrid from './components/RobotGrid.jsx';
import AlertFeed from './components/AlertFeed.jsx';
import IncidentSidebar from './components/IncidentSidebar.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import HITLModal from './components/HITLModal.jsx';
import ErrorTimeline from './components/ErrorTimeline.jsx';

import { API_BASE } from './api/config.js';

function StartScreen({ onStart, speed, onSpeedChange }) {
    return (
        <div className="start-screen">
            <div className="start-screen-content">
                <h1>🔧 Welding Robot Diagnostics</h1>
                <p>Multi-Agent Diagnostic System for Welding Shop Anomaly Detection</p>
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
    const [clientErrors, setClientErrors] = React.useState([]);
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
        reset,
        addErrorCount
    } = useSimulationStore();

    const incidentsList = Object.values(incidents);
    const activeIncident = activeIncidentId ? incidents[activeIncidentId] : null;

    const pushClientError = React.useCallback((context, err, detail) => {
        const message = err instanceof Error ? err.message : String(err);
        const detailText = detail
            ? (typeof detail === 'string' ? detail : JSON.stringify(detail))
            : '';
        const entry = {
            id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
            time: new Date().toLocaleTimeString(),
            context,
            message,
            detail: detailText
        };

        console.error(`[${context}]`, err, detail);

        setClientErrors(prev => [entry, ...prev].slice(0, 6));
    }, []);

    const clearClientErrors = React.useCallback(() => {
        setClientErrors([]);
    }, []);

    const requestJson = React.useCallback(async (url, options, context) => {
        try {
            const response = await fetch(url, options);
            const text = await response.text();
            let data = null;

            if (text) {
                try {
                    data = JSON.parse(text);
                } catch (err) {
                    data = { raw: text };
                }
            }

            if (!response.ok) {
                pushClientError(context, new Error(`HTTP ${response.status}`), text);
                return { ok: false, status: response.status, data };
            }

            return { ok: true, status: response.status, data };
        } catch (err) {
            pushClientError(context, err);
            return { ok: false, status: 0, data: null };
        }
    }, [pushClientError]);

    const requestNoBody = React.useCallback(async (url, options, context) => {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                const text = await response.text();
                pushClientError(context, new Error(`HTTP ${response.status}`), text);
                return false;
            }
            return true;
        } catch (err) {
            pushClientError(context, err);
            return false;
        }
    }, [pushClientError]);

    const errorPanel = clientErrors.length > 0 ? (
        <div className="client-error-panel">
            <div className="client-error-header">
                <span>Client errors ({clientErrors.length})</span>
                <button className="client-error-clear" onClick={clearClientErrors}>Clear</button>
            </div>
            <ul className="client-error-list">
                {clientErrors.map((item) => (
                    <li key={item.id} className="client-error-item">
                        <div className="client-error-message">{item.message}</div>
                        <div className="client-error-meta">{item.time} · {item.context}</div>
                        {item.detail && <div className="client-error-detail">{item.detail}</div>}
                    </li>
                ))}
            </ul>
        </div>
    ) : null;

    const handleStart = async () => {
        for (let attempt = 0; attempt < 10; attempt++) {
            try {
                const result = await requestJson(`${API_BASE}/sim/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ speed_factor: speed })
                }, 'sim/start');
                if (!result.ok) {
                    throw new Error('Server returned ' + result.status);
                }
                console.info('[sim/start] OK', result.data);
                // Connect SSE only after backend confirms it is running, so the
                // EventSource is never created against an unreachable server.
                sseRef.current = createSSEConnection(handleSimEvent, null, pushClientError);
                sseRef.current.connect();
                setTotalTicks(result.data?.total_ticks || 0);
                setIsRunning(true);
                return;
            } catch (err) {
                if (attempt < 9) {
                    await new Promise(r => setTimeout(r, 1000));
                } else {
                    alert('Cannot start simulation:\n\n' + err.message + '\n\nMake sure backend is running on port 8000:\n  uv run uvicorn app.robot_agent.main:app --reload --port 8000');
                }
            }
        }
    };

    const handleStop = async () => {
        await requestNoBody(`${API_BASE}/sim/stop`, { method: 'POST' }, 'sim/stop');
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
                const currentActiveId = useSimulationStore.getState().activeIncidentId;
                if (!currentActiveId) {
                    selectIncident(event.incident_id);
                }
                break;
            case 'incident_queued':
                updateIncidentStatus(event.incident_id, event.status === 'PROCESSING' ? 'PROCESSING' : 'QUEUED', { queuePosition: event.queue_position });
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
            case 'plan_task_update':
                updatePlanTask(event.incident_id, event.task_id, event.status || 'running');
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
                updateIncidentStatus(event.incident_id, event.status === 'failed' ? 'FAILED' : 'COMPLETED');
                break;
            case 'alert_storm_detected':
                addAlert({ robotId: 'SYSTEM', message: `⚠️ ALERT STORM: ${event.alert_count} robots triggered!`, type: 'agent' });
                break;
            case 'error_count':
                addErrorCount(event.robot_id, event.tick_index, event.sim_time, event.count);
                break;
            case 'sim_finished':
                addAlert({ robotId: 'SYSTEM', message: 'Simulation Finished', type: 'info' });
                break;
        }
    }

    const handleResolveHitl = async (queueId, action) => {
        const ok = await requestNoBody(`${API_BASE}/sim/hitl/${queueId}/${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                queue_id: queueId,
                status: action === 'approve' ? 'APPROVED' : 'REJECTED',
                resolved_by: 'Supervisor'
            })
        }, `hitl/${action}`);
        if (ok) {
            removeHitlItem(activeIncidentId, queueId);
            removeGlobalHitl(queueId);
        }
    };

    const handleSkip = async () => {
        await requestNoBody(`${API_BASE}/sim/skip_to_alert`, { method: 'POST' }, 'sim/skip_to_alert');
    };

    const handleSpeedChange = (newSpeed) => {
        setSpeed(newSpeed);
    };

    if (!isRunning) {
        return (
            <div className="app">
                {errorPanel}
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
            {errorPanel}
            <HITLModal onViewIncident={selectIncident} onError={pushClientError} />
            <header className="app-header">
                <h1>🔧 Welding Robot Diagnostics</h1>
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
                    <div className="dashboard-timeline">
                        <ErrorTimeline onApiFetch={requestJson} />
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
