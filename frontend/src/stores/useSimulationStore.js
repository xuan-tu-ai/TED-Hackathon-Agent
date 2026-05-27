import { create } from 'zustand';

const ROBOT_COUNT = 4;

const createInitialRobots = () =>
    Array.from({ length: ROBOT_COUNT }, (_, i) => {
        const id = `R-${i.toString().padStart(2, '0')}`;
        return { id, state: 'normal' };
    });

export const useSimulationStore = create((set, get) => ({
    robots: createInitialRobots(),
    incidents: {},
    activeIncidentId: null,
    tick: 0,
    simTime: null,
    speed: 1,
    alerts: [],
    hitlItems: [],
    globalHitlQueue: [],
    errorCounts: {}, // { robotId: [{ tick, time, count }] },

    updateRobot: (robotId, newState) =>
        set(state => ({
            robots: state.robots.map(r =>
                r.id === robotId ? { ...r, state: newState } : r
            )
        })),

    addIncident: (incident) =>
        set(state => ({
            incidents: {
                ...state.incidents,
                [incident.incidentId]: {
                    incidentId: incident.incidentId,
                    robotId: incident.robotId,
                    type: incident.type,
                    status: incident.status || 'ALERT',
                    steps: [],
                    hitlItems: [],
                    queuePosition: null,
                    plan: null,
                    createdAt: new Date().toISOString()
                }
            }
        })),

    setPlan: (incidentId, tasks) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...state.incidents[incidentId],
                        plan: { tasks }
                    }
                }
            };
        }),

    updatePlanTask: (incidentId, taskId, status) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            const incident = state.incidents[incidentId];
            if (!incident.plan) return state;
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...incident,
                        plan: {
                            tasks: incident.plan.tasks.map(t =>
                                String(t.id) === String(taskId) ? { ...t, status } : t
                            )
                        }
                    }
                }
            };
        }),

    addGlobalHitl: (hitlItem) =>
        set(state => ({
            globalHitlQueue: [...state.globalHitlQueue, hitlItem]
        })),

    removeGlobalHitl: (queueId) =>
        set(state => ({
            globalHitlQueue: state.globalHitlQueue.filter(h => h.queue_id !== queueId)
        })),

    updateIncidentStatus: (incidentId, status, extra = {}) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...state.incidents[incidentId],
                        status,
                        ...extra
                    }
                }
            };
        }),

    addAgentStep: (incidentId, step) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...state.incidents[incidentId],
                        steps: [...(state.incidents[incidentId].steps || []), step]
                    }
                }
            };
        }),

    updateAgentStepObservation: (incidentId, stepId, observation) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            const steps = state.incidents[incidentId].steps || [];
            const stepIndex = steps.findIndex(s => s.stepId === stepId);
            if (stepIndex === -1) return state;
            const updatedSteps = [...steps];
            updatedSteps[stepIndex] = { ...updatedSteps[stepIndex], observation };
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...state.incidents[incidentId],
                        steps: updatedSteps
                    }
                }
            };
        }),

    addHitlItem: (incidentId, hitl) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...state.incidents[incidentId],
                        hitlItems: [...(state.incidents[incidentId].hitlItems || []), hitl]
                    }
                }
            };
        }),

    removeHitlItem: (incidentId, queueId) =>
        set(state => {
            if (!state.incidents[incidentId]) return state;
            return {
                incidents: {
                    ...state.incidents,
                    [incidentId]: {
                        ...state.incidents[incidentId],
                        hitlItems: state.incidents[incidentId].hitlItems.filter(
                            h => h.queueId !== queueId
                        )
                    }
                }
            };
        }),

    addErrorCount: (robotId, tick, time, count) =>
        set(state => {
            const existing = state.errorCounts[robotId] || [];
            return {
                simTime: time,
                errorCounts: {
                    ...state.errorCounts,
                    [robotId]: [...existing, { tick, time, count }]
                }
            };
        }),

    clearErrorCounts: (robotId) =>
        set(state => ({
            errorCounts: {
                ...state.errorCounts,
                [robotId]: []
            }
        })),

    selectIncident: (incidentId) => set({ activeIncidentId: incidentId }),

    addAlert: (alert) =>
        set(state => ({
            alerts: [...state.alerts, { ...alert, time: new Date().toISOString() }]
        })),

    incrementTick: () => set(state => ({ tick: state.tick + 1 })),

    setTick: (tick) => set({ tick }),

    setSpeed: (speed) => set({ speed }),

    setRobots: (robots) => set({ robots }),

    reset: () =>
        set({
            robots: createInitialRobots(),
            incidents: {},
            activeIncidentId: null,
            tick: 0,
            simTime: null,
            speed: 1,
            alerts: [],
            hitlItems: [],
            globalHitlQueue: [],
            errorCounts: {}
        })
}));
