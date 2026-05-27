import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';
import { useSimulationStore } from '../stores/useSimulationStore.js';

const ROBOTS = ['R-00', 'R-01', 'R-02', 'R-03'];

const CHART_COLORS = {
    line: '#3b82f6',
    l1: '#f59e0b',
    l2: '#ef4444',
    grid: '#334155',
    text: '#94a3b8',
};

/**
 * ErrorTimeline — real-time error-count chart with L1/L2 thresholds.
 *
 * Props:
 *   onApiFetch   — async (url) => json response (from App.jsx requestJson wrapper)
 */
export default function ErrorTimeline({ onApiFetch }) {
    const [activeRobot, setActiveRobot] = useState('R-00');
    const [mode, setMode] = useState('easy'); // 'easy' | 'hard'
    const [thresholds, setThresholds] = useState({ l1: 30, l2: 0.005 });
    const [infoOpen, setInfoOpen] = useState(false);
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const dataCache = useRef({}); // { robotId: [{ tick, time, count }] }

    const errorCounts = useSimulationStore(s => s.errorCounts);
    const addErrorCount = useSimulationStore(s => s.addErrorCount);
    const clearErrorCounts = useSimulationStore(s => s.clearErrorCounts);

    // Fetch L1/L2 thresholds on mount (any robot gives same thresholds)
    useEffect(() => {
        if (!onApiFetch) return;
        (async () => {
            try {
                const res = await onApiFetch(`/api/robots/${activeRobot}/error-history`, {}, 'error-timeline/thresholds');
                if (res.ok && res.data) {
                    setThresholds({
                        l1: res.data.l1_threshold ?? 30,
                        l2: res.data.l2_threshold ?? 0.005,
                    });
                }
            } catch { /* use defaults */ }
        })();
    }, []);

    // Normalise any time string → epoch ms for dedup and sort.
    // Handles both "2025-07-01 06:00:00+00:00" (HTTP/Databricks) and
    // "2025-07-01T06:00:00+00:00" (SSE/ISO) as the same key.
    const toEpoch = (p) => {
        if (p.time) {
            const t = new Date(String(p.time).replace(' ', 'T'));
            if (!isNaN(t)) return t.getTime();
        }
        return (p.tick ?? 0) * 1e12; // tick-only fallback, kept distinct from epoch range
    };

    // Merge HTTP data + SSE data, deduplicating by epoch (format-agnostic).
    const mergeData = (httpPoints, ssePoints) => {
        const seen = new Map(); // epoch → point (first-wins; HTTP before SSE)
        for (const p of [...httpPoints, ...ssePoints]) {
            const key = toEpoch(p);
            if (!seen.has(key)) seen.set(key, p);
        }
        const result = [...seen.values()];
        result.sort((a, b) => toEpoch(a) - toEpoch(b));
        return result;
    };

    // Hard mode backfill
    const backfillData = useCallback(async (robotId) => {
        if (!onApiFetch) return;
        setLoading(true);
        setError(null);
        try {
            const res = await onApiFetch(`/api/robots/${robotId}/error-history`, {}, 'error-timeline/backfill');
            if (res.ok && res.data) {
                // Find the latest sim time across all robots from SSE data.
                // This prevents showing DB rows that belong to future simulation ticks.
                let cutoff = null;
                for (const pts of Object.values(errorCounts)) {
                    for (const p of pts) {
                        if (!p.time) continue;
                        const t = new Date(String(p.time).replace(' ', 'T'));
                        if (!isNaN(t) && (!cutoff || t > cutoff)) cutoff = t;
                    }
                }

                let points = (res.data.data || []).map(d => ({
                    time: d.hour,
                    count: d.count,
                }));

                if (cutoff) {
                    points = points.filter(p => {
                        const t = new Date(String(p.time).replace(' ', 'T'));
                        return !isNaN(t) && t <= cutoff;
                    });
                } else {
                    // No SSE data yet — simulation hasn't started; show nothing from DB
                    points = [];
                }

                const ssePoints = errorCounts[robotId] || [];
                const merged = mergeData(points, ssePoints);
                dataCache.current[robotId] = merged;
                setChartData(merged);
            }
        } catch (err) {
            setError('Failed to load history');
        } finally {
            setLoading(false);
        }
    }, [onApiFetch, errorCounts]);

    // Switch robot
    const switchRobot = useCallback((robotId) => {
        setActiveRobot(robotId);

        if (mode === 'easy') {
            // Easy: clear data for this robot, start fresh
            clearErrorCounts(robotId);
            dataCache.current[robotId] = [];
            setChartData([]);
        } else {
            // Hard: try cache first, else HTTP backfill
            if (dataCache.current[robotId]?.length > 0) {
                setChartData(dataCache.current[robotId]);
            } else {
                setChartData([]);
                backfillData(robotId);
            }
        }
    }, [mode, clearErrorCounts, backfillData]);

    // Listen for SSE error_count events
    useEffect(() => {
        const points = errorCounts[activeRobot] || [];
        if (points.length === 0) return;

        if (mode === 'easy') {
            // Easy: just use SSE data directly
            setChartData([...points]);
        } else {
            // Hard: merge with cached HTTP data
            const cached = dataCache.current[activeRobot] || [];
            const merged = mergeData(cached, points);
            dataCache.current[activeRobot] = merged;
            setChartData([...merged]);
        }
    }, [errorCounts, activeRobot, mode]);

    // Easy mode: clear chart when switching (done in switchRobot)

    const toggleMode = () => {
        const newMode = mode === 'easy' ? 'hard' : 'easy';
        setMode(newMode);
        if (newMode === 'easy') {
            // Switch to easy: clear, start fresh
            clearErrorCounts(activeRobot);
            dataCache.current[activeRobot] = [];
            setChartData([]);
        } else {
            // Switch to hard: backfill
            backfillData(activeRobot);
        }
    };

    // Custom tooltip
    const CustomTooltip = ({ active, payload, label }) => {
        if (!active || !payload || !payload.length) return null;
        return (
            <div style={{
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 13,
                color: '#e2e8f0',
            }}>
                <div style={{ marginBottom: 4, color: '#94a3b8' }}>{label}</div>
                {payload.map((p, i) => (
                    <div key={i} style={{ color: p.color }}>
                        {p.name}: {p.value}
                    </div>
                ))}
            </div>
        );
    };

    return (
        <div className="error-timeline">
            <div className="error-timeline-header">
                <div className="error-timeline-title-row">
                    <span className="error-timeline-title">📊 Error Timeline</span>
                    <button
                        className="error-timeline-info-btn"
                        onClick={() => setInfoOpen(v => !v)}
                        title="What are L1 and L2 triggers?"
                    >?</button>
                </div>
                <div className="error-timeline-controls">
                    {/* Robot selector */}
                    <div className="robot-selector">
                        {ROBOTS.map(rid => (
                            <button
                                key={rid}
                                className={`robot-select-btn ${activeRobot === rid ? 'active' : ''}`}
                                onClick={() => switchRobot(rid)}
                            >
                                {rid}
                            </button>
                        ))}
                    </div>
                    {/* Mode toggle */}
                    <div className="mode-toggle">
                        <span className={`mode-label ${mode === 'easy' ? 'active' : ''}`}>Easy</span>
                        <button
                            className={`mode-switch ${mode}`}
                            onClick={toggleMode}
                            title={mode === 'easy' ? 'Switch to Hard (backfill history)' : 'Switch to Easy (start fresh)'}
                        >
                            <div className="mode-switch-knob" />
                        </button>
                        <span className={`mode-label ${mode === 'hard' ? 'active' : ''}`}>Hard</span>
                    </div>
                </div>
            </div>
            {infoOpen && (
                <div className="error-timeline-info-panel">
                    <div className="info-panel-close" onClick={() => setInfoOpen(false)}>✕</div>
                    <p><strong>L1 Burst</strong> — detects a sudden spike in anomaly density within a short rolling window (24 h). Fires when the number of file-anomaly events per hour exceeds <strong>{thresholds.l1}</strong>. Indicates an acute fault requiring immediate attention.</p>
                    <p><strong>L2 Escalation</strong> — detects a slow upward trend in anomaly rate over a longer window (48 h, bucketed by 6 h). Fires when the linear slope of hourly error counts exceeds <strong>{thresholds.l2}</strong> errors/hour². Indicates progressive wear or a developing fault before it becomes critical.</p>
                    <p className="info-panel-note">Both triggers are evaluated independently; either one can open an incident.</p>
                </div>
            )}
            <div className="error-timeline-chart">
                {loading && <div className="error-timeline-loading">Loading history…</div>}
                {error && <div className="error-timeline-error">{error}</div>}
                {!loading && !error && chartData.length === 0 && (
                    <div className="error-timeline-empty">No data yet — simulation starting…</div>
                )}
                {chartData.length > 0 && (
                    <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                            <XAxis
                                dataKey="time"
                                tick={{ fontSize: 11, fill: CHART_COLORS.text }}
                                tickLine={false}
                                axisLine={{ stroke: CHART_COLORS.grid }}
                                tickFormatter={(v) => {
                                    if (!v) return '';
                                    const s = String(v);
                                    // ISO: "2025-07-08T06:00:00+00:00" or space: "2025-07-08 06:00:00"
                                    const sep = s.includes('T') ? 'T' : ' ';
                                    const [datePart, timePart] = s.split(sep);
                                    if (datePart && timePart) {
                                        return `${datePart.slice(5)} ${timePart.slice(0, 5)}`;
                                    }
                                    return s.slice(0, 16);
                                }}
                            />
                            <YAxis
                                tick={{ fontSize: 11, fill: CHART_COLORS.text }}
                                tickLine={false}
                                axisLine={{ stroke: CHART_COLORS.grid }}
                                allowDecimals={false}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                                wrapperStyle={{ fontSize: 12, color: CHART_COLORS.text }}
                            />
                            <Line
                                type="monotone"
                                dataKey="count"
                                name="Errors"
                                stroke={CHART_COLORS.line}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4 }}
                                isAnimationActive={false}
                            />
                            {/* L1 Threshold reference line */}
                            <ReferenceLine
                                y={thresholds.l1}
                                stroke={CHART_COLORS.l1}
                                strokeDasharray="6 4"
                                strokeWidth={1.5}
                                label={{
                                    value: `L1 (${thresholds.l1})`,
                                    position: 'insideTopRight',
                                    fill: CHART_COLORS.l1,
                                    fontSize: 11,
                                }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
}
