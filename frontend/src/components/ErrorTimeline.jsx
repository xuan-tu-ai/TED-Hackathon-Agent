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
    const [thresholds, setThresholds] = useState({ l1: 2.5, l2: 0.015 });
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
                        l1: res.data.l1_threshold ?? 2.5,
                        l2: res.data.l2_threshold ?? 0.015,
                    });
                }
            } catch { /* use defaults */ }
        })();
    }, []);

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
            if (dataCache.current[robotId] && dataCache.current[robotId].length > 0) {
                setChartData(dataCache.current[robotId]);
            } else {
                backfillData(robotId);
            }
        }
    }, [mode, clearErrorCounts]);

    // Hard mode backfill
    const backfillData = useCallback(async (robotId) => {
        if (!onApiFetch) return;
        setLoading(true);
        setError(null);
        try {
            const res = await onApiFetch(`/api/robots/${robotId}/error-history`, {}, 'error-timeline/backfill');
            if (res.ok && res.data) {
                const points = (res.data.data || []).map(d => ({
                    tick: 0,
                    time: d.hour,
                    count: d.count,
                }));
                dataCache.current[robotId] = points;
                // Also merge any SSE data we already have for this robot
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

    // Merge HTTP data + SSE data, deduplicating by time
    const mergeData = (httpPoints, ssePoints) => {
        const seen = new Set();
        const result = [];
        for (const p of [...httpPoints, ...ssePoints]) {
            const key = p.time || p.tick;
            if (!seen.has(key)) {
                seen.add(key);
                result.push(p);
            }
        }
        result.sort((a, b) => {
            if (a.tick !== undefined && b.tick !== undefined) return a.tick - b.tick;
            return String(a.time).localeCompare(String(b.time));
        });
        return result;
    };

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
                <span className="error-timeline-title">📊 Error Timeline</span>
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
                                    const parts = v.split('T');
                                    return parts.length > 1 ? parts[1].slice(0, 5) : v.slice(-8, -3);
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
                            {/* L2 Threshold reference line */}
                            <ReferenceLine
                                y={thresholds.l2}
                                stroke={CHART_COLORS.l2}
                                strokeDasharray="3 6"
                                strokeWidth={1.5}
                                label={{
                                    value: `L2 (${thresholds.l2})`,
                                    position: 'insideTopRight',
                                    fill: CHART_COLORS.l2,
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
