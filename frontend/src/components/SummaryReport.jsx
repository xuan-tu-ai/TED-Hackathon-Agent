import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Final diagnostic report rendered with react-markdown + remark-gfm.
 * Supports full GFM including tables, lists, code blocks, and inline formatting.
 */
function ReportContent({ content }) {
    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
                h2: ({ children, ...props }) => (
                    <h2 className="summary-report-heading summary-report-heading--h2" {...props}>{children}</h2>
                ),
                h3: ({ children, ...props }) => (
                    <h3 className="summary-report-heading summary-report-heading--h3" {...props}>{children}</h3>
                ),
                p: ({ children, ...props }) => (
                    <p className="summary-report-paragraph" {...props}>{children}</p>
                ),
                ul: ({ children, ...props }) => (
                    <ul className="summary-report-list" {...props}>{children}</ul>
                ),
                ol: ({ children, ...props }) => (
                    <ol className="summary-report-list" {...props}>{children}</ol>
                ),
                li: ({ children, ...props }) => (
                    <li className="summary-report-list-item" {...props}>{children}</li>
                ),
                table: ({ children, ...props }) => (
                    <div className="summary-report-table-wrapper">
                        <table className="summary-report-table" {...props}>{children}</table>
                    </div>
                ),
                th: ({ children, ...props }) => (
                    <th className="summary-report-th" {...props}>{children}</th>
                ),
                td: ({ children, ...props }) => (
                    <td className="summary-report-td" {...props}>{children}</td>
                ),
                code: ({ children, ...props }) => {
                    const isInline = !props.className;
                    if (isInline) {
                        return <code className="summary-report-code" {...props}>{children}</code>;
                    }
                    return (
                        <pre className="summary-report-pre">
                            <code {...props}>{children}</code>
                        </pre>
                    );
                },
                strong: ({ children, ...props }) => <strong {...props}>{children}</strong>,
                em: ({ children, ...props }) => <em {...props}>{children}</em>,
            }}
        >
            {content}
        </ReactMarkdown>
    );
}

function MsTeamsButton() {
    const [clicked, setClicked] = React.useState(false);
    const handleClick = () => {
        setClicked(true);
        setTimeout(() => setClicked(false), 2000);
    };
    return (
        <button
            className={`btn-ms-teams ${clicked ? 'btn-ms-teams--clicked' : ''}`}
            onClick={handleClick}
            disabled={clicked}
            title="Send this diagnostic report to Microsoft Teams (placeholder)"
        >
            {clicked ? '✓ Sent!' : '📤 Send to MS Teams'}
        </button>
    );
}

export default function SummaryReport({ report }) {
    if (!report) return null;

    // Strip "FINAL DIAGNOSTIC SUMMARY:" prefix if present
    const cleaned = report.replace(/^FINAL DIAGNOSTIC SUMMARY:\s*\n?/, '');

    return (
        <div className="summary-report">
            <ReportContent content={cleaned} />
            <div className="summary-report-footer">
                <MsTeamsButton />
            </div>
        </div>
    );
}
