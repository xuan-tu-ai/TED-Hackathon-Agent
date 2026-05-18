import React from 'react';

/**
 * Lightweight markdown-like renderer for the final diagnostic report.
 * Converts a limited subset of markdown to React elements without any
 * external dependency.
 *
 * Supported syntax:
 *  - ## / ### headings
 *  - **bold**
 *  - Single-line code with backticks
 *  - Unordered lists (- / *)
 *  - Ordered lists (1. 2. ...)
 *  - Inline code
 */
function renderLine(line) {
    // Escape HTML entities first
    let html = line
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    return html;
}

function parseMarkdown(text) {
    if (!text) return [];
    const lines = text.split('\n');
    const blocks = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];
        const trimmed = line.trim();

        // Empty line
        if (!trimmed) {
            i++;
            continue;
        }

        // Heading ## or ###
        const headingMatch = trimmed.match(/^(#{2,3})\s+(.+)/);
        if (headingMatch) {
            const level = headingMatch[1].length;
            blocks.push({
                type: 'heading',
                level,
                html: renderLine(headingMatch[2]),
            });
            i++;
            continue;
        }

        // Unordered list
        if (trimmed.match(/^[-*]\s+/)) {
            const items = [];
            while (i < lines.length && lines[i].trim().match(/^[-*]\s+/)) {
                items.push(renderLine(lines[i].trim().replace(/^[-*]\s+/, '')));
                i++;
            }
            blocks.push({ type: 'list', ordered: false, items });
            continue;
        }

        // Ordered list
        if (trimmed.match(/^\d+\.\s+/)) {
            const items = [];
            while (i < lines.length && lines[i].trim().match(/^\d+\.\s+/)) {
                items.push(renderLine(lines[i].trim().replace(/^\d+\.\s+/, '')));
                i++;
            }
            blocks.push({ type: 'list', ordered: true, items });
            continue;
        }

        // Regular paragraph
        const paraLines = [];
        while (i < lines.length && lines[i].trim()) {
            paraLines.push(renderLine(lines[i]));
            i++;
        }
        blocks.push({ type: 'paragraph', html: paraLines.join('<br/>') });
    }

    return blocks;
}

export default function SummaryReport({ report }) {
    if (!report) return null;

    // Strip "FINAL DIAGNOSTIC SUMMARY:" prefix if present
    const cleaned = report.replace(/^FINAL DIAGNOSTIC SUMMARY:\s*\n?/, '');

    const blocks = parseMarkdown(cleaned);

    return (
        <div className="summary-report">
            {blocks.map((block, idx) => {
                switch (block.type) {
                    case 'heading':
                        const Tag = block.level === 2 ? 'h2' : 'h3';
                        return (
                            <Tag
                                key={idx}
                                className={`summary-report-heading summary-report-heading--h${block.level}`}
                                dangerouslySetInnerHTML={{ __html: block.html }}
                            />
                        );
                    case 'list':
                        const ListTag = block.ordered ? 'ol' : 'ul';
                        return (
                            <ListTag key={idx} className="summary-report-list">
                                {block.items.map((item, j) => (
                                    <li
                                        key={j}
                                        className="summary-report-list-item"
                                        dangerouslySetInnerHTML={{ __html: item }}
                                    />
                                ))}
                            </ListTag>
                        );
                    case 'paragraph':
                        return (
                            <p
                                key={idx}
                                className="summary-report-paragraph"
                                dangerouslySetInnerHTML={{ __html: block.html }}
                            />
                        );
                    default:
                        return null;
                }
            })}
        </div>
    );
}
