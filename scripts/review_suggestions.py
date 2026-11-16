# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional GitHub suggestion fences for one concrete repair.

GitHub renders a fenced block whose info string is `suggestion` inside a
pull-request review comment, and the pull request author commits it
(https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/commenting-on-a-pull-request
and https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request).
The fence follows CommonMark 0.31.2 section 4.5: the opening fence is longer
than any backtick run in the replacement, and the closing fence is at least
that long. Those docs do not state a line cap. A fence that is not safe is
omitted; the prose finding stays.

The commented range is the finding's exact visible RIGHT range. `original`
must equal those head lines joined with newlines only. A trailing newline in
the blob does not create an extra line. The suggestion replaces that range;
an empty replacement is an empty fence body (a deletion).
"""
from __future__ import annotations

import re

try:
    from .review_followups import MAX_DATA_BYTES
except ImportError:
    from review_followups import MAX_DATA_BYTES


def head_lines(blob):
    if not isinstance(blob, (bytes, bytearray)) or b"\x00" in blob:
        return None
    try:
        text = blob.decode("utf-8")
    except UnicodeError:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _safe_text(text):
    if not isinstance(text, str) or "\x00" in text or "<!--" in text or "agent-review-" in text:
        return False
    return all(ch in "\n\t" or ord(ch) >= 32 and ord(ch) != 127 for ch in text)


def strip_fences(text):
    """Neutralize model-authored suggestion info strings, including nested fences.

    GitHub follows CommonMark fenced blocks. Changing only the info word keeps
    the proposed code visible as prose/code, without an unchecked Apply button.
    This also safely handles nested quote/list containers and longer fences.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"(`{3,}|~{3,})[ \t]*suggestion", r"\1text", text)


def render_fence(replacement):
    """The suggestion block, or None when the replacement cannot be fenced safely."""
    if not _safe_text(replacement):
        return None
    run = max((len(item.group(0)) for item in re.finditer(r"`+", replacement)), default=0)
    fence = "`" * max(3, run + 1)
    # CommonMark keeps the newline that ends the last content line. An empty
    # body has no content line, which is the deletion form.
    body = replacement if replacement.endswith("\n") or replacement == "" else replacement + "\n"
    if f"\n{fence}" in "\n" + body:
        return None
    return f"{fence}suggestion\n{body}{fence}\n"


def widget(finding, lines, visible, *, first_line=1):
    """A fence when original matches one unambiguous visible range, else None."""
    suggestion = finding.get("suggestion") if isinstance(finding, dict) else None
    if not isinstance(suggestion, dict):
        return None
    original, replacement = suggestion.get("original"), suggestion.get("replacement")
    start, end = finding.get("line"), finding.get("end_line")
    if (not isinstance(lines, list) or not _safe_text(original) or not isinstance(replacement, str)
            or type(start) is not int or type(end) is not int or start < first_line
            or end < start or end - first_line >= len(lines)):
        return None
    if not isinstance(visible, set) or any(line not in visible for line in range(start, end + 1)):
        return None
    if "\n".join(lines[start - first_line:end - first_line + 1]) != original:
        return None
    return render_fence(replacement)


def source_lines(repo, head, path, start, end):
    """Read only the proposed range, with the existing comment byte allowance.

    Oversized optional original lines omit the widget, never the finding.
    The stream still traverses preceding lines without retaining them.
    """
    try:
        from . import review_context as context
        from . import review_sections as sections
    except ImportError:
        import review_context as context
        import review_sections as sections
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        return None
    result, size = [], 0
    try:
        for number, (line, overflow) in enumerate(context._lines(
                sections.git_command(repo, 'cat-file', 'blob', head + ':' + path), MAX_DATA_BYTES), 1):
            if number < start:
                continue
            size += len(line)
            if overflow or size > MAX_DATA_BYTES or b'\0' in line:
                return None
            result.append(line.decode('utf-8').removesuffix('\n'))
            if number == end:
                return result
    except (OSError, ValueError, UnicodeError):
        pass
    return None


def attach(comment, fence):
    if not fence:
        return comment
    body = comment["body"] + "\n\n" + fence
    if len(body.encode("utf-8")) > MAX_DATA_BYTES:
        return comment
    return {**comment, "body": body}


def peel(body):
    """Recognize one complete publisher suffix, including replacement blank lines.

    The caller first tries an exact legacy body match. We peel only the precise
    fence grammar emitted by render_fence, never an arbitrary Markdown block.
    """
    if not isinstance(body, str):
        return body
    for match in re.finditer(r"\n\n(`{3,})suggestion\n", body):
        fence = match.group(1)
        suffix = body[match.end():]
        if not suffix.endswith(fence + "\n"):
            continue
        content = suffix[:-len(fence) - 1]
        if content and not content.endswith("\n"):
            continue
        if not _safe_text(content):
            continue
        if any(len(item.group(0)) >= len(fence) for item in re.finditer(r"`+", content)):
            continue
        return body[:match.start()]
    return body
