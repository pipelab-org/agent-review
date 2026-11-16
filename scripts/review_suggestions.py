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


FENCE = re.compile(r"(`{3,})suggestion[^\n]*\n")


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
    """Drop model-authored suggestion fences so only the publisher can add one."""
    if not isinstance(text, str) or "```" not in text:
        return text
    output, index = [], 0
    while index < len(text):
        match = FENCE.search(text, index)
        if not match:
            output.append(text[index:])
            break
        output.append(text[index:match.start()])
        length = len(match.group(1))
        close = re.search(rf"(?m)^`{{{length},}}[ \t]*$", text[match.end():])
        if not close:
            break
        index = match.end() + close.end()
        if index < len(text) and text[index] == "\n":
            index += 1
    return "".join(output)


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


def widget(finding, lines, visible):
    """A fence when original matches one unambiguous visible range, else None."""
    suggestion = finding.get("suggestion") if isinstance(finding, dict) else None
    if not isinstance(suggestion, dict):
        return None
    original, replacement = suggestion.get("original"), suggestion.get("replacement")
    start, end = finding.get("line"), finding.get("end_line")
    if (not isinstance(lines, list) or not _safe_text(original) or not isinstance(replacement, str)
            or type(start) is not int or type(end) is not int or start < 1 or end < start or end > len(lines)):
        return None
    if not isinstance(visible, set) or any(line not in visible for line in range(start, end + 1)):
        return None
    if "\n".join(lines[start - 1:end]) != original:
        return None
    return render_fence(replacement)


def attach(comment, fence):
    if not fence:
        return comment
    body = comment["body"] + "\n\n" + fence
    if len(body.encode("utf-8")) > MAX_DATA_BYTES:
        return comment
    return {**comment, "body": body}


def peel(body):
    """Split a publisher-appended suggestion fence off a comment. Legacy bodies stay whole."""
    if not isinstance(body, str) or "\n\n```" not in body:
        return body
    prefix, _, suffix = body.rpartition("\n\n")
    match = re.fullmatch(r"(`{3,})suggestion\n(.*)\1\n", suffix, re.S)
    if not match:
        return body
    fence, content = match.group(1), match.group(2)
    if content and not content.endswith("\n"):
        return body
    if any(len(item.group(0)) >= len(fence) for item in re.finditer(r"`+", content)):
        return body
    return prefix
