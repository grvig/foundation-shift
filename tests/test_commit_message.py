"""Tests for the commit-message gate."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_commit_message import check


def test_a_good_message_passes():
    assert check("Added the embedding cache") == []


def test_trailing_newline_and_git_comments_are_fine():
    assert check("Added the harness\n\n# Please enter the commit message\n") == []


def test_an_empty_message_is_rejected():
    assert len(check("")) == 1
    assert len(check("\n# only comments\n")) == 1


def test_a_body_is_rejected():
    assert len(check("Added the harness\n\nIt does a thing worth explaining.\n")) > 0


def test_an_over_long_subject_is_rejected():
    assert len(check("Added " + "x" * 100)) > 0


def test_attribution_trailers_are_rejected_by_shape():
    assert len(check("Added it\n\nCo-Authored-By: Someone <a@b.c>\n")) > 0
    assert len(check("Added it\n\nReviewed-By: Someone\n")) > 0
    assert len(check("Added it\n\nSigned-off-by: Someone\n")) > 0
    assert len(check("Added it\n\nCO-AUTHORED-BY: x\n")) > 0


def test_credit_lines_are_rejected():
    assert len(check("Added it\n\nGenerated with a tool\n")) > 0
    assert len(check("Added it\n\nAssisted by a tool\n")) > 0


def test_every_commit_in_this_repository_would_pass():
    root = Path(__file__).resolve().parents[1]
    output = subprocess.run(
        ["git", "log", "--format=%B%x00"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for message in output.stdout.split("\x00"):
        if message.strip() == "":
            continue
        assert check(message) == [], message
