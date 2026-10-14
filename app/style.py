"""The app's look: a printed results page rather than a dashboard.

Paper-coloured background, IBM Plex type (serif headings, sans body, mono numbers),
tables ruled above and below like a journal table, and numbered figures with captions.
The accent is a haematoxylin purple, the dye that colours cell nuclei in these slides.
Every page calls these helpers instead of Streamlit's default widgets for text, tables
and figures, so the style lives in one place.
"""

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
  --ink: #1c1a17;
  --muted: #6b665e;
  --rule: #d6d0c4;
  --paper: #f7f5f0;
  --accent: #5b3a8a;
  --right: #2f6b3a;
  --wrong: #a23b2a;
}
html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, li, label, input, select {
  font-family: 'IBM Plex Sans', sans-serif;
}
.block-container { max-width: 1000px; padding-top: 2.2rem; padding-bottom: 4rem; }
[data-testid="stAppDeployButton"], #MainMenu, footer { display: none; }
h1, h2, h3 { font-family: 'IBM Plex Serif', serif !important; color: var(--ink);
  letter-spacing: -0.01em; }
h1 { font-weight: 600 !important; font-size: 2.1rem !important; margin-bottom: 0.2rem; }

.kicker { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem; }
.lead { font-size: 1.08rem; line-height: 1.6; color: var(--ink); max-width: 46rem;
  margin: 0.4rem 0 1.6rem 0; }
.section { border-top: 1px solid var(--ink); margin-top: 2.6rem; padding-top: 0.7rem; }
.section .number { font-family: 'IBM Plex Mono', monospace; color: var(--muted);
  font-size: 0.85rem; margin-right: 0.6rem; }
.section h2 { display: inline; font-size: 1.35rem; font-weight: 600; }
.section p { color: var(--ink); max-width: 46rem; line-height: 1.55; margin-top: 0.6rem; }

table.paper { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.6rem 0 0.3rem 0;
  border-top: 1.5px solid var(--ink); border-bottom: 1.5px solid var(--ink); }
table.paper th { font-weight: 500; text-align: left; padding: 7px 10px;
  border-bottom: 1px solid var(--ink); color: var(--ink); background: transparent; }
table.paper td { padding: 5px 10px; border: none; color: var(--ink); background: transparent; }
table.paper tr.group td { border-top: 1px solid var(--rule); }
table.paper td.num, table.paper th.num { text-align: right; font-family: 'IBM Plex Mono', monospace;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
table.paper td.strong { font-weight: 600; }
table.paper td.quiet { color: var(--muted); }
.caption { font-size: 0.85rem; color: var(--muted); line-height: 1.45; max-width: 46rem;
  margin: 0.3rem 0 1.4rem 0; }
.caption b { color: var(--ink); font-weight: 600; }

.stat { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--muted);
  margin: 0.4rem 0 1rem 0; }
.patch { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; line-height: 1.5;
  margin: 0.25rem 0 1.1rem 0; color: var(--muted); }
.patch b { color: var(--ink); font-weight: 500; }
.right { color: var(--right); }
.wrong { color: var(--wrong); text-decoration: underline; text-decoration-thickness: 1px; }
[data-testid="stImage"] img { border-radius: 0; }
.stButton button { border-radius: 2px; border: 1px solid var(--ink); background: transparent;
  font-family: 'IBM Plex Sans', sans-serif; }
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def header(kicker, title, lead):
    st.markdown("<div class='kicker'>" + html.escape(kicker) + "</div>",
                unsafe_allow_html=True)
    st.title(title)
    st.markdown("<p class='lead'>" + lead + "</p>", unsafe_allow_html=True)


def section(number, title, text=""):
    body = ""
    if text != "":
        body = "<p>" + text + "</p>"
    st.markdown("<div class='section'><span class='number'>" + str(number)
                + "</span><h2>" + html.escape(title) + "</h2>" + body + "</div>",
                unsafe_allow_html=True)


def table(rows, columns, caption="", group=None):
    """A journal-style table.

    ``columns`` is a list of (key, heading, kind) with kind "text", "num", "strong" or
    "quiet". When ``group`` names a column, its value is printed only on the first row
    of each run and a thin rule separates the runs.
    """
    parts = ["<table class='paper'><thead><tr>"]
    for key, heading, kind in columns:
        css = ""
        if kind == "num":
            css = " class='num'"
        parts.append("<th" + css + ">" + html.escape(heading) + "</th>")
    parts.append("</tr></thead><tbody>")
    previous = None
    for row in rows:
        row_css = ""
        if group is not None and previous is not None and row[group] != previous:
            row_css = " class='group'"
        parts.append("<tr" + row_css + ">")
        for key, heading, kind in columns:
            value = str(row[key])
            if group is not None and key == group and row[group] == previous:
                value = ""
            css = ""
            if kind != "text":
                css = " class='" + kind + "'"
            parts.append("<td" + css + ">" + html.escape(value) + "</td>")
        parts.append("</tr>")
        if group is not None:
            previous = row[group]
    parts.append("</tbody></table>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    if caption != "":
        st.markdown("<div class='caption'>" + caption + "</div>", unsafe_allow_html=True)


def figure(path, caption, width=None):
    if not path.exists():
        return
    if width is None:
        st.image(str(path), use_container_width=True)
    else:
        st.image(str(path), width=width)
    st.markdown("<div class='caption'>" + caption + "</div>", unsafe_allow_html=True)


def note(text):
    st.markdown("<div class='stat'>" + text + "</div>", unsafe_allow_html=True)
