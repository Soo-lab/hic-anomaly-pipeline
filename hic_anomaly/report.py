"""Single self-contained HTML report (inline CSS/JS, base64 PNGs)."""

import base64
import html
import os
import platform
import sys
from datetime import datetime, timezone

_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;"
    "padding:0 1em;color:#222}table{border-collapse:collapse;width:100%}"
    "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
    "th{cursor:pointer;background:#f0f0f0}td:first-child,th:first-child"
    "{text-align:left}img{max-width:100%}figure{margin:1.5em 0}"
    "figcaption{font-size:.9em;color:#555}"
)
_JS = (
    "document.querySelectorAll('th').forEach(function(h,c){h.onclick=function(){"
    "var t=h.closest('table'),b=t.tBodies[0],r=Array.from(b.rows),"
    "d=h.dataset.d=h.dataset.d==='1'?'-1':'1';r.sort(function(x,y){"
    "var a=x.cells[c].textContent,e=y.cells[c].textContent,"
    "n=parseFloat(a),m=parseFloat(e);"
    "if(!isNaN(n)&&!isNaN(m))return(n-m)*d;return a.localeCompare(e)*d});"
    "r.forEach(function(x){b.appendChild(x)})}});"
)


def _versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "pysam"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "n/a"
    try:
        from . import __version__

        v["hic_anomaly"] = __version__
    except Exception:
        pass
    return v


def _kv_table(d):
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in d.items()
    )
    return f"<table>{rows}</table>"


def write_html_report(outdir, run_info, figures, calls_df, name="report.html"):
    """Write outdir/report.html. figures: list of PNG paths or (caption, path)."""
    os.makedirs(outdir, exist_ok=True)
    run_info = dict(run_info or {})
    parts = [f"<h1>{html.escape(str(run_info.get('title', 'Hi-C anomaly report')))}</h1>"]
    parts.append(f"<p>Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}</p>")
    parts.append("<h2>Parameters</h2>" + _kv_table(run_info))
    parts.append("<h2>Figures</h2>")
    for fig in figures or []:
        cap, path = fig if isinstance(fig, (tuple, list)) else (os.path.basename(fig), fig)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        parts.append(
            f'<figure><img alt="{html.escape(str(cap))}" '
            f'src="data:image/png;base64,{b64}">'
            f"<figcaption>{html.escape(str(cap))}</figcaption></figure>"
        )
    parts.append("<h2>Calls</h2>")
    if calls_df is None or len(calls_df) == 0:
        parts.append("<p>No anomalous regions called.</p>")
    else:
        cols = list(calls_df.columns)
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(f'{v:.4g}' if isinstance(v, float) else str(v))}</td>"
                for v in row
            )
            + "</tr>"
            for row in calls_df.itertuples(index=False)
        )
        parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
    parts.append("<h2>Versions</h2>" + _kv_table(_versions()))
    doc = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Hi-C anomaly report'
        f"</title><style>{_CSS}</style></head><body>{''.join(parts)}"
        f"<script>{_JS}</script></body></html>"
    )
    out = os.path.join(outdir, name)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out
