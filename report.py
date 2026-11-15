"""Credit memo: internal rating grade, memo content as format-neutral blocks, and Markdown / Word / PDF renderers.

The memo is built once as a list of blocks - ("title", text), ("subtitle", text), ("h2", text), ("p", text),
("bullets", [text]), ("table", header, rows), ("image", png_bytes, caption), ("callout", text, rgb) - and rendered
three ways, so the PDF, the Word file and the on-screen Markdown always say the same thing. **bold** marks emphasis.
Charts are drawn with matplotlib (no browser needed, unlike Plotly image export).
"""
import io
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

import metrics as mt
import insights as ins

# ------------------------------------------------------------------ rating scale
# Indicative one-year PDs are rounded long-run average corporate default rates for the equivalent agency grade.
GRADES = [  # (minimum score, internal grade, agency-style equivalent, indicative 1-year PD, meaning)
    (90, "CR1", "AAA", 0.0001, "Highest safety"),
    (84, "CR2", "AA", 0.0002, "High safety"),
    (78, "CR3", "A", 0.0006, "Adequate to high safety"),
    (70, "CR4", "BBB", 0.0018, "Moderate safety - lowest investment grade"),
    (62, "CR5", "BB", 0.0070, "Moderate risk - sub-investment grade"),
    (54, "CR6", "B+", 0.0200, "High risk"),
    (46, "CR7", "B", 0.0500, "High risk"),
    (38, "CR8", "CCC", 0.1500, "Very high risk"),
    (0, "CR9", "CC / C", 0.3000, "Near default"),
]
EW_NOTCH = {"ELEVATED CONCERN": -2, "WATCH": -1, "COMFORTABLE": 0}
MGMT_NOTCH = {"Strong": 1, "Adequate": 0, "Weak": -1}
OUTLOOK_NOTCH = {"Positive": 0, "Stable": 0, "Negative": -1}
MAX_QUAL = 2               # qualitative and early-warning notches together move the grade at most 2 steps either way


def grade_index(score):
    if score is None or pd.isna(score):
        return None
    for i, (lo, *_rest) in enumerate(GRADES):
        if score >= lo:
            return i
    return len(GRADES) - 1


def grade_info(i):
    lo, g, eq, pd_, meaning = GRADES[max(0, min(i, len(GRADES) - 1))]
    return dict(grade=g, equiv=eq, pd=pd_, meaning=meaning, label=f"{g} (≈ {eq})")


def rate(score, ew_level=None, management="Adequate", outlook="Stable", security_cover=None):
    """Issuer and facility grade with the step-by-step build-up. Returns dict(issuer, facility, steps)."""
    base = grade_index(score)
    if base is None:
        return None
    steps = [("Quantitative score (textbook cut-offs)", f"{score:.0f} / 100", grade_info(base)["label"])]
    qual = 0
    for label, n, why in (("Early-warning signals", EW_NOTCH.get(ew_level, 0), ew_level.title() if ew_level else "not available"),
                          ("Management quality", MGMT_NOTCH.get(management, 0), management),
                          ("Industry outlook", OUTLOOK_NOTCH.get(outlook, 0), outlook)):
        qual += n
        steps.append((label, f"{why}: {n:+d} notch" + ("" if abs(n) == 1 else "es"), None))
    qual_c = max(-MAX_QUAL, min(MAX_QUAL, qual))
    if qual_c != qual:
        steps.append(("Cap on adjustments", f"limited to {qual_c:+d} notches", None))
    issuer_i = max(0, min(len(GRADES) - 1, base - qual_c))
    steps.append(("Issuer grade", "", grade_info(issuer_i)["label"]))
    sec_n = 1 if (security_cover is not None and not pd.isna(security_cover) and security_cover >= 1.5) else 0
    fac_i = max(0, issuer_i - sec_n)
    steps.append(("Security cover", ("n/a (unsecured)" if security_cover is None or pd.isna(security_cover) else f"{security_cover:.2f}x")
                  + f": {sec_n:+d} notch (≥ 1.5x cover lifts the facility one step)", None))
    steps.append(("Facility grade", "", grade_info(fac_i)["label"]))
    return dict(issuer=grade_info(issuer_i), facility=grade_info(fac_i), steps=steps, base=grade_info(base))


def relative_standing(sr_score, tb_score, sector):
    """How the borrower sits against its own sector (sector-relative score), as text. bias=True when the textbook grade
    looks weak only because the whole sector does (typical of utilities and infrastructure)."""
    if sr_score is None or pd.isna(sr_score):
        return dict(text="Sector-relative standing not available.", bias=False)
    pos = ("among the stronger names" if sr_score >= 75 else "around the norm" if sr_score >= 55 else "among the weaker names")
    tb_i = grade_index(tb_score)
    bias = tb_i is not None and tb_i >= 5 and sr_score >= 65
    txt = (f"Against {sector} peers (sector-relative cut-offs) the score is {sr_score:.0f}: {pos} in the sector. "
           "This is a ranking among peers, so it is not mapped to a grade.")
    if bias:
        txt += (" **The grade looks weak mainly because the whole sector does:** textbook cut-offs penalise capital-intensive "
                "businesses such as utilities and infrastructure, and agency ratings also weigh ownership, regulation and state "
                "support, which this model leaves out. Read the grade together with this sector standing.")
    return dict(text=txt, bias=bias, score=sr_score, position=pos)


# ------------------------------------------------------------------ charts (matplotlib, PNG bytes)
BLUE, ORANGE, GREY, GREEN_, AMBER_, RED_ = "#1f6feb", "#e8710a", "#8b949e", "#2da44e", "#d29922", "#cf222e"


def _png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


def chart_pillars(p_a, name_a, p_b=None, name_b=None):
    plt = _plt()
    keys = [k for k in p_a if not pd.isna(p_a.get(k))]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(keys))
    h = 0.38 if p_b else 0.6
    ax.barh(y - (h / 2 if p_b else 0), [p_a[k] for k in keys], h, color=BLUE, label=name_a)
    if p_b:
        ax.barh(y + h / 2, [0 if pd.isna(p_b.get(k)) else p_b.get(k) for k in keys], h, color=GREY, label=name_b)
    ax.set_yticks(y, keys)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Factor score (0-100)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    return _png(fig)


def chart_trends(m_df, name, bench_df=None, bench_name=None, keys=("icr", "dscr", "nd_ebitda", "cce")):
    plt = _plt()
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 3.6))
    drawn = 0
    for ax, k in zip(axs.flat, keys):
        meta = mt.M[k]
        for df, nm, col, ls in ((m_df, name, BLUE, "-"), (bench_df, bench_name, GREY, "--")):
            if df is not None and k in df.index:
                s = df.loc[k].dropna()
                if len(s):
                    ax.plot([f"FY{y}" for y in s.index], s.values, ls, color=col, marker="o", ms=3, lw=1.8, label=nm)
                    drawn += 1
        g, r = mt.thresholds(k)
        if g is not None and r is not None:
            ax.axhline(g, color=GREEN_, lw=0.8, alpha=0.7)
            ax.axhline(r, color=RED_, lw=0.8, alpha=0.7)
        ax.set_title(meta["label"], fontsize=8)
        if meta["unit"] == "%":
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    if not drawn:
        plt.close(fig)
        return None
    axs.flat[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return _png(fig)


def chart_heatmap(grid, target, dm_list, dr_list):
    if grid is None or getattr(grid, "empty", True):
        return None
    plt = _plt()
    from matplotlib.colors import ListedColormap
    z = grid.values.astype(float)
    cls = np.where(np.isnan(z), np.nan, np.where(z < 1.0, 0, np.where(z < target, 1, 2)))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.imshow(cls, cmap=ListedColormap([RED_, AMBER_, GREEN_]), vmin=0, vmax=2, aspect="auto", alpha=0.8)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(z[i, j]) else f"{z[i, j]:.2f}x", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(dm_list)), [f"{d * 100:+.0f}pp" for d in dm_list])
    ax.set_yticks(range(len(dr_list)), [f"+{d * 100:.0f}pp" for d in dr_list])
    ax.set_xlabel("Change in EBITDA margin")
    ax.set_ylabel("Change in interest rate")
    return _png(fig)


def chart_dd(hist):
    if hist is None or len(hist) < 3:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    top = max(float(hist["DD"].max()) * 1.1, 4)
    ax.axhspan(0, 1.645, color=RED_, alpha=0.12)
    ax.axhspan(1.645, 2.326, color=AMBER_, alpha=0.12)
    ax.plot(hist.index, hist["DD"], color=BLUE, marker="o", ms=3, lw=1.8)
    ax.set_ylim(0, top)
    ax.set_ylabel("Distance-to-default")
    fig.autofmt_xdate()
    return _png(fig)


# ------------------------------------------------------------------ memo content
def _pct(v, d=1):
    if v is None or pd.isna(v):
        return "n/a"
    return "< 0.01%" if 0 <= v < 0.0001 else f"{v:.{d}%}"


def _tint(rgb, k=0.85):
    """A light tint of a colour (blend towards white) for callout backgrounds."""
    return tuple(int(c + (255 - c) * k) for c in rgb)


def build_memo(c):
    """c: dict with every input (see app.py). Returns the list of blocks."""
    fmt = ins.fmt
    vals, flags = c["vals"], c["flags"]
    v = ins.verdict(vals, flags)
    req, scen = c.get("req"), c.get("scen")
    breaches = sum(1 for r in (c.get("cov_rows") or []) if r[3] == "Breach")
    sys_stance, sys_why = ins.stance(v["signal"], scen["pro"]["dscr"] if scen else None, breaches, req["target"] if req else 1.25)
    ov = c.get("override")
    stance_txt, why = (ov, c.get("override_reason") or "") if ov and ov != "Use system stance" else (sys_stance, sys_why)
    rating = c.get("rating")
    a = c.get("analyst", {})
    ew = c.get("ew")

    B = [("title", f"Credit appraisal note - {c['name']}"),
         ("subtitle", f"{c['sector']} · {c['as_of']} · prepared {datetime.now():%d %b %Y}"
                      + (f" by {a['name']}" if a.get("name") else "") + " · draft for analyst review, not a credit decision")]

    # 1. executive summary
    head = []
    if rating:
        head.append(f"**Issuer grade {rating['issuer']['label']}** · facility grade **{rating['facility']['label']}** · "
                    f"indicative 1-year PD **{_pct(rating['facility']['pd'], 2)}**")
    head.append(f"**Recommendation: {stance_txt}.** {why}")
    B.append(("callout", "  \n".join(head), (46, 160, 67) if stance_txt == "Supportable" else
              (210, 153, 34) if stance_txt.startswith("Supportable") else (207, 34, 46)))
    strengths = [f"{mt.M[k]['label']} {fmt(k, vals[k])}" for k in ins.KEY_STRENGTH if mt.rag(k, vals.get(k)) == "G"][:3]
    risks = [f"{mt.M[k]['label']} {fmt(k, vals[k])}" for k in ins.REASON_ORDER if mt.rag(k, vals.get(k)) == "R"][:3]
    summ = [f"Headline signal **{v['signal']}** - {v['g']} healthy, {v['a']} watch, {v['r']} concern metrics "
            f"({c.get('basis', 'Textbook').lower()} cut-offs).",
            "Strengths: " + ("; ".join(strengths) or "none clearly in the healthy zone") + ".",
            "Main risks: " + ("; ".join(risks) or "no metric in the concern zone") + "."]
    if ew:
        summ.append(f"Early warning: **{ew['level'].title()}** ({ew['n']['R']} concern, {ew['n']['A']} watch of {len(ew['sigs'])} signals).")
    B += [("h2", "1. Executive summary"), ("bullets", summ)]

    # 2. borrower and request
    sec = a.get("collateral_type", "Unsecured")
    cover = c.get("security_cover")
    rows = [["Borrower", c["name"]], ["Sector", c["sector"]]]
    if req:
        rows += [["Facility", req["type"]], ["Amount", f"Rs {req['amount']:,.0f} cr"], ["Rate", f"{req['rate']:.2%}"],
                 ["Tenor", "revolving" if not req["type"].startswith("Term") else f"{req['tenor']:.0f} years"],
                 [f"Largest facility at DSCR ≥ {req['target']:.2f}x", f"Rs {req['capacity']:,.0f} cr"]]
    rows += [["Purpose", a.get("purpose") or "not stated"],
             ["Security", sec + (f" worth Rs {a.get('collateral_value', 0):,.0f} cr" if sec != "Unsecured" else "")],
             ["Security cover", "n/a" if cover is None or pd.isna(cover) else f"{cover:.2f}x"]]
    B += [("h2", "2. Borrower and request"), ("table", ["Item", "Detail"], rows)]

    # 3. rating
    if rating:
        B += [("h2", "3. Internal rating"),
              ("p", f"The quantitative score on textbook (absolute) cut-offs maps to **{rating['base']['label']}**; early-warning and analyst judgements move it by "
                    f"whole notches (at most {MAX_QUAL} either way), and security can lift the facility one step above the issuer."),
              ("table", ["Step", "Effect", "Grade"], [[s, e, g or ""] for s, e, g in rating["steps"]])]
        if rating.get("relative"):
            B.append(("p", rating["relative"]["text"]))
        pil = c.get("pillars") or {}
        B.append(("p", "Factor scores: " + "; ".join(f"{p} {s:.0f}" for p, s in pil.items() if not pd.isna(s)) + "."))
        if c.get("charts", {}).get("pillars"):
            B.append(("image", c["charts"]["pillars"], f"Factor scores: {c['name']} vs {c.get('cmp_name', 'comparator')}"))
        B.append(("p", f"Grade scale: CR1 (≈ AAA) to CR9 (≈ CC/C). Indicative PDs are rounded long-run average default rates for the "
                       "equivalent agency grade, not a model estimate for this company."))

    # 4. financials
    snap = c["snap"]
    B += [("h2", "4. Financial snapshot (Rs crore)"), ("table", ["Metric"] + snap["cols"], [[l] + vs for l, vs in snap["rows"]])]
    if c.get("charts", {}).get("trends"):
        B.append(("image", c["charts"]["trends"], "Key credit ratios (green / red lines = healthy / concern cut-offs in force)"))

    # 5. strengths and risks
    st_all = [f"{mt.M[k]['label']}: {fmt(k, vals[k])}" for k in ins.KEY_STRENGTH if mt.rag(k, vals.get(k)) == "G"][:6]
    rk_all = [f"{mt.M[k]['label']}: {fmt(k, vals[k])}" for k in mt.M if mt.rag(k, vals.get(k)) == "R"][:8]
    rk_all += [t for _, t in flags]
    bf = c.get("benford")
    if bf and bf.get("flag") is not None:
        rk_all.append(f"Benford screen: KS {bf['ks']:.3f} vs cut-off {bf['cutoff']:.3f} - "
                      + ("unusual leading-digit pattern, verify source data" if bf["flag"] else "no unusual pattern"))
    B += [("h2", "5. Strengths"), ("bullets", st_all or ["None clearly in the healthy zone."]),
          ("h2", "6. Risks and red flags"), ("bullets", rk_all or ["No concern-zone metrics or red flags."])]

    # 7. early warning
    B.append(("h2", "7. Early warning (latest quarter and market data)"))
    if ew:
        flagged = [s for s in ew["sigs"] if s["status"] in ("R", "A")]
        B.append(("p", f"Verdict **{ew['level'].title()}**: {ew['n']['R']} concern, {ew['n']['A']} watch, {ew['n']['G']} clear."
                       + (f" Merton distance-to-default {ew['mert']['DD']:.1f} (1-year PD {_pct(ew['mert']['PD'], 2)})."
                          if ew.get("mert") and not ew["mert"].get("no_debt") else "")))
        if flagged:
            B.append(("table", ["Signal", "Value", "Status"],
                      [[s["signal"], s["value"], {"R": "Concern", "A": "Watch"}[s["status"]]] for s in flagged]))
        if c.get("charts", {}).get("dd"):
            B.append(("image", c["charts"]["dd"], "Merton distance-to-default, last 12 months (red: PD > 5%, amber: PD > 1%)"))
    else:
        B.append(("p", "Not available (no quarterly or market data, or the dashboard is set to a past year)."))

    # 8. stress
    if scen:
        rows = []
        for label, k, f in (("Interest cover", "icr", "{:.2f}x"), ("DSCR", "dscr", "{:.2f}x"), ("Debt / Equity", "de", "{:.2f}x"),
                            ("Net debt / EBITDA", "nd_ebitda", "{:.2f}x"), ("Operating cash flow (Rs cr)", "cfo", "{:,.0f}")):
            rows.append([label] + [("n/a" if pd.isna(scen[x][k]) else f.format(scen[x][k])) for x in ("base", "pro", "stress")])
        B += [("h2", "8. Pro-forma and stress"), ("table", ["", "Current", "With facility", "Stressed"], rows),
              ("p", f"Stress case: {scen['desc']}.")]
        if c.get("charts", {}).get("heatmap"):
            B.append(("image", c["charts"]["heatmap"], "DSCR with the facility under margin and rate shocks (red < 1.0x, amber < target)"))

    # 9. covenants
    if c.get("cov_rows"):
        B += [("h2", "9. Covenant headroom (latest year)"),
              ("table", ["Covenant", "Actual", "Limit", "Status"], [list(r) for r in c["cov_rows"]])]

    # 10. analyst assessment
    ass = [f"Management quality: **{a.get('management', 'Adequate')}**", f"Industry outlook: **{a.get('outlook', 'Stable')}**"]
    if a.get("notes"):
        ass.append(f"Analyst notes: {a['notes']}")
    B += [("h2", "10. Analyst assessment"), ("bullets", ass)]

    # 11. recommendation
    conds = ins.conditions(vals, stance_txt)
    if cover is not None and not pd.isna(cover) and cover < 1.0 and sec != "Unsecured":
        conds.append(f"Security cover is {cover:.2f}x - below 1.0x; seek additional collateral or reduce the amount.")
    rec = [("h2", "11. Recommendation"), ("p", f"**{stance_txt}.** {why}")]
    if ov and ov != "Use system stance":
        rec.append(("p", f"*Analyst override of the system stance ({sys_stance}).*"))
    rec += [("p", "**Conditions / next steps**"), ("bullets", conds)]
    B += rec
    B.append(("p", "*Limitations: annual and quarterly public statements only (Yahoo Finance); DSCR is a proxy; the grade is an "
                   "internal-style indication, not an agency rating. Figures in Rs crore.*"))
    return B


# ------------------------------------------------------------------ renderers
def _plain(t):
    return re.sub(r"\*\*(.+?)\*\*", r"\1", re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", str(t)))


def to_markdown(blocks):
    out = []
    for b in blocks:
        k = b[0]
        if k == "title":
            out += [f"# {b[1]}", ""]
        elif k == "subtitle":
            out += [f"*{b[1]}*", ""]
        elif k == "h2":
            out += [f"## {b[1]}", ""]
        elif k in ("p", "callout"):
            out += [("> " + b[1].replace("\n", "\n> ")) if k == "callout" else b[1], ""]
        elif k == "bullets":
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "table":
            out += ["| " + " | ".join(b[1]) + " |", "|" + "---|" * len(b[1])]
            out += ["| " + " | ".join(str(x) for x in r) + " |" for r in b[2]] + [""]
        elif k == "image":
            out += [f"*[Chart: {b[2]} - included in the PDF and Word versions]*", ""]
    return "\n".join(out)


def _docx_runs(par, text):
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        italic = part.startswith("*") and part.endswith("*") and len(part) > 2
        run = par.add_run(part.strip("*") if italic else part)
        run.bold = i % 2 == 1
        run.italic = italic


def to_docx(blocks):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor, Inches

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.8)
        sec.top_margin = sec.bottom_margin = Inches(0.7)
    for b in blocks:
        k = b[0]
        if k == "title":
            doc.add_heading(b[1], level=0)
        elif k == "subtitle":
            p = doc.add_paragraph()
            r = p.add_run(b[1]); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        elif k == "h2":
            doc.add_heading(b[1], level=2)
        elif k == "p":
            _docx_runs(doc.add_paragraph(), b[1])
        elif k == "callout":
            t = doc.add_table(rows=1, cols=1)
            cell = t.rows[0].cells[0]
            shd = OxmlElement("w:shd")
            shd.set(qn("w:fill"), "%02X%02X%02X" % _tint(b[2]))
            cell._tc.get_or_add_tcPr().append(shd)
            cell.paragraphs[0].text = ""
            for i, line in enumerate(b[1].split("  \n")):
                par = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
                _docx_runs(par, line)
            doc.add_paragraph()
        elif k == "bullets":
            for x in b[1]:
                _docx_runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "table":
            t = doc.add_table(rows=1, cols=len(b[1]))
            t.style = "Light Grid Accent 1"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(b[1]):
                t.rows[0].cells[i].text = _plain(h)
            for row in b[2]:
                cells = t.add_row().cells
                for i, x in enumerate(row):
                    cells[i].text = _plain(x)
            for row in t.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        for run in par.runs:
                            run.font.size = Pt(8.5)
            doc.add_paragraph()
        elif k == "image":
            doc.add_picture(io.BytesIO(b[1]), width=Inches(6.0))
            cap = doc.add_paragraph()
            r = cap.add_run(b[2]); r.italic = True; r.font.size = Pt(8)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _font_dir():
    import matplotlib
    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf")


def to_pdf(blocks):
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(15, 15, 15)
    fd = _font_dir()
    pdf.add_font("DejaVu", "", os.path.join(fd, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(fd, "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", os.path.join(fd, "DejaVuSans-Oblique.ttf"))
    pdf.add_font("DejaVu", "BI", os.path.join(fd, "DejaVuSans-BoldOblique.ttf"))
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def md(text):
        """Our markup -> fpdf2's: **bold** stays, *italic* becomes __italic__."""
        return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"__\1__", text.replace("  \n", "\n"))

    def numeric(col):
        vals = [str(x) for x in col if str(x).strip() not in ("", "n/a", "–", "-")]
        return bool(vals) and sum(bool(re.match(r"^(Rs |[≥≤<>] ?)?[-+−]?[\d.,]+(x|%| cr| d)?$", v)) for v in vals) >= 0.7 * len(vals)

    def para(text, size=9.5, style=""):
        pdf.set_font("DejaVu", style, size)
        pdf.multi_cell(W, 5, md(text), markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")

    for b in blocks:
        k = b[0]
        if k == "title":
            pdf.set_font("DejaVu", "B", 17)
            pdf.multi_cell(W, 8, b[1], new_x="LMARGIN", new_y="NEXT")
        elif k == "subtitle":
            pdf.set_text_color(110, 110, 110)
            para(b[1], 8, "I")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        elif k == "h2":
            pdf.ln(2)
            pdf.set_font("DejaVu", "B", 12)
            pdf.set_text_color(31, 111, 235)
            pdf.multi_cell(W, 7, b[1], new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        elif k == "p":
            para(b[1])
            pdf.ln(1)
        elif k == "callout":
            pdf.set_fill_color(*_tint(b[2]))
            pdf.set_draw_color(*b[2])
            pdf.set_font("DejaVu", "", 10)
            pdf.multi_cell(W, 6, md(b[1]), border="L", fill=True, markdown=True, padding=3, new_x="LMARGIN", new_y="NEXT")
            pdf.set_fill_color(255, 255, 255)
            pdf.set_draw_color(190, 190, 190)
            pdf.ln(2)
        elif k == "bullets":
            pdf.set_font("DejaVu", "", 9.5)
            for x in b[1]:
                pdf.set_x(pdf.l_margin + 3)
                pdf.multi_cell(W - 3, 5, "•  " + md(x), markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif k == "table":
            pdf.set_font("DejaVu", "", 8)
            pdf.set_draw_color(190, 190, 190)
            n = len(b[1])
            first = 0.34 if n > 2 else 0.4
            widths = [first] + [(1 - first) / (n - 1)] * (n - 1)
            align = ["LEFT"] + ["RIGHT" if numeric([r[i] for r in b[2]]) else "LEFT" for i in range(1, n)]
            with pdf.table(col_widths=[w * 100 for w in widths], text_align=align,
                           headings_style=FontFace(emphasis="BOLD", fill_color=(232, 238, 248)), line_height=4.6,
                           padding=1.2, cell_fill_color=(248, 249, 251), cell_fill_mode="ROWS") as t:
                t.row([_plain(h) for h in b[1]])
                for row in b[2]:
                    t.row([_plain(x) for x in row])
            pdf.ln(2)
        elif k == "image":
            from PIL import Image
            im = Image.open(io.BytesIO(b[1]))
            w_mm = min(W, 170)
            h_mm = w_mm * im.height / im.width
            if pdf.get_y() + h_mm + 8 > pdf.h - pdf.b_margin:
                pdf.add_page()
            pdf.image(io.BytesIO(b[1]), x=pdf.l_margin + (W - w_mm) / 2, w=w_mm)
            pdf.set_text_color(110, 110, 110)
            para(b[2], 7.5, "I")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
    return bytes(pdf.output())
