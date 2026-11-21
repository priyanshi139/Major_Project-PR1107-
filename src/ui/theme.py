"""Visual theme + HTML components for the dashboard (original styling, system fonts only)."""
import html
import streamlit as st

CSS = """
<style>
.stApp { font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: #f6f7fb; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.6rem !important; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -.01em; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(185deg,#0f172a 0%,#1e1b4b 60%,#2b1b4e 100%); }
[data-testid="stSidebar"] * { color: #e2e8f0; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
.ht-brand { font-size: 1.45rem; font-weight: 800; background: linear-gradient(90deg,#67e8f9,#a78bfa,#f0abfc);
            -webkit-background-clip: text; background-clip: text; color: transparent !important; margin-bottom: .2rem; }
.ht-side-card { text-align: center; padding: .8rem .4rem 0 .4rem; }
.ht-side-card .name { font-weight: 700; font-size: 1.02rem; color: #fff; }
.ht-side-card .sub { font-size: .8rem; opacity: .75; }
.ht-qs { font-size: .92rem; line-height: 1.9; padding: .2rem .2rem; }
.ht-qs b { color: #fff; }
.ht-pill-status { display:block; text-align:center; padding:.55rem; border-radius:12px; font-size:.85rem; font-weight:600;
                  background: rgba(251,191,36,.12); border: 1px solid rgba(251,191,36,.35); color:#fcd34d !important; }

[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] { background: rgba(255,255,255,.10) !important; border: 1px solid rgba(255,255,255,.25) !important; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] input { color: #f1f5f9 !important; -webkit-text-fill-color: #f1f5f9 !important; background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] button svg { fill: #e2e8f0 !important; }

/* ---------- top navigation (radio rendered as pills) ---------- */
.st-key-topnav { background: linear-gradient(90deg,#ede9fe,#e0e7ff 50%,#fae8ff); border-radius: 18px; padding: .5rem .6rem;
                 box-shadow: inset 0 0 0 1px rgba(99,102,241,.12); margin-bottom: 1.1rem; }
.st-key-topnav [role="radiogroup"] { gap: .3rem; flex-wrap: wrap; }
.st-key-topnav label[data-testid="stRadioOption"] { padding: .45rem .95rem; border-radius: 12px; cursor: pointer; margin: 0;
                       transition: all .15s ease; }
.st-key-topnav label[data-testid="stRadioOption"]:hover { background: rgba(99,102,241,.13); }
.st-key-topnav label[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }   /* hide the radio circle */
.st-key-topnav label[data-testid="stRadioOption"] p { font-weight: 600; font-size: .93rem; color: #4c4f7a; margin: 0; }
.st-key-topnav label[data-testid="stRadioOption"][data-selected="true"] { background: linear-gradient(135deg,#6366f1,#8b5cf6);
                       box-shadow: 0 6px 16px rgba(99,102,241,.35); }
.st-key-topnav label[data-testid="stRadioOption"][data-selected="true"] p { color: #fff !important; }

/* ---------- hero + headers ---------- */
.ht-hero { text-align: center; padding: 1.6rem 1rem 1.2rem 1rem; }
.ht-hero .logo { width: 74px; height: 74px; margin: 0 auto .8rem auto; border-radius: 22px; display:flex; align-items:center;
                 justify-content:center; font-size: 2.2rem; background: linear-gradient(135deg,#6366f1,#ec4899);
                 box-shadow: 0 12px 30px rgba(99,102,241,.4); }
.ht-hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0; background: linear-gradient(90deg,#4f46e5,#7c3aed,#db2777);
              -webkit-background-clip: text; background-clip: text; color: transparent; }
.ht-hero p { color: #5b6180; font-size: 1.05rem; max-width: 760px; margin: .6rem auto 0 auto; }
.ht-pagehead { background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%); color:#fff; border-radius: 20px;
               padding: 1.1rem 1.5rem; margin-bottom: .9rem; box-shadow: 0 10px 28px rgba(124,58,237,.28); }
.ht-pagehead h2 { margin:0; color:#fff; font-size: 1.55rem; font-weight: 750; }
.ht-pagehead p { margin:.2rem 0 0 0; opacity:.92; font-size:.95rem; }

/* ---------- cards ---------- */
.ht-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem; margin: .5rem 0 1.1rem 0; }
.ht-card { border-radius: 20px; padding: 1.2rem 1.2rem; color: #fff; box-shadow: 0 10px 26px rgba(0,0,0,.13);
           transition: transform .15s ease, box-shadow .15s ease; text-align: center; }
.ht-card:hover { transform: translateY(-3px); box-shadow: 0 16px 32px rgba(0,0,0,.18); }
.ht-card .l { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; opacity: .92; font-weight: 600; }
.ht-card .v { font-size: 2.15rem; font-weight: 800; line-height: 1.15; margin: .35rem 0; }
.ht-card .s { font-size: .8rem; opacity: .95; background: rgba(255,255,255,.22); display:inline-block; padding:.12rem .7rem; border-radius: 999px; }
.ht-ok { background: linear-gradient(135deg,#10b981,#6ee7b7); } .ht-warn { background: linear-gradient(135deg,#f59e0b,#fcd34d); }
.ht-crit { background: linear-gradient(135deg,#ef4444,#fb7185); } .ht-info { background: linear-gradient(135deg,#3b82f6,#22d3ee); }
.ht-neutral { background: linear-gradient(135deg,#64748b,#94a3b8); } .ht-purple { background: linear-gradient(135deg,#8b5cf6,#d946ef); }
.ht-pink { background: linear-gradient(135deg,#ec4899,#f472b6); }
.ht-badge { display:inline-block; padding:.2rem .75rem; border-radius:999px; font-size:.74rem; font-weight:700; color:#fff; margin-right:.3rem; }
.ht-profile { background: linear-gradient(160deg,#1e1b4b,#312e81 60%,#4c1d95); color:#e2e8f0; border-radius: 22px; padding: 1.5rem 1.2rem;
              box-shadow: 0 12px 30px rgba(49,46,129,.35); height: 100%; }
.ht-avatar { width: 78px; height: 78px; border-radius: 50%; background: linear-gradient(135deg,#818cf8,#c084fc); display:flex; align-items:center;
             justify-content:center; font-size: 1.6rem; font-weight:800; color:#fff; margin: 0 auto .7rem auto; box-shadow: 0 0 0 5px rgba(255,255,255,.12); }
.ht-feat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap: 1rem; margin: 1rem 0 1.4rem 0; }
.ht-feat { background:#fff; border-radius: 20px; padding: 1.3rem 1rem; text-align:center; box-shadow: 0 6px 20px rgba(30,27,75,.07);
           transition: transform .15s ease; }
.ht-feat:hover { transform: translateY(-4px); }
.ht-feat .ic { font-size: 2rem; width: 58px; height: 58px; border-radius: 18px; display:flex; align-items:center; justify-content:center;
               margin: 0 auto .6rem auto; background: linear-gradient(135deg,#e0e7ff,#fae8ff); }
.ht-feat .t { font-weight: 700; color: #1e1b4b; } .ht-feat .d { color:#6b7280; font-size:.88rem; margin-top:.2rem; }
.ht-pcard { background:#fff; border-radius: 20px; padding: 1.1rem 1.2rem; box-shadow: 0 6px 20px rgba(30,27,75,.08); display:flex; gap:1rem; align-items:center; }
.ht-pcard .av { width:56px; height:56px; flex: 0 0 56px; border-radius:50%; background: linear-gradient(135deg,#6366f1,#a855f7); color:#fff; display:flex;
                align-items:center; justify-content:center; font-weight:800; }
.ht-pcard .n { font-weight:700; color:#1e1b4b; } .ht-pcard .m { color:#6b7280; font-size:.85rem; margin:.15rem 0 .35rem 0; }
.ht-cond { background:#fff; border-radius:18px; padding: 1rem; box-shadow: 0 6px 18px rgba(30,27,75,.07); text-align:center; }
.ht-cond .ic { font-size: 1.6rem; } .ht-cond .t { font-weight:700; color:#1e1b4b; margin:.2rem 0; } .ht-cond .d { font-size:.82rem; color:#6b7280; }
.ht-cond.soon { background: transparent; box-shadow:none; border: 2px dashed #c7c9e6; opacity: .85; }
.ht-note { border-left: 4px solid #8b5cf6; background: rgba(139,92,246,.08); padding: .65rem .95rem; border-radius: 10px; font-size: .9rem; color:#3b3f66; }
.ht-footer { text-align:center; color:#9aa0b8; font-size:.78rem; margin: 2rem 0 .5rem 0; }

/* ---------- widgets ---------- */
.stButton > button { border-radius: 12px; border: 0; background: linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; font-weight:600;
                     padding: .5rem 1.1rem; transition: all .15s ease; box-shadow: 0 4px 12px rgba(99,102,241,.3); }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(99,102,241,.4); color:#fff; border: 0; }
.stButton > button:focus { color:#fff; border:0; }
[data-testid="stTabs"] [role="tablist"] { gap: .4rem; background: linear-gradient(90deg,#ede9fe,#fae8ff); padding:.4rem; border-radius: 14px; }
[data-testid="stTab"] { border-radius: 10px; padding: .45rem 1rem; transition: all .15s ease; }
[data-testid="stTab"]:hover { background: rgba(99,102,241,.12); }
[data-testid="stTab"][data-selected="true"] { background: linear-gradient(135deg,#6366f1,#8b5cf6); box-shadow: 0 4px 12px rgba(99,102,241,.3); }
[data-testid="stTab"][data-selected="true"] p { color:#fff !important; }
[data-testid="stTab"] > div[data-rac] { display:none !important; }
[data-testid="stSidebar"] .stButton > button { width: 100%; }
div[data-testid="stPlotlyChart"] { background:#fff; border-radius: 18px; padding: .4rem; box-shadow: 0 6px 18px rgba(30,27,75,.06); }

/* ---------- timeline + insights ---------- */
.ht-tl { position: relative; margin: .6rem 0 1rem .6rem; padding-left: 1.4rem; border-left: 3px solid #dfe3f5; }
.ht-tl-item { position: relative; background:#fff; border-radius: 14px; padding: .65rem .9rem; margin-bottom: .6rem; box-shadow: 0 4px 14px rgba(30,27,75,.06); }
.ht-tl-item::before { content:""; position:absolute; left:-1.95rem; top:.95rem; width:14px; height:14px; border-radius:50%; background: var(--c,#6366f1); border: 3px solid #f6f7fb; }
.ht-tl-item .t { font-weight: 650; color:#1e1b4b; font-size:.93rem; } .ht-tl-item .m { color:#7a809c; font-size:.78rem; margin-top:.1rem; }
.ht-ins { background:#fff; border-radius: 16px; padding: .85rem 1rem; margin-bottom:.7rem; box-shadow: 0 4px 14px rgba(30,27,75,.06); border-left: 5px solid var(--c,#6366f1); }
.ht-ins .t { color:#1f2340; font-size:.95rem; } .ht-ins .m { color:#7a809c; font-size:.78rem; margin-top:.3rem; }
</style>
"""
PROV = {"historical": ("PUBLIC DATASET RECORD", "info"), "synthetic": ("SYNTHETIC DEMO DATA", "warn"),
        "uploaded": ("UPLOADED DATA", "purple"), "live_entry": ("LIVE ENTRY", "ok"), None: ("NO PATIENT LOADED", "neutral")}
TONE_BG = {"ok": "#10b981", "warn": "#f59e0b", "crit": "#ef4444", "info": "#3b82f6", "neutral": "#64748b", "purple": "#8b5cf6", "pink": "#ec4899"}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, icon="🩺"):
    st.markdown(f'<div class="ht-hero"><div class="logo">{icon}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def page_header(title, subtitle, badges_html=""):
    st.markdown(f'<div class="ht-pagehead"><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p>'
                f'<div style="margin-top:.55rem">{badges_html}</div></div>', unsafe_allow_html=True)


def badge(text, tone="info"):
    return f'<span class="ht-badge" style="background:{TONE_BG.get(tone, "#64748b")}">{html.escape(str(text))}</span>'


def provenance_badge(provenance):
    text, tone = PROV.get(provenance, PROV[None])
    return badge(text, tone)


def cards(items):
    """items: list of (label, value, sub, tone)"""
    def _v(v):
        t = html.escape(str(v)); n = len(str(v))
        return f'<div class="v" style="font-size:{1.35 if n > 16 else 1.7}rem">{t}</div>' if n > 11 else f'<div class="v">{t}</div>'
    h = "".join(f'<div class="ht-card ht-{t}"><div class="l">{html.escape(str(l))}</div>{_v(v)}'
                f'<div class="s">{html.escape(str(s)) or "&nbsp;"}</div></div>' for l, v, s, t in items)
    st.markdown(f'<div class="ht-grid">{h}</div>', unsafe_allow_html=True)


def feature_cards(items):
    h = "".join(f'<div class="ht-feat"><div class="ic">{i}</div><div class="t">{html.escape(t)}</div><div class="d">{html.escape(d)}</div></div>'
                for i, t, d in items)
    st.markdown(f'<div class="ht-feat-grid">{h}</div>', unsafe_allow_html=True)


def patient_card(avatar, name, meta, badge_html):
    st.markdown(f'<div class="ht-pcard"><div class="av">{html.escape(avatar)}</div><div><div class="n">{html.escape(name)}</div>'
                f'<div class="m">{html.escape(meta)}</div>{badge_html}</div></div>', unsafe_allow_html=True)


def condition_cards(items):
    """items: list of (icon, title, detail, soon_bool)"""
    h = "".join(f'<div class="ht-cond{" soon" if soon else ""}"><div class="ic">{i}</div><div class="t">{html.escape(t)}</div>'
                f'<div class="d">{html.escape(d)}</div></div>' for i, t, d, soon in items)
    st.markdown(f'<div class="ht-grid">{h}</div>', unsafe_allow_html=True)


def profile_card(initials, name, line1, line2="", badge_html=""):
    st.markdown(f'<div class="ht-profile"><div class="ht-avatar">{html.escape(initials)}</div>'
                f'<div style="text-align:center;font-weight:700;font-size:1.15rem;color:#fff">{html.escape(name)}</div>'
                f'<div style="text-align:center;opacity:.85;font-size:.9rem;margin-top:.2rem">{html.escape(line1)}</div>'
                f'<div style="text-align:center;opacity:.7;font-size:.82rem;margin:.25rem 0 .6rem 0">{html.escape(line2)}</div>'
                f'<div style="text-align:center">{badge_html}</div></div>', unsafe_allow_html=True)


def sidebar_patient(initials, name, sub):
    st.markdown(f'<div class="ht-side-card"><div class="ht-avatar" style="width:68px;height:68px;font-size:1.3rem">{html.escape(initials)}</div>'
                f'<div class="name">{html.escape(name)}</div><div class="sub">{html.escape(sub)}</div></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="ht-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def footer(text):
    st.markdown(f'<div class="ht-footer">{html.escape(text)}</div>', unsafe_allow_html=True)


EVENT_COLORS = {"registration": "#64748b", "observation": "#3b82f6", "prediction": "#8b5cf6", "anomaly": "#f59e0b", "alert": "#ef4444",
                "state_change": "#14b8a6", "alert_action": "#10b981", "whatif": "#ec4899", "profile": "#94a3b8", "import": "#0ea5e9"}
SEV_COLORS = {"notable": "#ef4444", "attention": "#f59e0b", "info": "#6366f1"}


def timeline_html(events):
    """events: list of dicts with event_type, title, time (str), source (str)"""
    items = "".join(f'<div class="ht-tl-item" style="--c:{EVENT_COLORS.get(e["event_type"], "#6366f1")}"><div class="t">{html.escape(e["title"])}</div>'
                    f'<div class="m">{html.escape(e["time"])} · {html.escape(e["event_type"].replace("_", " "))}'
                    f'{" · " + html.escape(e["source"]) if e.get("source") else ""}</div></div>' for e in events)
    st.markdown(f'<div class="ht-tl">{items}</div>', unsafe_allow_html=True)


def insight_cards(items):
    h = "".join(f'<div class="ht-ins" style="--c:{SEV_COLORS.get(i["severity"], "#6366f1")}"><div class="t">{html.escape(i["text"])}</div>'
                f'<div class="m">Source: {html.escape(i["source"])} · Time: {html.escape(str(i["time"]))}</div></div>' for i in items)
    st.markdown(h, unsafe_allow_html=True)
