"""
app_analyst.py
===============
Streamlit-Dashboard — Qualitaetsbefund deutscher Krankenhaeuser.

Gestaltet als analytischer Befund, nicht als generisches BI-Dashboard:
    - Drei "Befund"-Karten fuehren die zentralen Erkenntnisse VOR den
      Rohdaten ein, nicht als Fussnote danach.
    - Farbsystem strikt getrennt: Cluster-Farben (Gruen/Blau/Rot-braun)
      erscheinen NUR fuer Cluster; ein einziger Akzentton (Bernstein)
      ist reserviert fuer Befund-Momente; alle anderen Kategorien
      (Traeger, Region, Groesse) nutzen eine eigene, klar andere Palette.

Seiten:
    1. Übersicht      -- Befunde, KPIs, Karte
    2. Cluster-Analyse -- Methodik-Nachweis im Detail
    3. Klinik-Profil   -- Einzelansicht mit Fachbereichs-Piktogrammen

Ausfuehren:
    streamlit run app_analyst.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATEI = Path("data/processed/master_clustered.parquet")

# =============================================================================
# DESIGN-TOKENS
# =============================================================================

COLOR = {
    "bg":        "#F2F6FA",
    "card":      "#FFFFFF",
    "border":    "#DCE4EC",
    "ink":       "#0E2A45",
    "ink_muted": "#55708C",
    "ueber":     "#4C9A4E",
    "im":        "#2E6FA8",
    "unter":     "#B5563A",
    "akzent":    "#B99A2E",   # gedaempftes Gold-Oliv, angelehnt an den Logo-Farbverlauf
}
CLUSTER_FARBEN = {"Über Ø": COLOR["ueber"], "Im Ø": COLOR["im"], "Unter Ø": COLOR["unter"],
                   "Keine QS-Daten": "#9AA5B1"}
CLUSTER_REIHENFOLGE = ["Unter Ø", "Im Ø", "Über Ø"]

# Eigene Palette fuer NICHT-Cluster-Kategorien -- bewusst ohne Gruen/Blau/Rot-braun
NEBEN_FARBEN = ["#6B5B8A", "#2E8F8F", "#9B7B3F", "#6E8F6E", "#8A4B6B"]
GROESSEN_FARBEN = {"Klein": "#6B5B8A", "Mittel": "#2E8F8F", "Gross": "#9B7B3F", "Sehr gross": "#8A4B6B"}
GROESSENKLASSEN_ALLE = ["Klein", "Mittel", "Gross", "Sehr gross"]

FACHBEREICHE = {
    "geburtshilfe_gynaekologie": ("Geburtshilfe & Gynäkologie", "score_geburtshilfe_gynaekologie"),
    "kardiologie_herzchirurgie": ("Kardiologie & Herzchirurgie", "score_kardiologie_herzchirurgie"),
    "orthopaedie_chirurgie":     ("Orthopädie & Chirurgie", "score_orthopaedie_chirurgie"),
    "allgemeine_versorgung":     ("Allgemeine Versorgung", "score_allgemeine_versorgung"),
    "onkologie":                 ("Onkologie", "score_onkologie"),
    "transplantation":           ("Transplantation", "score_transplantation"),
}

FEATURE_LABELS = {
    "feat_arztdichte":   "Arztdichte (relativ)",
    "feat_pflegedichte": "Pflegedichte (relativ)",
    "feat_ppr":          "Pflegepersonal-Erfüllungsgrad",
    "feat_ausstattung":  "Geräteausstattung (Breite)",
    "feat_ausbildung":   "Fortbildungsangebot (Breite)",
    "feat_notfall":      "Notfallversorgungsstufe",
    "feat_hygiene":      "Hygienemaßnahmen (Breite)",
    "feat_rm":           "Risikomanagement (Breite)",
    "feat_amts":         "Arzneimittelsicherheit (Breite)",
    "feat_bm":           "Beschwerdemanagement (Breite)",
    "feat_if_score":     "Fehlermeldesysteme (Breite)",
    "feat_fehler":       "Fehlermanagement/CIRS (Breite)",
}

# Feste, dokumentierte Kennzahl aus der Methodik (02_transform.py) -- die
# Rohdaten (QS_Qualitätsindikator.csv) liegen dem Dashboard nicht vor, daher
# hier als bekannte Konstante hinterlegt, nicht zur Laufzeit neu berechnet.
QI_INDIKATOREN_GESAMT = 199

DE_CENTER = dict(lat=51.2, lon=10.4)
DE_ZOOM = 5.7

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; font-size: 42px; }}
.stApp {{ background-color: {COLOR["bg"]}; color: {COLOR["ink"]}; }}
.block-container {{
    padding-top: 3.2rem !important; padding-bottom: 1rem !important;
    max-width: 100% !important;
}}
p, li, span, label, div {{ font-size: 42px !important; color: {COLOR["ink"]}; }}
[data-testid="stSidebar"] {{ background-color: #EEF1F5 !important; border-right: 1px solid {COLOR["border"]}; min-width: 320px !important; }}
[data-testid="stSidebar"] .stMarkdown p {{ font-size: 42px !important; }}

/* Explizite Textfarben -- unabhaengig vom Hell/Dunkel-Modus des Browsers,
   damit Text auf hellem Kartenhintergrund IMMER lesbar bleibt. */
button p, button span, button div {{ color: inherit !important; font-size: 42px !important; }}
[data-testid="stSidebar"] button[kind="secondary"] {{
    background-color: #FFFFFF !important; border: 1px solid {COLOR["border"]} !important;
}}
[data-testid="stSidebar"] button[kind="secondary"] p {{ color: {COLOR["ink"]} !important; }}
[data-testid="stSidebar"] button[kind="primary"] {{
    background-color: {COLOR["ink"]} !important; border: 1px solid {COLOR["ink"]} !important;
}}
[data-testid="stSidebar"] button[kind="primary"] p {{ color: #FFFFFF !important; }}
.stButton button[kind="primary"] p {{ color: #FFFFFF !important; }}
.stButton button[kind="secondary"] p {{ color: {COLOR["ink"]} !important; }}
[data-testid="stWidgetLabel"] p {{ color: {COLOR["ink"]} !important; font-size: 48px !important; }}
[role="radiogroup"] label p {{ color: {COLOR["ink"]} !important; }}
[data-baseweb="select"] * {{ color: {COLOR["ink"]} !important; }}

.display {{
    font-weight: 500; color: {COLOR["ink"]};
    letter-spacing: -0.01em;
}}
.eyebrow {{
    font-family: 'Inter', sans-serif; font-size: 48px; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; color: {COLOR["akzent"]};
    margin-bottom: 6px;
}}

.letterhead {{ border-bottom: 3px solid {COLOR["ink"]}; padding-bottom: 16px; margin-bottom: 22px; }}
.letterhead .titel {{ font-size: 70px; line-height: 1.15; }}
.letterhead .sub {{ color: {COLOR["ink_muted"]}; font-size: 34px; margin-top: 4px; }}

.befund {{
    background: {COLOR["card"]}; border: 1px solid {COLOR["border"]};
    border-left: 5px solid {COLOR["akzent"]}; border-radius: 6px;
    padding: 30px 40px; margin-bottom: 12px; height: 100%;
}}
.befund .nr {{
    font-size: 48px; font-weight: 700;
    color: {COLOR["akzent"]}; letter-spacing: 0.06em; text-transform: uppercase;
}}
.befund .titel {{ font-size: 70px; color: {COLOR["ink"]}; margin: 8px 0 6px 0; }}
.befund .text {{ font-size: 48px; color: {COLOR["ink_muted"]}; line-height: 1.3; }}

.kpi {{
    background: {COLOR["card"]}; border: 1px solid {COLOR["border"]}; border-radius: 10px;
    padding: 12px 20px; margin-bottom: 8px; display: flex; flex-direction: row;
    align-items: center; justify-content: space-between; gap: 16px;
}}
.kpi-left {{ text-align: left; flex: 1; min-width: 0; }}
.kpi .label {{ font-size: 48px; font-weight: 700; color: {COLOR["ink_muted"]}; }}
.kpi .sub {{ font-size: 34px; color: {COLOR["ink_muted"]}; line-height: 1.3; }}
.kpi-value {{
    font-size: 115px; font-weight: 700; line-height: 1.0;
    color: {COLOR["ink"]}; text-align: right; flex-shrink: 0; white-space: nowrap;
}}

.caveat {{
    background: #FBF3E7; border-left: 5px solid {COLOR["akzent"]}; border-radius: 6px;
    padding: 14px 20px; font-size: 42px; color: #7A5A1E; margin: 10px 0;
}}

.badge {{ display: inline-block; padding: 6px 20px; border-radius: 20px; font-size: 48px; font-weight: 600; }}

.fach-card {{ background: {COLOR["card"]}; border: 1px solid {COLOR["border"]}; border-radius: 12px;
              padding: 20px 10px; text-align: center; }}
.fach-icon {{ font-size: 70px; }}
.fach-name {{ font-size: 48px; color: {COLOR["ink_muted"]}; margin: 8px 0; min-height: 40px; }}
.fach-pct  {{ font-size: 115px; font-weight: 600; }}

/* Gleiche Hoehe fuer alle Karten in einer Spalten-Reihe (Befund-Karten, KPIs) */
[data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
[data-testid="column"] {{ display: flex; flex-direction: column; }}
[data-testid="column"] > div {{ height: 100%; }}
[data-testid="column"] > div > div {{ height: 100%; }}
[data-testid="stSidebar"] button {{
    font-size: 42px !important; padding: 11px 14px !important; font-weight: 600 !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size: 48px !important; }}

/* Cluster-/Groessenklasse-Filterknoepfe -- Farbe entspricht IMMER der Karte/Legende */
/* KPI-Zeile: Cluster-Knoepfe auf gleiche Hoehe wie die Kachel "Klinken mit QS-Daten" */
.st-key-kpibtn_ueber, .st-key-kpibtn_im, .st-key-kpibtn_unter {{ height: 100%; }}
.st-key-kpibtn_ueber .stButton, .st-key-kpibtn_im .stButton, .st-key-kpibtn_unter .stButton {{ height: 100%; }}
.st-key-kpibtn_ueber button, .st-key-kpibtn_im button, .st-key-kpibtn_unter button {{
    height: 100%; font-size: 48px !important;
}}

.st-key-filterbtn_ueber button[kind="primary"], .st-key-kpibtn_ueber button[kind="primary"] {{
    background-color: {COLOR["ueber"]} !important; border-color: {COLOR["ueber"]} !important; }}
.st-key-filterbtn_ueber button[kind="primary"] p, .st-key-kpibtn_ueber button[kind="primary"] p {{ color: #FFFFFF !important; }}
.st-key-filterbtn_ueber button[kind="secondary"], .st-key-kpibtn_ueber button[kind="secondary"] {{
    background-color: #FFFFFF !important; border: 2px solid {COLOR["ueber"]} !important; }}
.st-key-filterbtn_ueber button[kind="secondary"] p, .st-key-kpibtn_ueber button[kind="secondary"] p {{ color: {COLOR["ueber"]} !important; }}

.st-key-filterbtn_im button[kind="primary"], .st-key-kpibtn_im button[kind="primary"] {{
    background-color: {COLOR["im"]} !important; border-color: {COLOR["im"]} !important; }}
.st-key-filterbtn_im button[kind="primary"] p, .st-key-kpibtn_im button[kind="primary"] p {{ color: #FFFFFF !important; }}
.st-key-filterbtn_im button[kind="secondary"], .st-key-kpibtn_im button[kind="secondary"] {{
    background-color: #FFFFFF !important; border: 2px solid {COLOR["im"]} !important; }}
.st-key-filterbtn_im button[kind="secondary"] p, .st-key-kpibtn_im button[kind="secondary"] p {{ color: {COLOR["im"]} !important; }}

.st-key-filterbtn_unter button[kind="primary"], .st-key-kpibtn_unter button[kind="primary"] {{
    background-color: {COLOR["unter"]} !important; border-color: {COLOR["unter"]} !important; }}
.st-key-filterbtn_unter button[kind="primary"] p, .st-key-kpibtn_unter button[kind="primary"] p {{ color: #FFFFFF !important; }}
.st-key-filterbtn_unter button[kind="secondary"], .st-key-kpibtn_unter button[kind="secondary"] {{
    background-color: #FFFFFF !important; border: 2px solid {COLOR["unter"]} !important; }}
.st-key-filterbtn_unter button[kind="secondary"] p, .st-key-kpibtn_unter button[kind="secondary"] p {{ color: {COLOR["unter"]} !important; }}

/* Groessenklasse-Knoepfe -- bewusst EINFARBIG (dunkelgrau), Farbe ist nur
   fuer Cluster-Kennzeichnung und Visualisierungen (z.B. Kartenfarbe) reserviert */
.st-key-filterbtn_klein button[kind="primary"],
.st-key-filterbtn_mittel button[kind="primary"],
.st-key-filterbtn_gross button[kind="primary"],
.st-key-filterbtn_sehrgross button[kind="primary"] {{
    background-color: #414B54 !important; border-color: #414B54 !important; }}
.st-key-filterbtn_klein button[kind="primary"] p,
.st-key-filterbtn_mittel button[kind="primary"] p,
.st-key-filterbtn_gross button[kind="primary"] p,
.st-key-filterbtn_sehrgross button[kind="primary"] p {{ color: #FFFFFF !important; }}
.st-key-filterbtn_klein button[kind="secondary"],
.st-key-filterbtn_mittel button[kind="secondary"],
.st-key-filterbtn_gross button[kind="secondary"],
.st-key-filterbtn_sehrgross button[kind="secondary"] {{
    background-color: #FFFFFF !important; border: 1px solid #414B54 !important; }}
.st-key-filterbtn_klein button[kind="secondary"] p,
.st-key-filterbtn_mittel button[kind="secondary"] p,
.st-key-filterbtn_gross button[kind="secondary"] p,
.st-key-filterbtn_sehrgross button[kind="secondary"] p {{ color: #414B54 !important; }}
</style>
"""


# =============================================================================
# DATEN
# =============================================================================

@st.cache_data
def daten_laden() -> pd.DataFrame:
    df = pd.read_parquet(DATEI)
    for col in ["SO.Latitude", "SO.Longitude", "SO.Betten"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["cluster_label"] = df["cluster_label"].fillna("Keine QS-Daten")

    df["groessenklasse"] = pd.qcut(df["SO.Betten"], q=4, labels=GROESSENKLASSEN_ALLE, duplicates="drop").astype(str)
    df.loc[df["SO.Betten"].isna(), "groessenklasse"] = "Keine QS-Daten"
    df.loc[df["cluster_label"] == "Keine QS-Daten", "groessenklasse"] = "Keine QS-Daten"

    for key, (_, col) in FACHBEREICHE.items():
        if col in df.columns:
            m, s = df[col].mean(), df[col].std()
            z = (df[col] - m) / s if s and s > 0 else 0.0
            df[f"{col}_z"] = z
            df[f"{col}_pct"] = z.apply(lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2))) if pd.notna(v) else np.nan)
    return df


@st.cache_data
def fachbereich_statistik(df: pd.DataFrame) -> dict:
    """Populations-Kennzahlen je Fachbereich -- fuer die Hover-Erklaerung der
    Prozentzahlen (Streuung + typische Indikatorenzahl je Fachbereich)."""
    stats = {}
    for key, (label, col) in FACHBEREICHE.items():
        n_ind_col = f"n_indikatoren_{key}"
        stats[key] = {
            "std": df[col].std() if col in df.columns else np.nan,
            "mean": df[col].mean() if col in df.columns else np.nan,
            "n_ind_median": df[n_ind_col].median() if n_ind_col in df.columns else np.nan,
        }
    return stats


@st.cache_data
def groessen_grenzen(df: pd.DataFrame) -> dict:
    """Reale Betten-Grenzen je Groessenklasse (Quartil), fuer die Sidebar-Hilfetexte."""
    betten = df["SO.Betten"].dropna()
    _, bins = pd.qcut(betten, q=4, retbins=True, duplicates="drop")
    labels = GROESSENKLASSEN_ALLE[: len(bins) - 1]
    grenzen = {}
    for i, lab in enumerate(labels):
        lo = int(betten.min()) if i == 0 else int(bins[i]) + 1
        hi = int(bins[i + 1])
        grenzen[lab] = (lo, hi)
    return grenzen


def name_spalte_finden(df: pd.DataFrame) -> str:
    for k in ["SO.Name", "KH.Name"]:
        if k in df.columns:
            return k
    return df.select_dtypes(include="object").columns[0]


@st.cache_data
def elbow_silhouette_berechnen(scores: pd.Series, k_bereich=range(2, 11)) -> pd.DataFrame:
    X = StandardScaler().fit_transform(scores.to_frame())
    zeilen = []
    for k in k_bereich:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        zeilen.append({"k": k, "WCSS": km.inertia_, "Silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(zeilen)


# =============================================================================
# BAUSTEINE
# =============================================================================

def hex_zu_rgba(farbe: str, alpha: float = 0.15) -> str:
    r, g, b = int(farbe[1:3], 16), int(farbe[3:5], 16), int(farbe[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def befund_karte(nr: str, titel: str, text: str) -> None:
    text_attr = text.replace('"', "&quot;")
    st.markdown(f"""
        <div class="befund" title="{text_attr}">
            <div class="nr">Befund {nr}</div>
            <div class="titel">{titel}</div>
            <div class="text" style="font-style:italic;opacity:0.7;">Hinweis: Details beim Überfahren mit der Maus (Hover)</div>
        </div>
    """, unsafe_allow_html=True)


def kpi_karte(label: str, value: str, sub: str = "", hover: str = None) -> None:
    title_attr = f' title="{hover.replace(chr(34), "&quot;")}"' if hover else ""
    st.markdown(f"""
        <div class="kpi"{title_attr}>
            <div class="kpi-left">
                <div class="label">{label}</div>
                <div class="sub">{sub}</div>
            </div>
            <div class="kpi-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def info_karte(label: str, value: str) -> None:
    st.markdown(f"""
        <div class="kpi" style="display:block;">
            <div class="label">{label}</div>
            <div style="font-size:48px;font-weight:700;color:{COLOR['ink']};margin-top:4px;">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def cluster_badge(name: str) -> str:
    farbe = CLUSTER_FARBEN.get(name, "#7f8c8d")
    anzeige = "Keine QS-Daten (Ergebnisqualität)" if name == "Keine QS-Daten" else name
    return (f'<span class="badge" style="background:{hex_zu_rgba(farbe,0.16)};'
            f'color:{farbe};border:1px solid {farbe};">{anzeige}</span>')


def fach_farbe(pct: float) -> str:
    if pd.isna(pct):
        return "#9AA5B1"
    if pct >= 0.60:
        return COLOR["ueber"]
    if pct >= 0.40:
        return COLOR["akzent"]
    return COLOR["unter"]


CLUSTER_HELP = {
    "Über Ø": "Ergebnisqualität über dem Bundesdurchschnitt (nach Fachbereichs-"
              "Normalisierung und Korrektur für kleine Fallzahlen).",
    "Im Ø":   "Ergebnisqualität nahe am Bundesdurchschnitt.",
    "Unter Ø": "Ergebnisqualität unter dem Bundesdurchschnitt. Kein absolutes "
               "Qualitätsurteil — hängt oft mit Klinikgröße/Fallkomplexität "
               "zusammen (siehe Befund 02).",
}
def groesse_help_text(grenzen: dict, label: str) -> str:
    if label not in grenzen:
        return ""
    lo, hi = grenzen[label]
    beschreibung = {
        "Klein": "Unterstes Viertel der Kliniken nach Bettenzahl",
        "Mittel": "Zweites Viertel der Kliniken nach Bettenzahl",
        "Gross": "Drittes Viertel der Kliniken nach Bettenzahl",
        "Sehr gross": "Oberstes Viertel — oft Universitätskliniken/Maximalversorger",
    }
    return f"{beschreibung.get(label, '')}: {lo}–{hi} Betten."


CLUSTER_KEY = {"Über Ø": "ueber", "Im Ø": "im", "Unter Ø": "unter"}
GROESSE_KEY = {"Klein": "klein", "Mittel": "mittel", "Gross": "gross", "Sehr gross": "sehrgross"}


def sidebar_filter(grenzen: dict, cluster_counts: dict) -> tuple:
    if "filter_cluster" not in st.session_state:
        st.session_state.filter_cluster = list(CLUSTER_REIHENFOLGE)
    if "filter_groesse" not in st.session_state:
        st.session_state.filter_groesse = list(GROESSENKLASSEN_ALLE)

    st.sidebar.markdown('<div class="eyebrow">Cluster</div>', unsafe_allow_html=True)
    for c in CLUSTER_REIHENFOLGE:
        aktiv = c in st.session_state.filter_cluster
        label = f"{cluster_counts.get(c, 0):,} · {c}"
        with st.sidebar.container(key=f"filterbtn_{CLUSTER_KEY[c]}"):
            if st.button(label, key=f"c_{c}", use_container_width=True, help=CLUSTER_HELP.get(c),
                          type="primary" if aktiv else "secondary"):
                if aktiv and len(st.session_state.filter_cluster) > 1:
                    st.session_state.filter_cluster.remove(c)
                elif not aktiv:
                    st.session_state.filter_cluster.append(c)
                st.rerun()

    st.sidebar.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="eyebrow">Größenklasse</div>', unsafe_allow_html=True)
    for g in GROESSENKLASSEN_ALLE:
        aktiv = g in st.session_state.filter_groesse
        with st.sidebar.container(key=f"filterbtn_{GROESSE_KEY[g]}"):
            if st.button(g, key=f"g_{g}", use_container_width=True,
                          help=groesse_help_text(grenzen, g),
                          type="primary" if aktiv else "secondary"):
                if aktiv and len(st.session_state.filter_groesse) > 1:
                    st.session_state.filter_groesse.remove(g)
                elif not aktiv:
                    st.session_state.filter_groesse.append(g)
                st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("Filter zurücksetzen", use_container_width=True, type="secondary"):
        st.session_state.filter_cluster = list(CLUSTER_REIHENFOLGE)
        st.session_state.filter_groesse = list(GROESSENKLASSEN_ALLE)
        st.rerun()

    return st.session_state.filter_cluster, st.session_state.filter_groesse


# =============================================================================
# SEITE 1: ÜBERSICHT
# =============================================================================

def seite_uebersicht(df: pd.DataFrame, name_col: str, cluster_filter: list, groesse_filter: list) -> None:
    st.markdown(f"""
        <div class="letterhead">
            <div class="display titel">Qualitätsbefund deutscher Krankenhäuser</div>
            <div class="sub">Datenstand 2023 · G-BA/IQTIG Qualitätsberichte · 2.310 Standorte</div>
        </div>
    """, unsafe_allow_html=True)

    # Basis fuer methodische Kennzahlen: nur nach Groessenklasse gefiltert --
    # diese Kacheln beschreiben den METHODIK-UMFANG (was analysiert wurde),
    # nicht die aktuelle Kartenansicht, daher unabhaengig vom Cluster-Filter.
    basis_df = df[df["groessenklasse"].isin(groesse_filter + ["Keine QS-Daten"])]
    mit_qs_basis = basis_df[basis_df["cluster_label"] != "Keine QS-Daten"]

    # Basis fuer die beiden Durchschnittswerte: VOLLSTAENDIG gefiltert (Cluster
    # UND Groesse) -- diese sollen die aktuelle Ansicht widerspiegeln, nicht
    # den methodischen Gesamtumfang.
    voll_gefiltert = df[
        df["cluster_label"].isin(cluster_filter)
        & df["groessenklasse"].isin(groesse_filter + ["Keine QS-Daten"])
    ]
    n_ind_pro_klinik = voll_gefiltert["n_indikatoren"].mean()
    erfolg_mittel = voll_gefiltert["feat_erfolgsquote"].mean()
    n_bundeslaender = df["SO.Bundesland"].nunique()

    karte_df = df[
        df["cluster_label"].isin(cluster_filter + ["Keine QS-Daten"])
        & df["groessenklasse"].isin(groesse_filter + ["Keine QS-Daten"])
    ].dropna(subset=["SO.Latitude", "SO.Longitude"]).copy()

    col_map, col_side = st.columns([1, 1])
    with col_side:
        befund_karte("01", "Drei Gruppen, empirisch bestätigt",
                     "Silhouette = 0,617 bei k=3 — das globale Maximum im getesteten "
                     "Bereich (k=2…10). Die Dreiteilung ist nicht nur am leichtesten "
                     "zu kommunizieren, sondern auch statistisch am besten begründet.")
        befund_karte("02", "Größe erklärt mehr als „Qualität“",
                     "Große, komplexe Häuser liegen deutlich häufiger „unter Ø“ — "
                     "wahrscheinlich, weil sie schwerere, zugewiesene Fälle behandeln. "
                     "Ein Vergleich gg. Bundesdurchschnitt bildet das nicht vollständig ab.")
        befund_karte("03", "Trägerschaft spielt eine kleinere Rolle als erwartet",
                     "Der Unterschied zwischen privaten und öffentlichen Häusern wird "
                     "nach Kontrolle für Klinikgröße spürbar kleiner — aber „privat” "
                     "bleibt in jeder Größenklasse leicht vorne, der Effekt verschwindet "
                     "nicht vollständig.")

        st.markdown('<div class="eyebrow">Umfang der Analyse</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            kpi_karte("Fachbereiche", f"{len(FACHBEREICHE)}", hover=(
                "Kardiologie & Herzchirurgie, Orthopädie & Chirurgie, Geburtshilfe & "
                "Gynäkologie, Onkologie, Transplantation, Allgemeine Versorgung"))
        with m2:
            struktur_liste = ", ".join(FEATURE_LABELS.values())
            kpi_karte("Struktur-Merkmale", f"{len(FEATURE_LABELS)}", hover=struktur_liste)

        m3, m4 = st.columns(2)
        with m3:
            kpi_karte("Qualitätsindikatoren", f"{QI_INDIKATOREN_GESAMT}", hover=(
                "Hauptindikatoren (QI) aus G-BA/IQTIG-Qualitätsberichten, ohne Teilkennzahlen"))
        with m4:
            kpi_karte("Bundesländer", f"{n_bundeslaender}", "bundesweite Abdeckung")

        m5, m6 = st.columns(2)
        with m5:
            kpi_karte("Ø Indikatoren / Klinik", f"{n_ind_pro_klinik:.0f}",
                      "im aktuellen Filter (Cluster + Größe), Kliniken mit QS-Daten")
        with m6:
            kpi_karte("Ø Erfolgsquote", f"{erfolg_mittel*100:.0f}%",
                      "gewichtetes Mittel im aktuellen Filter (Cluster + Größe)")

    with col_map:
        st.markdown('<div class="eyebrow">Geografische Verteilung</div>', unsafe_allow_html=True)
        einfaerbung = st.radio("Einfärbung:", ["Cluster", "Größenklasse"], horizontal=True,
                               label_visibility="collapsed")
        karte_df["hover"] = ("<b>" + karte_df[name_col].fillna("Unbekannt") + "</b><br>Cluster: " +
                             karte_df["cluster_label"] + "<br>Betten: " +
                             karte_df["SO.Betten"].fillna(0).astype(int).astype(str))

        fig = go.Figure()
        gruppen = CLUSTER_FARBEN if einfaerbung == "Cluster" else {**GROESSEN_FARBEN, "Keine QS-Daten": "#9AA5B1"}
        spalte = "cluster_label" if einfaerbung == "Cluster" else "groessenklasse"
        for name, farbe in gruppen.items():
            teil = karte_df[karte_df[spalte] == name]
            if len(teil) == 0:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=teil["SO.Latitude"], lon=teil["SO.Longitude"], mode="markers",
                marker=dict(size=25, color=farbe, opacity=0.75), text=teil["hover"],
                hoverinfo="text", name=f"{name} ({len(teil)})",
            ))
        fig.update_layout(
            mapbox=dict(style="carto-positron", center=DE_CENTER, zoom=DE_ZOOM,
                       bounds=dict(west=4.8, east=16.2, south=46.3, north=56.0)),
            height=2750, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)",
                       font=dict(size=27)),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(size=27),
        )
        auswahl = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="uebk")
        if auswahl and auswahl.get("selection", {}).get("points"):
            pt = auswahl["selection"]["points"][0]
            trace_name = list(gruppen.keys())[pt.get("trace_index", 0)]
            teil = karte_df[karte_df[spalte] == trace_name]
            idx = pt.get("point_index", 0)
            if idx < len(teil):
                kh_name = teil.iloc[idx][name_col]
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{kh_name}** {cluster_badge(teil.iloc[idx]['cluster_label'])}",
                                unsafe_allow_html=True)
                with col_b:
                    if st.button("Profil öffnen", type="primary", use_container_width=True):
                        st.session_state.profil_kh = kh_name
                        st.session_state.seite = "Klinik-Profil"
                        st.rerun()

    with col_side:
        kpi_karte("Klinken mit QS-Daten", f"{len(mit_qs_basis):,}",
                  f"von {len(basis_df):,} gesamt erfasst — nur diese werden geclustert")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Top / Flop 5 Kliniken</div>', unsafe_allow_html=True)
        bewertet = karte_df[karte_df["feat_erfolgsquote"].notna()]
        top5 = bewertet.nlargest(5, "feat_erfolgsquote")
        flop5 = bewertet.nsmallest(5, "feat_erfolgsquote")
        tf = pd.concat([flop5, top5]).drop_duplicates(subset=[name_col])
        tf = tf.sort_values("feat_erfolgsquote").reset_index(drop=True)
        tf["kurzname"] = tf[name_col].str.slice(0, 28)

        fig_tf = go.Figure()
        for cl in CLUSTER_REIHENFOLGE:
            teil = tf[tf["cluster_label"] == cl]
            if len(teil) == 0:
                continue
            fig_tf.add_trace(go.Bar(
                x=teil["feat_erfolgsquote"], y=teil["kurzname"], orientation="h",
                marker=dict(color=CLUSTER_FARBEN[cl]), name=cl,
                text=teil["feat_erfolgsquote"].apply(lambda v: f"{v*100:.0f}%"),
                textfont=dict(size=44, color=COLOR["ink"]), textposition="outside",
                cliponaxis=False,
                customdata=teil[name_col],
            ))
        fig_tf.update_layout(height=750, margin=dict(l=10, r=70, t=10, b=0),
                            xaxis_title=None, yaxis_title=None, showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(size=42), xaxis=dict(range=[0, 1]), bargap=0.5,
                            yaxis=dict(categoryorder="array", categoryarray=tf["kurzname"].tolist()))
        auswahl_tf = st.plotly_chart(fig_tf, use_container_width=True, on_select="rerun", key="topflop")
        if auswahl_tf and auswahl_tf.get("selection", {}).get("points"):
            pt = auswahl_tf["selection"]["points"][0]
            kh_name_tf = pt.get("customdata")
            if kh_name_tf:
                st.session_state.profil_kh = kh_name_tf[0] if isinstance(kh_name_tf, list) else kh_name_tf
                st.session_state.seite = "Klinik-Profil"
                st.rerun()
        st.caption("Klick auf einen Balken öffnet das Klinik-Profil.")



# =============================================================================
# SEITE 2: CLUSTER-ANALYSE
# =============================================================================

def seite_cluster_analyse(df: pd.DataFrame, cluster_filter: list, groesse_filter: list) -> None:
    st.markdown('<div class="display" style="font-size:70px;">Cluster-Analyse</div>',
               unsafe_allow_html=True)
    st.caption("Methodischer Nachweis hinter den drei Befunden der Übersicht.")

    mit_qs = df[
        (df["cluster_label"] != "Keine QS-Daten")
        & df["cluster_label"].isin(cluster_filter)
        & df["groessenklasse"].isin(groesse_filter)
    ].copy()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="eyebrow">Zu Befund 01 — Warum k=3?</div>', unsafe_allow_html=True)
        es = elbow_silhouette_berechnen(mit_qs["feat_erfolgsquote"])
        fig_es = go.Figure()
        fig_es.add_trace(go.Scatter(x=es["k"], y=es["Silhouette"], mode="lines+markers",
                                    name="Silhouette", line=dict(color=COLOR["akzent"], width=4),
                                    marker=dict(size=12)))
        fig_es.add_vline(x=3, line_dash="dash", line_color=COLOR["ink_muted"],
                         annotation_text="k=3 gewählt", annotation_font=dict(size=46))
        fig_es.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Anzahl Cluster (k)", yaxis_title="Silhouette",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(size=42))
        st.plotly_chart(fig_es, use_container_width=True)

    with col_b:
        st.markdown('<div class="eyebrow">Zu Befund 03 — Trägerschaft, kontrolliert nach Größe</div>',
                   unsafe_allow_html=True)
        mit_qs2 = mit_qs.copy()
        mit_qs2["betten_q"] = pd.qcut(mit_qs2["SO.Betten"], q=4, duplicates="drop")
        kontroll = mit_qs2.groupby(["betten_q", "KH.Träger.Art"])["cluster_label"].apply(
            lambda x: (x == "Über Ø").mean() * 100
        ).reset_index(name="Anteil Über Ø (%)")
        kontroll["betten_q"] = kontroll["betten_q"].astype(str)
        fig_tr = px.line(kontroll, x="betten_q", y="Anteil Über Ø (%)", color="KH.Träger.Art",
                         markers=True, color_discrete_sequence=NEBEN_FARBEN)
        fig_tr.update_traces(line=dict(width=4), marker=dict(size=11))
        fig_tr.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Größen-Quartil",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(size=38), legend=dict(font=dict(size=38)))
        st.plotly_chart(fig_tr, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Zu Befund 02 — Größe & Struktur</div>', unsafe_allow_html=True)
    st.caption('„(Breite)” = Anzahl angebotener Maßnahmen von mehreren möglichen — kein Qualitätsmaß. '
              'Grün = höchster Wert der drei Cluster, Rot = niedrigster.')

    zeilen_f = []
    for col, label in FEATURE_LABELS.items():
        werte = {cl: mit_qs[mit_qs["cluster_label"] == cl][col].mean() for cl in CLUSTER_REIHENFOLGE}
        bester = max(werte, key=werte.get)
        schlechtester = min(werte, key=werte.get)
        zellen = ""
        for cl in CLUSTER_REIHENFOLGE:
            if werte[bester] == werte[schlechtester]:
                farbe = COLOR["ink_muted"]
            elif cl == bester:
                farbe = COLOR["ueber"]
            elif cl == schlechtester:
                farbe = COLOR["unter"]
            else:
                farbe = COLOR["ink_muted"]
            zellen += (f'<td style="padding:10px 8px;text-align:right;font-size:34px;'
                      f'font-weight:700;color:{farbe};">{werte[cl]*100:.0f}%</td>')
        zeilen_f.append(
            f'<tr style="border-bottom:1px solid {COLOR["border"]};">'
            f'<td style="padding:10px 8px;font-size:42px;color:{COLOR["ink"]};">{label}</td>'
            f'{zellen}</tr>'
        )
    kopf_f = "".join(
        f'<th style="text-align:right;padding:10px 8px;font-size:42px;font-weight:700;'
        f'color:{CLUSTER_FARBEN[cl]};">{cl}</th>' for cl in CLUSTER_REIHENFOLGE
    )
    tabelle_f = (
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr style="border-bottom:2px solid {COLOR["ink"]};">'
        f'<th style="text-align:left;padding:10px 8px;font-size:42px;font-weight:700;'
        f'color:{COLOR["ink_muted"]};">Merkmal</th>{kopf_f}</tr>{"".join(zeilen_f)}</table>'
    )
    st.markdown(tabelle_f, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    gewaehltes_feat = st.selectbox(
        "Detaillierte Verteilung anzeigen für:", list(FEATURE_LABELS.values()),
        key="feat_detail_auswahl",
    )
    st.caption("Zahl = Median (die Tabelle oben zeigt den Mittelwert — beide können leicht abweichen).")
    feat_key = next(k for k, v in FEATURE_LABELS.items() if v == gewaehltes_feat)
    fig_detail = px.box(mit_qs, x="cluster_label", y=feat_key, color="cluster_label",
                       color_discrete_map=CLUSTER_FARBEN, category_orders={"cluster_label": CLUSTER_REIHENFOLGE})
    for cl in CLUSTER_REIHENFOLGE:
        median_feat = mit_qs[mit_qs["cluster_label"] == cl][feat_key].median()
        if pd.notna(median_feat):
            fig_detail.add_annotation(
                x=cl, y=median_feat, text=f"{median_feat*100:.0f}%", showarrow=False,
                yshift=16, font=dict(size=38, color="#FFFFFF", family="Inter"),
            )
    fig_detail.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                            xaxis_title=None, yaxis_title=gewaehltes_feat,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(size=40))
    st.plotly_chart(fig_detail, use_container_width=True)

    st.markdown("**Klinikgröße (Betten) je Cluster**")
    st.caption("Achse auf 1.000 Betten begrenzt — einzelne Ausreißer darüber (Großkliniken) "
              "sind nicht vollständig dargestellt. Zahl = Median.")
    fig_betten = px.box(mit_qs, x="cluster_label", y="SO.Betten", color="cluster_label",
                        color_discrete_map=CLUSTER_FARBEN, category_orders={"cluster_label": CLUSTER_REIHENFOLGE})
    for cl in CLUSTER_REIHENFOLGE:
        median_betten = mit_qs[mit_qs["cluster_label"] == cl]["SO.Betten"].median()
        fig_betten.add_annotation(
            x=cl, y=median_betten, text=f"{median_betten:.0f}", showarrow=False,
            yshift=18, font=dict(size=42, color="#FFFFFF", family="Inter"),
        )
    fig_betten.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                            xaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(range=[0, 1000]),
                            font=dict(size=50))
    st.plotly_chart(fig_betten, use_container_width=True)

    # -- Homogenitaet: LIVE berechnet statt statischem Text ---------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Wie homogen sind die Cluster wirklich?</div>', unsafe_allow_html=True)
    st.caption("Live berechnet aus den aktuell gefilterten Daten — kein vorab festgelegter Wert.")

    col_cv, col_fach = st.columns([1, 2])
    with col_cv:
        st.markdown("**Streuung innerhalb des Clusters (Variationskoeffizient)**")
        cv = mit_qs.groupby("cluster_label")["feat_erfolgsquote"].apply(
            lambda x: x.std() / x.mean() if x.mean() else np.nan
        ).reindex(CLUSTER_REIHENFOLGE)
        fig_cv = go.Figure(go.Bar(
            x=cv.index, y=cv.values, marker_color=[CLUSTER_FARBEN[c] for c in cv.index],
            text=[f"{v:.2f}" for v in cv.values], textposition="inside",
            textfont=dict(color="#FFFFFF", size=50),
        ))
        fig_cv.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(size=35), yaxis_title="Std / Mittelwert")
        st.plotly_chart(fig_cv, use_container_width=True)

    with col_fach:
        st.markdown("**Fachbereichs-Beteiligung je Cluster (%)**")
        rows_fach = []
        for cl in CLUSTER_REIHENFOLGE:
            sub = mit_qs[mit_qs["cluster_label"] == cl]
            for key, (label, col) in FACHBEREICHE.items():
                anteil = sub[col].notna().mean() * 100
                rows_fach.append({"Cluster": cl, "Fachbereich": label,
                                  "Beteiligung": anteil, "Text": f"{anteil:.0f}%"})
        fig_fach = px.bar(pd.DataFrame(rows_fach), x="Fachbereich", y="Beteiligung",
                          color="Cluster", barmode="group", color_discrete_map=CLUSTER_FARBEN,
                          category_orders={"Cluster": CLUSTER_REIHENFOLGE}, text="Text")
        fig_fach.update_traces(textposition="outside", textfont=dict(color=COLOR["ink"], size=44), cliponaxis=False)
        fig_fach.update_layout(height=360, margin=dict(l=0, r=0, t=90, b=0), legend_title=None,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(size=46), xaxis=dict(tickfont=dict(size=42)),
                              yaxis=dict(range=[0, 115]))
        st.plotly_chart(fig_fach, use_container_width=True)

    st.markdown("""
        <div class="caveat"><b>Homogenität:</b> „Unter Ø” zeigt einen deutlich höheren
        Variationskoeffizienten als „Über Ø” und eine abweichende Fachbereichs-Beteiligung
        (z.B. überproportional viele Transplantationszentren) — ein Marker für sehr große,
        umfassende Häuser, nicht für generell schwächere Versorgung. Details je Fachbereich:
        siehe Klinik-Profil.</div>
    """, unsafe_allow_html=True)


# =============================================================================
# SEITE 3: KLINIK-PROFIL
# =============================================================================

def seite_profil(df: pd.DataFrame, name_col: str) -> None:
    st.markdown('<div class="display" style="font-size:70px;">Klinik-Profil</div>',
               unsafe_allow_html=True)

    alle_namen = sorted(df[name_col].dropna().unique().tolist())
    vorauswahl = 0
    if "profil_kh" in st.session_state and st.session_state.profil_kh in alle_namen:
        vorauswahl = alle_namen.index(st.session_state.profil_kh)
    kh_name = st.selectbox("Krankenhaus suchen", alle_namen, index=vorauswahl)
    zeile = df[df[name_col] == kh_name].iloc[0]
    cluster = zeile.get("cluster_label", "Keine QS-Daten")

    col_info, col_gauge = st.columns([2, 1])
    with col_info:
        st.markdown(cluster_badge(cluster), unsafe_allow_html=True)

        n_ind_kh = zeile.get("n_indikatoren", None)
        n_fach_kh = zeile.get("n_fachgruppen", None)

        i1, i2, i3 = st.columns(3)
        with i1: info_karte("Bundesland", zeile.get("SO.Bundesland", "-"))
        with i2: info_karte("Betten", f"{int(zeile['SO.Betten']):,}" if pd.notna(zeile["SO.Betten"]) else "-")
        with i3: info_karte("Träger", zeile.get("KH.Träger.Art", "-"))

        i4, i5 = st.columns(2)
        with i4: info_karte("Größenklasse", zeile.get("groessenklasse", "-"))
        with i5:
            info_karte("Datengrundlage", (
                f"{int(n_ind_kh)} Ind. / {int(n_fach_kh)} von 6 Fachb."
                if pd.notna(n_ind_kh) else "-"
            ))

        ist_gross_unter = zeile.get("groessenklasse") == "Sehr gross" and cluster == "Unter Ø"
        ist_transplant = zeile.get("hat_transplantation", 0) == 1
        if ist_transplant or ist_gross_unter:
            grund = "Führt Transplantationen durch" if ist_transplant else "Zählt zu den größten 25% der Kliniken"
            st.markdown(f"""
                <div class="caveat">{grund} — siehe <b>Befund 02</b> in der Cluster-Analyse:
                diese Häuser sind überproportional groß/komplex, ein „Unter Ø”-Label sollte
                hier mit Vorsicht gelesen werden.</div>
            """, unsafe_allow_html=True)

    with col_gauge:
        erfolg = zeile.get("feat_erfolgsquote", None)
        if pd.notna(erfolg):
            farbe = CLUSTER_FARBEN.get(cluster, COLOR["im"])
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(erfolg * 100, 1),
                number=dict(suffix="%", font=dict(size=60, color=farbe)),
                gauge=dict(axis=dict(range=[0, 100], tickfont=dict(size=34)),
                          bar=dict(color=farbe, thickness=0.25),
                          steps=[dict(range=[0, 40], color=hex_zu_rgba(COLOR["unter"], 0.12)),
                                 dict(range=[40, 60], color=hex_zu_rgba(COLOR["im"], 0.12)),
                                 dict(range=[60, 100], color=hex_zu_rgba(COLOR["ueber"], 0.12))]),
            ))
            fig_g.update_layout(height=620, margin=dict(l=35, r=35, t=20, b=35),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(size=50))
            st.markdown(
                f"<div style='text-align:center;font-size:42px;font-weight:600;"
                f"color:{COLOR['ink_muted']};'>Position ggü. Bundesdurchschnitt</div>",
                unsafe_allow_html=True
            )
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.warning("Keine QS-Daten für diese Klinik")

    st.markdown('<div class="eyebrow">Qualität je Fachbereich</div>', unsafe_allow_html=True)
    st.caption("Position relativ zu den echten Konkurrenten IM SELBEN Fachbereich — "
              "nicht zum Gesamtdurchschnitt aller Kliniken. Details (Streuung, Datenbasis) beim Überfahren mit der Maus.")

    fach_stats = fachbereich_statistik(df)
    median_std_alle = np.median([s["std"] for s in fach_stats.values() if pd.notna(s["std"])])

    cols = st.columns(6)
    for i, (key, (label, col)) in enumerate(FACHBEREICHE.items()):
        pct = zeile.get(f"{col}_pct", np.nan)
        with cols[i]:
            if pd.notna(pct):
                farbe = fach_farbe(pct)
                z_wert = zeile.get(f"{col}_z", np.nan)
                n_ind_kh = zeile.get(f"n_indikatoren_{key}", np.nan)
                st_info = fach_stats.get(key, {})
                streuung = "eng — kleine Unterschiede wirken hier extremer" if st_info.get("std", 1) < median_std_alle else "breit gestreut"
                hover_txt = (
                    f"Position: {z_wert:+.2f} Standardabweichungen vom Fachbereichs-Durchschnitt. "
                    f"Streuung dieser Gruppe ist {streuung}. "
                    f"Diese Klinik: {int(n_ind_kh) if pd.notna(n_ind_kh) else '?'} Indikatoren in "
                    f"diesem Fachbereich (typisch: {st_info.get('n_ind_median', float('nan')):.0f})."
                ) if pd.notna(z_wert) else ""
                hover_txt = hover_txt.replace('"', "&quot;")
                st.markdown(f"""
                    <div class="fach-card" title="{hover_txt}">
                    <div class="fach-name">{label}</div>
                    <div class="fach-pct" style="color:{farbe};">{pct*100:.0f}%</div></div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="fach-card" style="opacity:0.4;">
                    <div class="fach-name">{label}</div>
                    <div class="fach-pct" style="color:#9AA5B1;">–</div></div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Struktur im Vergleich</div>', unsafe_allow_html=True)
    st.caption("Grün = über dem Gesamtdurchschnitt, Rot = darunter. „(Breite)” = Anzahl "
              "angebotener Maßnahmen, kein Qualitätsmaß.")
    maske_cl = df["cluster_label"] == cluster

    zeilen = []
    for key, label in FEATURE_LABELS.items():
        kh_wert = zeile.get(key, np.nan)
        cl_wert = df[maske_cl][key].mean()
        ges_wert = df[key].mean()
        if pd.isna(kh_wert):
            farbe_kh, kh_txt = COLOR["ink_muted"], "–"
        else:
            diff = kh_wert - ges_wert
            farbe_kh = COLOR["ueber"] if diff > 0.02 else (COLOR["unter"] if diff < -0.02 else COLOR["ink_muted"])
            kh_txt = f"{kh_wert*100:.0f}%"
        cl_txt = f"{cl_wert*100:.0f}%" if pd.notna(cl_wert) else "–"
        ges_txt = f"{ges_wert*100:.0f}%" if pd.notna(ges_wert) else "–"
        zeilen.append(
            f'<tr style="border-bottom:1px solid {COLOR["border"]};">'
            f'<td style="padding:12px 8px;font-size:42px;color:{COLOR["ink"]};">{label}</td>'
            f'<td style="padding:12px 8px;text-align:right;font-size:48px;font-weight:700;color:{farbe_kh};">{kh_txt}</td>'
            f'<td style="padding:12px 8px;text-align:right;font-size:48px;font-weight:700;color:{COLOR["ink_muted"]};">{cl_txt}</td>'
            f'<td style="padding:12px 8px;text-align:right;font-size:48px;font-weight:700;color:{COLOR["ink_muted"]};">{ges_txt}</td>'
            f'</tr>'
        )
    zeilen_html = "".join(zeilen)

    tabelle_html = (
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr style="border-bottom:2px solid {COLOR["ink"]};">'
        f'<th style="text-align:left;padding:12px 8px;font-size:48px;font-weight:700;color:{COLOR["ink_muted"]};">Merkmal</th>'
        f'<th style="text-align:right;padding:12px 8px;font-size:48px;font-weight:700;color:{COLOR["ink"]};">Diese Klinik</th>'
        f'<th style="text-align:right;padding:12px 8px;font-size:48px;font-weight:700;color:{COLOR["ink_muted"]};">Cluster Ø</th>'
        f'<th style="text-align:right;padding:12px 8px;font-size:48px;font-weight:700;color:{COLOR["ink_muted"]};">Gesamt Ø</th>'
        f'</tr>{zeilen_html}</table>'
    )
    st.markdown(tabelle_html, unsafe_allow_html=True)


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="Qualitätsbefund Krankenhäuser", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    if not DATEI.exists():
        st.error(f"Datei nicht gefunden: {DATEI}")
        return

    df = daten_laden()
    name_col = name_spalte_finden(df)

    for key, val in [("seite", "Übersicht"), ("profil_kh", None)]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.sidebar.markdown('<div class="display" style="font-size:48px;padding:8px 0;">Navigation</div>',
                       unsafe_allow_html=True)
    seiten = ["Übersicht", "Cluster-Analyse", "Klinik-Profil"]
    seite = st.sidebar.radio("", seiten, index=seiten.index(st.session_state.seite),
                            label_visibility="collapsed", key="nav")
    st.session_state.seite = seite
    st.sidebar.markdown("---")

    if seite != "Klinik-Profil":
        cluster_counts = df[df["cluster_label"] != "Keine QS-Daten"]["cluster_label"].value_counts().to_dict()
        cluster_filter, groesse_filter = sidebar_filter(groessen_grenzen(df), cluster_counts)
    else:
        cluster_filter, groesse_filter = CLUSTER_REIHENFOLGE, GROESSENKLASSEN_ALLE
        st.sidebar.caption("Filter nicht aktiv auf dieser Seite.")

    if seite == "Übersicht":
        seite_uebersicht(df, name_col, cluster_filter, groesse_filter)
    elif seite == "Cluster-Analyse":
        seite_cluster_analyse(df, cluster_filter, groesse_filter)
    elif seite == "Klinik-Profil":
        seite_profil(df, name_col)


if __name__ == "__main__":
    main()
