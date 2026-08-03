"""
02_transform.py
===============
Feature Engineering fuer drei Qualitaetsbereiche.

Einheitliche Scoring-Logik:
    Strukturelle Features = Anzahl vorhandener Items / Maximum moeglich
    Beispiel: Klinik hat 12 von 25 Geraeten -> 12/25 = 0.48

Maximale Nenner (aus Rohdaten ermittelt):
    AA.Key  -- 25 Geraetetypen
    HB.Key  -- 19 Ausbildungstypen
    HM.Key  --  6 Hygienemassnahmen
    IF.Key  --  3 Fehlermeldesysteme
    EF.Key  -- 13 CIRS-Systeme
    AMTS    -- 13 Massnahmen
    BM      --  8 Ja/Nein-Kriterien
    RM.Key  -- 17 Risikomanagement-Instrumente

Erfolgsquote-Logik:
    Nur QI-Indikatoren (199 Hauptindikatoren, keine Teilkennzahlen)
    Richtung aus QSQI.Operator (<= oder >=)
    Pro Indikator drei Vergleiche (je 0 oder 1 Punkt):
        1. Ergebnis vs. Bundesdurchschnitt
        2. Konfidenzintervall guenstig
        3. Ergebnis vs. Referenzwert (falls vorhanden)
    Gewichtung nach klinischer Schwere der Kategorie

Drei Qualitaetsbereiche mit Gewichtung:
    A. Ergebnisqualitaet  50%
    B. Strukturqualitaet  30%
    C. Prozessqualitaet   20%

Fehlende Werte:
    Organisatorische Features -> 0 (kein Bericht = kein Programm)
    Klinische Features        -> NaN (nur KH mit QS-Daten werden geclustert)

Ausfuehren:
    python 02_transform.py
"""

import pandas as pd
import numpy as np
import math
import re
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

# Maximale Anzahl moeglicher Items pro Feature
MAX_WERTE = {
    "ausstattung": 25,
    "ausbildung":  19,
    "hygiene":      6,
    "if_score":     3,
    "fehler":      13,
    "amts":        13,
    "bm":           8,
    "rm":          17,
}



# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def zu_zahl(spalte: pd.Series) -> pd.Series:
    """
    Wandelt eine Spalte sicher in Zahlen um.
    Behandelt:
        - Deutsches Komma-Format: 1,23 -> 1.23
        - Datumswerte: 01.04.2004 -> NaN (Formatfehler in Quelldaten)
        - Texte und NaN -> NaN
    """
    bereinigt = spalte.astype(str).str.strip().str.replace(",", ".", regex=False)

    # Datumsmuster erkennen und als ungueltig markieren
    datum = bereinigt.str.match(r"^\d{1,2}\.\d{2}\.\d{4}$|^\d{4}-\d{2}-\d{2}$")
    bereinigt[datum] = np.nan

    return pd.to_numeric(bereinigt, errors="coerce")


def anteil(anzahl: pd.Series, maximum: int) -> pd.Series:
    """
    Berechnet Anteil: Anzahl / Maximum, immer zwischen 0 und 1.
    Werte ueber 1 werden gedeckelt (Datenfehler abfangen).
    """
    return (anzahl / maximum).clip(upper=1.0)


def ist_anonymisiert(spalte: pd.Series) -> pd.Series:
    """
    Erkennt anonymisierte Fallzahlen (<=3, le3 usw.).
    Diese Indikatoren sind statistisch nicht verwertbar.
    """
    return spalte.astype(str).str.contains(
        r"≤|<=|le\s*3|<\s*3", regex=True, na=False
    )


def ja_nein_zu_zahl(spalte: pd.Series) -> pd.Series:
    """Wandelt Ja/Nein und 1/0 in numerische Werte um. Unbekannt -> 0."""
    s = spalte.astype(str).str.strip().str.lower()
    return s.map({"ja": 1.0, "1": 1.0, "nein": 0.0, "0": 0.0}).fillna(0.0)


# =============================================================================
# A. ERGEBNISQUALITAET
# =============================================================================

def erfolgsquote_berechnen(df_qsqi: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet die gewichtete Erfolgsquote aus QS.Qualitaetsindikator.csv.

    Schritt 1 -- Filter:
        Nur QI-Indikatoren (Hauptindikatoren, keine Teilkennzahlen)
        Keine anonymisierten Fallzahlen (<=3)
        Nur Zeilen mit Ergebnis und Bundesdurchschnitt

    Schritt 2 -- Drei Vergleiche pro Indikator:
        Vergleich 1: Ergebnis vs. Bundesdurchschnitt
        Vergleich 2: Konfidenzintervall des KH guenstig
        Vergleich 3: Ergebnis vs. Referenzwert (nur wenn vorhanden)
        Richtung aus QSQI.Operator: >= bedeutet hoeher=besser

    Schritt 3 -- Gewichtung nach Kategorie:
        Mortalitaet: 3.0  Komplikationen: 2.5  Infektionen: 2.0
        Ergebnis: 1.5     Wiederaufnahme: 1.0  Prozess: 1.0
        Wartezeit: 0.5

    Schritt 4 -- Erfolgsquote pro KH:
        Summe(erreichte Punkte x Gewicht) / Summe(moegliche Punkte x Gewicht)
    """
    df = df_qsqi.copy()

    # -- Filter 1: Nur Hauptindikatoren ---------------------------------------
    df = df[df["QSQI.ArtDesWertes"] == "QI"]

    # -- Filter 2: Anonymisierte Fallzahlen ausschliessen ---------------------
    df = df[~ist_anonymisiert(df["QSQI.FallzahlGrundgesamtheit"])]

    # -- Zahlenspalten bereinigen ----------------------------------------------
    # HINWEIS: Kommt die bereinigte Datei aus 0_cleanup.py, wurden dort
    # zwei moegliche Datenfehler bereits geprueft und -- falls vorhanden --
    # behoben: Excel-Datum-Korruption in QSQI.Ergebnis (rekonstruiert ueber
    # Beobachtete/Grundgesamtheit, falls moeglich) und ein x100-Skalenfehler
    # in QSQI.Bundesdurchschnitt bei bestimmten Indikatoren (56102-56116).
    # Ob und wie oft diese Faelle im jeweiligen Datenstand tatsaechlich
    # auftreten, zeigt das Log von 0_cleanup.py beim Ausfuehren.
    # Liegt die bereinigte Datei vor, werden die Werte hier direkt uebernommen
    # statt erneut ueber zu_zahl() zu laufen (das wuerde die Reparatur wieder
    # verwerfen, da zu_zahl() Datumsmuster grundsaetzlich als NaN behandelt).
    if "QSQI.Ergebnis_original" in df.columns:
        df["ergebnis_num"] = pd.to_numeric(df["QSQI.Ergebnis"], errors="coerce")
        df["bund_num"]     = pd.to_numeric(df["QSQI.Bundesdurchschnitt"], errors="coerce")
    else:
        df["ergebnis_num"] = zu_zahl(df["QSQI.Ergebnis"])
        df["bund_num"]     = zu_zahl(df["QSQI.Bundesdurchschnitt"])
    df["referenz_num"] = zu_zahl(df["QSQI.Referenzwert"])

    # Konfidenzintervall des Krankenhauses aufteilen (Format: "1.2 - 3.4")
    kh_vb = df["QSQI.KHVertrauensbereich"].astype(str).str.replace(",", ".", regex=False)
    df["kh_vb_low"]  = zu_zahl(kh_vb.str.split(r"\s*-\s*").str[0])
    df["kh_vb_high"] = zu_zahl(kh_vb.str.split(r"\s*-\s*").str[-1])

    # -- Filter 3: Nur Zeilen mit verwertbarem Ergebnis -----------------------
    df = df.dropna(subset=["ergebnis_num", "bund_num"])

    # -- Richtung aus Operator ------------------------------------------------
    # >= bedeutet hoeher ist besser (z.B. Ueberlebensrate)
    # <= bedeutet niedriger ist besser (z.B. Sterblichkeit)
    # HINWEIS: Fuer 26.5% der QI-Zeilen ist QSQI.Operator strukturell leer
    # (bestimmte Indikatoren haben nie einen Operator in den Quelldaten).
    # QSQI.Operator_bestimmt (aus 0_cleanup.py) verwendet fuer diese
    # Faelle einen explizit recherchierten Override (z.B. Patientenbefragungs-
    # Indikatoren 56102-56118, wo hoeher=besser gilt, per IQTIG-Bundesauswertung
    # bestaetigt) und sonst den Default '<=' (passend fuer die meisten
    # unbelegten Indikatoren: Mortalitaet, Komplikationen, Fehlfunktion).
    if "QSQI.Operator_bestimmt" in df.columns:
        df["hoeher_besser"] = df["QSQI.Operator_bestimmt"] == ">="
    else:
        df["hoeher_besser"] = df["QSQI.Operator"].astype(str).str.strip() == ">="

    # -- Vergleich 1: Ergebnis vs. Bundesdurchschnitt -------------------------
    df["vgl_ergebnis"] = np.where(
        df["hoeher_besser"],
        (df["ergebnis_num"] >= df["bund_num"]).astype(float),
        (df["ergebnis_num"] <= df["bund_num"]).astype(float),
    )

    # -- Vergleich 2: Konfidenzintervall --------------------------------------
    # Guenstig wenn KH-KI komplett besser als Bundesdurchschnitt liegt
    df["vgl_konfidenz"] = np.nan
    hat_vb = df["kh_vb_low"].notna() & df["kh_vb_high"].notna()

    # hoeher besser: untere Grenze des KH-KI >= Bundesdurchschnitt
    maske_h = hat_vb & df["hoeher_besser"]
    df.loc[maske_h, "vgl_konfidenz"] = (
        df.loc[maske_h, "kh_vb_low"] >= df.loc[maske_h, "bund_num"]
    ).astype(float)

    # niedriger besser: obere Grenze des KH-KI <= Bundesdurchschnitt
    maske_n = hat_vb & ~df["hoeher_besser"]
    df.loc[maske_n, "vgl_konfidenz"] = (
        df.loc[maske_n, "kh_vb_high"] <= df.loc[maske_n, "bund_num"]
    ).astype(float)

    # -- Vergleich 3: Ergebnis vs. Referenzwert -------------------------------
    df["vgl_referenz"] = np.nan
    hat_ref = df["referenz_num"].notna()

    maske_rh = hat_ref & df["hoeher_besser"]
    df.loc[maske_rh, "vgl_referenz"] = (
        df.loc[maske_rh, "ergebnis_num"] >= df.loc[maske_rh, "referenz_num"]
    ).astype(float)

    maske_rn = hat_ref & ~df["hoeher_besser"]
    df.loc[maske_rn, "vgl_referenz"] = (
        df.loc[maske_rn, "ergebnis_num"] <= df.loc[maske_rn, "referenz_num"]
    ).astype(float)

    # -- Punkte berechnen -----------------------------------------------------
    df["punkte_moeglich"] = (
        1                        # Vergleich 1 immer moeglich
        + hat_vb.astype(int)     # Vergleich 2 nur wenn KI vorhanden
        + hat_ref.astype(int)    # Vergleich 3 nur wenn Referenzwert vorhanden
    ).astype(float)

    df["punkte_erreicht"] = (
        df["vgl_ergebnis"].fillna(0)
        + df["vgl_konfidenz"].fillna(0)
        + df["vgl_referenz"].fillna(0)
    )

    # HINWEIS: Es gab hier frueher eine "Kategorie-Gewichtung" nach klinischer
    # Schwere (KATEGORIE_GEWICHTE), die auf QSQI.Leistungsbereich gemappt wurde.
    # Das war toter Code -- QSQI.Leistungsbereich enthaelt Verfahrens-/Prozedur-
    # Codes (z.B. "PCI", "GYN-OP"), keine Schwere-Kategorien wie "Mortalitaet" --
    # der Map traf nie, jede Zeile bekam durch .fillna(1.0) stillschweigend
    # Gewicht=1.0. Entfernt, um irrefuehrenden toten Code nicht zu behalten.
    # Die "_gew_"-Namen bleiben (identisch zu den ungewichteten Werten) nur
    # damit der Rest der Funktion unveraendert bleibt.
    df["punkte_gew_erreicht"] = df["punkte_erreicht"]
    df["punkte_gew_moeglich"] = df["punkte_moeglich"]

    # -- Aggregation pro Krankenhaus ------------------------------------------
    agg = df.groupby("SO.QBID").agg(
        punkte_gew_sum  = ("punkte_gew_erreicht", "sum"),
        punkte_gew_max  = ("punkte_gew_moeglich", "sum"),
        n_indikatoren   = ("QSQI.Indikator",      "count"),
    ).reset_index()

    agg["feat_erfolgsquote_roh"] = np.where(
        agg["punkte_gew_max"] > 0,
        agg["punkte_gew_sum"] / agg["punkte_gew_max"],
        np.nan,
    )

    # -- Score pro Versorgungsbereich (Gruppe), MIT gruppenspezifischem -------
    # Shrinkage, dann gewichtete Zusammenfuehrung zu EINEM feat_erfolgsquote.
    #
    # Begruendung (siehe Analyse mit Nutzerin):
    #   1. Ein gepooltes feat_erfolgsquote ueber ALLE Leistungsbereiche
    #      vermischt fachlich inkommensurable Fallmixe (Kardiologie vs.
    #      Geburtshilfe) -- Cluster wurden dadurch scheinbar durch Groesse
    #      (n_indikatoren) erklaerbar, nicht durch echte Qualitaet.
    #   2. Gruppierung nach Fachbereich behebt das, aber NICHT das
    #      Stichprobenrauschen -- das bleibt auch INNERHALB einer Fachgruppe
    #      bestehen (empirisch bestaetigt: |Score-Mittelwert| korreliert
    #      weiterhin negativ mit n_in_gruppe, z.B. Orthopaedie r=-0.456).
    #   Deshalb: pro Fachgruppe eigenes Shrinkage (PSEUDO_N proportional zum
    #   gruppentypischen Nenner, nicht eine globale Konstante -- eine feste
    #   Konstante skaliert schlecht zwischen Gruppen mit sehr unterschiedlichen
    #   typischen Nennergroessen), dann gewichtetes Mittel ueber alle Gruppen,
    #   in denen die Klinik ueberhaupt Daten hat.
    #
    # BEKANNTE, BEWUSST NICHT BEHOBENE LIMITATION -- administrative Redundanz:
    #   Innerhalb einzelner Gruppen wiederholen sich manche Konzepte (z.B.
    #   "Mortalitaet") ueber mehrere administrative Prozedur-Subvarianten
    #   hinweg (z.B. 6x in Kardiologie: Schrittmacher/Defibrillator x
    #   Implantation/Wechsel/Revision; in Orthopaedie gehoeren 85% aller
    #   Indikatoren zu nur 4 Konzept-Familien, in Transplantation 57% allein
    #   zu "Mortalitaet"). Das gibt diesen Konzepten durch schiere Zeilenzahl
    #   mehr Gewicht im gepoolten Gruppen-Score als Konzepten, die nur 1x
    #   vorkommen -- eine Form von Redundanz, keine bewusste Gewichtung.
    #   EIN VERSUCH, dies durch zweistufige Aggregation (erst Konzept-Ebene,
    #   dann gleich gewichtetes Mittel ueber Konzepte) zu beheben, wurde
    #   getestet und wieder verworfen: er verschaerfte das Stichproben-
    #   rauschen drastisch (Korrelation |Score-0.5| vs. n_indikatoren von
    #   -0.12 auf -0.60, std von 0.035 auf 0.246), weil (a) einzelne Konzepte
    #   pro Klinik oft nur auf sehr wenigen Zeilen beruhen und (b) die
    #   Konzepte selbst unterschiedliche Basisraten haben, was ohne eigene
    #   Z-Normalisierung auf Konzept-Ebene eine neue Verzerrung erzeugte.
    #   Diese Idee koennte mit mehr Aufwand (Konzept-Ebene ebenfalls
    #   z-normalisieren, zuverlaessigkeitsgewichtet statt gleichgewichtet
    #   kombinieren) sauberer geloest werden, wurde aber angesichts des
    #   Aufwand-Nutzen-Verhaeltnisses fuer dieses Projekt zurueckgestellt.
    BEREICH_GRUPPEN = {
        "Geburtshilfe_Gynaekologie": [
            "PM-GEBH", "PM-NEO", "GYN-OP",
        ],
        "Kardiologie_Herzchirurgie": [
            "PCI", "HSMDEF-HSM-IMPL", "HSMDEF-HSM-AGGW", "HSMDEF-HSM-REV",
            "HSMDEF-DEFI-IMPL", "HSMDEF-DEFI-AGGW", "HSMDEF-DEFI-REV",
            "KCHK-AK-CHIR", "KCHK-AK-KATH", "KCHK-KC", "KCHK-MK-CHIR",
            "KCHK-MK-KATH", "KCHK-KC-KOMB",
        ],
        "Orthopaedie_Chirurgie": [
            "HGV-HEP", "HGV-OSFRAK", "KEP", "KAROTIS", "CHE",
        ],
        "Allgemeine_Versorgung": [
            "DEK", "CAP",
        ],
        "Onkologie": [
            "MC",
        ],
        "Transplantation": [
            "TX-HTX", "TX-MKU", "NET-NTX", "NET-PNTX",
            "TX-NLS", "TX-LTX", "TX-LLS", "TX-LUTX",
        ],
    }

    SHRINKAGE_ANTEIL = 4.0  # PSEUDO_N = 400% des gruppentypischen (Median-)Nenners
    # Empirisch bestimmt: die Korrelation |Score-0.5| vs. n_indikatoren erreicht
    # bei staerkerem Shrinkage ein Plateau bei ~-0.10 (statt Nulldurchgang wie
    # zunaechst erwartet) -- vermutlich ein kleiner strukureller Rest-Effekt,
    # kein reines Stichprobenrauschen mehr. Staerkeres Shrinkage darueber hinaus
    # (ANTEIL=10: korr=-0.094) bringt kaum weitere Verbesserung, drueckt aber
    # die Gesamtvarianz weiter zusammen (std faellt von 0.037 auf 0.033) --
    # d.h. echtes Signal geht verloren fuer minimalen Rauschgewinn. ANTEIL=4
    # ist der praktische Kompromiss vor diesem Plateau.

    df["lb_kurz"] = df["QSQI.Leistungsbereich"].astype(str).str.split(" ").str[0]

    bereich_scores       = []
    gewichtete_summe_teile = []  # (SO.QBID, gewichtete_score, gewicht) je Gruppe
    for gruppe, lb_liste in BEREICH_GRUPPEN.items():
        df_gruppe = df[df["lb_kurz"].isin(lb_liste)]
        if len(df_gruppe) == 0:
            continue
        agg_gruppe = df_gruppe.groupby("SO.QBID").agg(
            punkte_sum = ("punkte_gew_erreicht", "sum"),
            punkte_max = ("punkte_gew_moeglich", "sum"),
            n_ind_gruppe = ("QSQI.Indikator", "count"),
        ).reset_index()

        # Gruppenspezifisches Shrinkage: PSEUDO_N relativ zum typischen
        # (Median-)Nenner DIESER Gruppe, nicht eine globale Konstante.
        median_nenner   = agg_gruppe.loc[agg_gruppe["punkte_max"] > 0, "punkte_max"].median()
        pseudo_n_gruppe = max(1.0, median_nenner * SHRINKAGE_ANTEIL) if pd.notna(median_nenner) else 1.0
        gruppen_mittel  = (
            agg_gruppe["punkte_sum"].sum() / agg_gruppe["punkte_max"].sum()
            if agg_gruppe["punkte_max"].sum() > 0 else np.nan
        )

        col_name = f"score_{gruppe.lower()}"
        n_ind_col = f"n_indikatoren_{gruppe.lower()}"
        agg_gruppe[col_name] = np.where(
            agg_gruppe["punkte_max"] > 0,
            (agg_gruppe["punkte_sum"] + pseudo_n_gruppe * gruppen_mittel)
            / (agg_gruppe["punkte_max"] + pseudo_n_gruppe),
            np.nan,
        )
        agg_gruppe = agg_gruppe.rename(columns={"n_ind_gruppe": n_ind_col})
        bereich_scores.append(agg_gruppe[["SO.QBID", col_name, n_ind_col]])

        # -- Z-Normalisierung INNERHALB der Gruppe -----------------------------
        # Grund (siehe Analyse): score_orthopaedie_chirurgie liegt im Schnitt bei
        # 0.37, score_onkologie bei 0.61 -- unterschiedliche Fachbereiche haben
        # strukturell unterschiedliche Basisraten, "Bundesdurchschnitt schlagen"
        # ist in manchen Bereichen inherent leichter/schwerer. Ohne Normalisierung
        # wuerde eine Klinik automatisch besser abschneiden, wenn sie zufaellig
        # nur in "leichten" Bereichen (z.B. Onkologie) taetig ist -- unabhaengig
        # von ihrer tatsaechlichen Position relativ zu ihren echten Konkurrenten.
        # Die Z-Normalisierung macht "0.37 im Mittelfeld der Orthopaedie" und
        # "0.61 im Mittelfeld der Onkologie" vergleichbar: beide werden zu z=0.
        gruppe_std = agg_gruppe[col_name].std()
        gruppe_mean_norm = agg_gruppe[col_name].mean()
        z_col = f"{col_name}_z"
        agg_gruppe[z_col] = (
            (agg_gruppe[col_name] - gruppe_mean_norm) / gruppe_std
            if gruppe_std and gruppe_std > 0 else 0.0
        )

        # Fuer die gewichtete Gesamtzusammenfuehrung: Gewicht = punkte_max
        # (Gruppen mit mehr dokumentierten Vergleichen zaehlen staerker)
        teil = agg_gruppe[["SO.QBID", z_col, "punkte_max"]].copy()
        teil = teil[teil[z_col].notna()]
        gewichtete_summe_teile.append(
            teil.rename(columns={z_col: "score", "punkte_max": "gewicht"})
        )

    # -- Gewichtete Zusammenfuehrung aller Fachgruppen-Z-Scores zu EINEM Score -
    alle_teile = pd.concat(gewichtete_summe_teile, ignore_index=True)
    alle_teile["score_x_gewicht"] = alle_teile["score"] * alle_teile["gewicht"]
    kombiniert = alle_teile.groupby("SO.QBID").agg(
        summe_score_gewicht = ("score_x_gewicht", "sum"),
        summe_gewicht        = ("gewicht", "sum"),
        n_fachgruppen        = ("score", "count"),
    ).reset_index()
    kombiniert["feat_erfolgsquote_z"] = (
        kombiniert["summe_score_gewicht"] / kombiniert["summe_gewicht"]
    )
    # Zurueck auf eine 0-1-Skala (Normalverteilungs-CDF), interpretierbar wie
    # zuvor: ~0.5 = durchschnittlich relativ zu echten Fachbereichs-Konkurrenten,
    # aber jetzt korrekt ueber Fachbereiche mit unterschiedlichen Basisraten
    # hinweg vergleichbar.
    kombiniert["feat_erfolgsquote"] = kombiniert["feat_erfolgsquote_z"].apply(
        lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    )

    agg = agg.merge(
        kombiniert[["SO.QBID", "feat_erfolgsquote", "n_fachgruppen"]],
        on="SO.QBID", how="left",
    )
    # Kliniken, die in KEINER der BEREICH_GRUPPEN vorkommen (sollte selten
    # sein, da die Gruppen alle 32 bekannten Leistungsbereiche abdecken):
    # Fallback auf den ungewichteten Pool-Wert statt NaN.
    fehlt_gruppe = agg["feat_erfolgsquote"].isna() & agg["feat_erfolgsquote_roh"].notna()
    agg.loc[fehlt_gruppe, "feat_erfolgsquote"] = agg.loc[fehlt_gruppe, "feat_erfolgsquote_roh"]
    print(f"  Fallback auf Pool-Wert (keine Fachgruppe zugeordnet): {fehlt_gruppe.sum()} KH")

    ergebnis = agg[[
        "SO.QBID", "n_indikatoren",
        "feat_erfolgsquote", "feat_erfolgsquote_roh", "n_fachgruppen",
    ]].copy()

    for bs in bereich_scores:
        ergebnis = ergebnis.merge(bs, on="SO.QBID", how="left")

    tx_lb = ["TX-HTX", "TX-MKU", "NET-NTX", "NET-PNTX",
             "TX-NLS", "TX-LTX", "TX-LLS", "TX-LUTX"]
    df_tx = df[df["lb_kurz"].isin(tx_lb)]
    tx_flag = df_tx.groupby("SO.QBID").size().reset_index()[["SO.QBID"]]
    tx_flag["hat_transplantation"] = 1
    ergebnis = ergebnis.merge(tx_flag, on="SO.QBID", how="left")
    ergebnis["hat_transplantation"] = ergebnis["hat_transplantation"].fillna(0).astype(int)

    score_cols = [c for c in ergebnis.columns if c.startswith("score_")]
    print(f"  Ergebnisqualitaet: {len(ergebnis)} KH, Ø Erfolgsquote: {ergebnis['feat_erfolgsquote'].mean():.3f}")
    print(f"  Bereichs-Scores:   {len(score_cols)} Gruppen")
    for col in score_cols:
        n = ergebnis[col].notna().sum()
        print(f"    {col}: {n} KH")
    return ergebnis








# =============================================================================
# B. STRUKTURQUALITAET
# =============================================================================

def struktur_features_berechnen(
    df_so: pd.DataFrame,
    df_fa: pd.DataFrame,
    df_ppr: pd.DataFrame,
    df_aa: pd.DataFrame,
    df_hb: pd.DataFrame,
    df_notfall: pd.DataFrame,
    df_personal: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strukturfeatures -- alle groessenneutral oder als Anteil berechnet.

    Features:
        feat_arztdichte   -- Faelle pro Bett (auf 95. Perzentil normiert)
        feat_ppr          -- PPR-Erfuellungsgrad / 100
        feat_ausstattung  -- Anzahl Geraete / 25
        feat_ausbildung   -- Anzahl Ausbildungstypen / 19
        feat_notfall      -- Notfallstufe / 3 (0 = keine Teilnahme)
    """
    teile = []

    # -- Arzt- und Pflegedichte: Personal pro Bett ----------------------------
    sp = df_personal.copy()
    sp["SO.QBID"] = zu_zahl(sp["SO.QBID"])
    sp["Anzahl"]  = zu_zahl(
        sp["SO.Personal.Anzahl"].astype(str).str.replace(",", ".", regex=False)
    )

    aerzte = (
        sp[sp["SO.Personal.Type"] == "Ärzte ohne Belegärzte"]
        .groupby("SO.QBID")["Anzahl"].sum()
        .reset_index(name="aerzte_anzahl")
    )
    pflege = (
        sp[sp["SO.Personal.Type"] == "Gesundheits- und Krankenpfleger/in"]
        .groupby("SO.QBID")["Anzahl"].sum()
        .reset_index(name="pflege_anzahl")
    )

    personal = df_so[["SO.QBID", "SO.Betten"]].copy()
    personal["SO.QBID"]   = zu_zahl(personal["SO.QBID"])
    personal["SO.Betten"] = zu_zahl(personal["SO.Betten"])
    personal = personal.merge(aerzte, on="SO.QBID", how="left")
    personal = personal.merge(pflege, on="SO.QBID", how="left")

    # Pro Bett normieren -- groessenneutral
    personal["arzt_pro_bett"]  = (
        personal["aerzte_anzahl"] / personal["SO.Betten"].replace(0, np.nan)
    )
    personal["pflege_pro_bett"] = (
        personal["pflege_anzahl"] / personal["SO.Betten"].replace(0, np.nan)
    )

    # Normierung auf 95. Perzentil
    p95_arzt   = personal["arzt_pro_bett"].quantile(0.95)
    p95_pflege = personal["pflege_pro_bett"].quantile(0.95)

    personal["feat_arztdichte"]   = (personal["arzt_pro_bett"]  / p95_arzt).clip(upper=1.0)
    personal["feat_pflegedichte"]  = (personal["pflege_pro_bett"] / p95_pflege).clip(upper=1.0)
    teile.append(personal[["SO.QBID", "feat_arztdichte", "feat_pflegedichte"]])
    # -- PPR: Pflegepersonalregelung ------------------------------------------
    ppr = df_ppr.copy()
    ppr["SO.QBID"]         = zu_zahl(ppr["SO.QBID"])
    ppr["erfuellungsgrad"] = zu_zahl(ppr["PPR.Erfüllungsgrad"])
    ppr_agg = ppr.groupby("SO.QBID")["erfuellungsgrad"].mean().reset_index()
    ppr_agg["feat_ppr"] = (ppr_agg["erfuellungsgrad"] / 100).clip(upper=1.0)
    teile.append(ppr_agg[["SO.QBID", "feat_ppr"]])

    # -- Ausstattung: Anzahl Geraete / 25 -------------------------------------
    aa = df_aa.copy()
    aa["SO.QBID"] = zu_zahl(aa["SO.QBID"])
    aa_agg = aa.groupby("SO.QBID")["AA.Key"].nunique().reset_index()
    aa_agg["feat_ausstattung"] = anteil(aa_agg["AA.Key"], MAX_WERTE["ausstattung"])
    teile.append(aa_agg[["SO.QBID", "feat_ausstattung"]])

    # -- Ausbildung: Anzahl Ausbildungstypen / 19 -----------------------------
    hb = df_hb.copy()
    hb["SO.QBID"] = zu_zahl(hb["SO.QBID"])
    hb_agg = hb.groupby("SO.QBID")["HB.Key"].nunique().reset_index()
    hb_agg["feat_ausbildung"] = anteil(hb_agg["HB.Key"], MAX_WERTE["ausbildung"])
    teile.append(hb_agg[["SO.QBID", "feat_ausbildung"]])

    # -- Notfallversorgung: Stufe / 3 -----------------------------------------
    notfall = df_notfall.copy()
    notfall["SO.QBID"] = zu_zahl(notfall["SO.QBID"])

    def notfallstufe(wert: str) -> float:
        """Extrahiert Notfallstufe 1/2/3 aus Text. Keine Teilnahme = 0."""
        wert = str(wert).lower()
        if "stufe 3" in wert or "stufe3" in wert:
            return 3.0
        elif "stufe 2" in wert or "stufe2" in wert:
            return 2.0
        elif "stufe 1" in wert or "stufe1" in wert:
            return 1.0
        elif "zugeordnet" in wert:
            return 1.0   # Stufe zugeordnet aber Nummer unbekannt
        else:
            return 0.0

    notfall["stufe"] = notfall[
        "Notfallversorgung.TeilnahmeNotfallstufe"
    ].apply(notfallstufe)
    notfall_agg = notfall.groupby("SO.QBID")["stufe"].max().reset_index()
    notfall_agg["feat_notfall"] = notfall_agg["stufe"] / 3.0
    teile.append(notfall_agg[["SO.QBID", "feat_notfall"]])

    # Zusammenfuehren
    basis = teile[0]
    for teil in teile[1:]:
        basis = basis.merge(teil, on="SO.QBID", how="outer")

    # Fehlende Werte mit 0 auffuellen -- NUR fuer Features, wo "0" inhaltlich
    # "kein Programm vorhanden" bedeutet. feat_arztdichte UND feat_pflegedichte
    # bleiben bewusst NaN, wenn SO.Betten == 0 -- das bedeutet "nicht anwendbar"
    # (z.B. Tagesklinik ohne stationaere Betten), nicht "schlechtester Wert".
    for col in ["feat_ppr", "feat_ausstattung", "feat_ausbildung", "feat_notfall"]:
        basis[col] = basis[col].fillna(0.0)

    print(f"  Strukturqualitaet: {len(basis)} KH")
    return basis


# =============================================================================
# C. PROZESSQUALITAET
# =============================================================================

def prozess_features_berechnen(
    df_amts: pd.DataFrame,
    df_amts_mass: pd.DataFrame,
    df_bm: pd.DataFrame,
    df_hm: pd.DataFrame,
    df_if: pd.DataFrame,
    df_ef: pd.DataFrame,
    df_rm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prozessfeatures -- einheitliche Logik: Anzahl Items / Maximum.

    Features:
        feat_amts     -- AMTS-Massnahmen / 13
        feat_bm       -- BM-Kriterien erfuellt / 8
        feat_hygiene  -- HM-Massnahmen / 6
        feat_if_score -- IF-Systeme / 3
        feat_fehler   -- EF-Systeme / 13
        feat_rm       -- RM-Instrumente / 17

    Fehlende Werte = 0 (kein Programm vorhanden).
    """
    teile = []

    # -- AMTS -----------------------------------------------------------------
    amts = df_amts.copy()
    amts["SO.QBID"] = zu_zahl(amts["SO.QBID"])

    mass_anzahl = (
        df_amts_mass
        .groupby("AMTS.ID")["AMTSMassnahme.Key"]
        .nunique()
        .reset_index(name="n_amts")
    )
    amts = amts.merge(mass_anzahl, on="AMTS.ID", how="left")
    amts["n_amts"] = amts["n_amts"].fillna(0)

    amts_agg = amts.groupby("SO.QBID")["n_amts"].sum().reset_index()
    amts_agg["feat_amts"] = anteil(amts_agg["n_amts"], MAX_WERTE["amts"])
    teile.append(amts_agg[["SO.QBID", "feat_amts"]])

    # -- BM -------------------------------------------------------------------
    bm = df_bm.copy()
    bm["SO.QBID"] = zu_zahl(bm["SO.QBID"])

    bm_kriterien = [
        "BM.Eingeführt",
        "BM.Patientenbefragungen.durchgeführt",
        "BM.Schriftliches.Konzept",
        "BM.Anonym.Möglich",
        "BM.Umgang.Mdl.Beschwerden.geregelt",
        "BM.Umgang.Schriftl.Beschwerden.geregelt",
        "BM.Zeitziele.Rückmeldung.Definiert",
        "BM.Einweiserbefragungen.durchgeführt",
    ]
    vorhanden = [k for k in bm_kriterien if k in bm.columns]
    for k in vorhanden:
        bm[k] = ja_nein_zu_zahl(bm[k])

    bm["n_bm"] = bm[vorhanden].sum(axis=1)
    bm_agg = bm.groupby("SO.QBID")["n_bm"].max().reset_index()
    bm_agg["feat_bm"] = anteil(bm_agg["n_bm"], MAX_WERTE["bm"])
    teile.append(bm_agg[["SO.QBID", "feat_bm"]])

    # -- HM (Hygiene) ---------------------------------------------------------
    hm = df_hm.copy()
    hm["SO.QBID"] = zu_zahl(hm["SO.QBID"])
    hm_agg = hm.groupby("SO.QBID")["HM.Key"].nunique().reset_index()
    hm_agg["feat_hygiene"] = anteil(hm_agg["HM.Key"], MAX_WERTE["hygiene"])
    teile.append(hm_agg[["SO.QBID", "feat_hygiene"]])

    # -- IF (Fehlermeldesysteme) -----------------------------------------------
    iff = df_if.copy()
    iff["SO.QBID"] = zu_zahl(iff["SO.QBID"])
    if_agg = iff.groupby("SO.QBID")["IF.Key"].nunique().reset_index()
    if_agg["feat_if_score"] = anteil(if_agg["IF.Key"], MAX_WERTE["if_score"])
    teile.append(if_agg[["SO.QBID", "feat_if_score"]])

    # -- EF (CIRS) ------------------------------------------------------------
    ef = df_ef.copy()
    ef["SO.QBID"] = zu_zahl(ef["SO.QBID"])
    ef_agg = ef.groupby("SO.QBID")["EF.Key"].nunique().reset_index()
    ef_agg["feat_fehler"] = anteil(ef_agg["EF.Key"], MAX_WERTE["fehler"])
    teile.append(ef_agg[["SO.QBID", "feat_fehler"]])

    # -- RM (Risikomanagement) ------------------------------------------------
    rm = df_rm.copy()
    rm["SO.QBID"] = zu_zahl(rm["SO.QBID"])
    rm_agg = (
        rm.dropna(subset=["RM.Key"])
        .groupby("SO.QBID")["RM.Key"]
        .nunique()
        .reset_index()
    )
    rm_agg["feat_rm"] = anteil(rm_agg["RM.Key"], MAX_WERTE["rm"])
    teile.append(rm_agg[["SO.QBID", "feat_rm"]])

    # Zusammenfuehren
    basis = teile[0]
    for teil in teile[1:]:
        basis = basis.merge(teil, on="SO.QBID", how="outer")

    # Fehlende Werte = 0 (kein Programm = nicht vorhanden)
    prozess_cols = [
        "feat_amts", "feat_bm", "feat_hygiene",
        "feat_if_score", "feat_fehler", "feat_rm",
    ]
    basis[prozess_cols] = basis[prozess_cols].fillna(0.0)

    print(f"  Prozessqualitaet:  {len(basis)} KH")
    return basis


# =============================================================================
# FACHBEREICHE FUER KARTE
# =============================================================================

def fachbereich_features_berechnen(
    df_fa: pd.DataFrame,
    df_am: pd.DataFrame,
    df_am_vavu: pd.DataFrame,
    df_am_vavu_key: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erstellt Fachbereichsliste fuer Kartenpiktogramme.

    Verknuepfungskette:
        FA.csv -> AM.csv -> AM.VAVU.csv -> AM.VAVU.Key.csv

    Nachweis: Fachbereich in offiziellen Daten vorhanden
    Reihenfolge: nach Fallanteil absteigend (groesstes Piktogramm zuerst)
    """
    fa = df_fa.copy()
    fa["FA.QBID"]    = zu_zahl(fa["FA.QBID"])
    fa["FA.FZ.Voll"] = zu_zahl(fa["FA.FZ.Voll"]).fillna(0)
    fa["ABTID"]      = zu_zahl(fa["ABTID"])

    # Fallanteil pro Abteilung berechnen
    gesamt = fa.groupby("FA.QBID")["FA.FZ.Voll"].sum().rename("faelle_gesamt")
    fa = fa.merge(gesamt, on="FA.QBID")
    fa["fallanteil"] = fa["FA.FZ.Voll"] / fa["faelle_gesamt"].replace(0, np.nan)

    # Verknuepfungskette fuer Piktogramme
    am       = df_am.copy()
    vavu     = df_am_vavu.copy()
    vavu_key = df_am_vavu_key.copy()
    am["ABTID"] = zu_zahl(am["ABTID"])

    verknuepft = (
        fa[["FA.QBID", "ABTID", "FA.FZ.Voll", "fallanteil"]]
        .merge(am[["ABTID", "AM.VS.Link"]],   on="ABTID",      how="left")
        .merge(vavu[["AM.VS.Link", "AM.VS.Key"]], on="AM.VS.Link", how="left")
        .merge(
            vavu_key[["AM.VS.Key", "AM.VS.Obergruppe"]],
            on="AM.VS.Key", how="left",
        )
    )

    # Liste der Fachbereiche sortiert nach Fallanteil
    agg = (
        verknuepft.dropna(subset=["AM.VS.Obergruppe"])
        .groupby(["FA.QBID", "AM.VS.Obergruppe"])["fallanteil"]
        .sum()
        .reset_index()
        .sort_values(["FA.QBID", "fallanteil"], ascending=[True, False])
    )

    fachbereich_liste = (
        agg.groupby("FA.QBID")["AM.VS.Obergruppe"]
        .apply(list)
        .reset_index()
        .rename(columns={
            "FA.QBID":           "SO.QBID",
            "AM.VS.Obergruppe":  "fachbereiche_liste",
        })
    )

    ergebnis = fachbereich_liste
    print(f"  Fachbereiche:      {len(ergebnis)} KH mit Spezialisierung")
    return ergebnis


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def transformieren(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Fuehrt alle Features zusammen -- eine Zeile pro Krankenhaus.
    """
    print("\n── Feature Engineering ──────────────────────────────────")

    # A. Ergebnisqualitaet
    feat_ergebnis = erfolgsquote_berechnen(dfs["QS_Qualitätsindikator.csv"])

    # B. Strukturqualitaet
    feat_struktur = struktur_features_berechnen(
        dfs["SO.csv"],
        dfs["FA.csv"],
        dfs["Pflegepersonalregelung.csv"],
        dfs["AA.csv"],
        dfs["HB.csv"],
        dfs["Notfallversorgung.csv"],
        dfs["SO.Personalliste.csv"],
    )

    # C. Prozessqualitaet
    feat_prozess = prozess_features_berechnen(
        dfs["AMTS.csv"],
        dfs["AMTS_Massnahme.csv"],
        dfs["BM.csv"],
        dfs["HM.csv"],
        dfs["IF.csv"],
        dfs["EF.csv"],
        dfs["RM.csv"],
    )

    # Fachbereiche fuer Kartenpiktogramme
    feat_fachbereich = fachbereich_features_berechnen(
        dfs["FA.csv"],
        dfs["AM.csv"],
        dfs["AM.VAVU.csv"],
        dfs["AM.VAVU.Key.csv"],
    )

    # Basistabelle aus SO.csv
    so = dfs["SO.csv"][[
        "SO.QBID", "SO.Name", "SO.Latitude", "SO.Longitude",
        "SO.Bundesland", "SO.Betten", "SO.Psychiatrie",
        "SO.Uni", "KH.Träger.Art",
    ]].copy()
    so["SO.QBID"] = zu_zahl(so["SO.QBID"])

    # Alle Features zusammenfuehren
    master = so.copy()
    for feat_df in [feat_ergebnis, feat_struktur,
                    feat_prozess, feat_fachbereich]:
        master = master.merge(feat_df, on="SO.QBID", how="left")

    # Doppelte Spalten entfernen
    master = master.loc[:, ~master.columns.duplicated()]

    # Zusammenfassung ausgeben
    feature_spalten = [s for s in master.columns if s.startswith("feat_")]
    print(f"\n── Master-DataFrame ─────────────────────────────────────")
    print(f"  Krankenhaeuser:   {len(master):,}")
    print(f"  Spalten gesamt:   {len(master.columns)}")
    print(f"  Feature-Spalten:  {len(feature_spalten)}")
    print(f"\n  Fehlende Werte pro Feature:")
    fehlend = master[feature_spalten].isna().sum()
    fehlend = fehlend[fehlend > 0]
    for spalte, n in fehlend.items():
        print(f"    {spalte:<35} {n:>4} ({n/len(master)*100:.1f}%)")

    return master


def speichern(master: pd.DataFrame) -> None:
    """Speichert den Master-DataFrame als Parquet-Datei."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pfad = PROCESSED_DIR / "master.parquet"
    master.to_parquet(pfad, index=False)
    print(f"\n  Gespeichert: {pfad}")
    print(f"  Groesse:     {pfad.stat().st_size / 1024:.1f} KB")


# =============================================================================
# AUSFUEHREN
# =============================================================================

if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "extract", Path(__file__).parent / "01_extract.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dfs    = mod.alle_einlesen()
    master = transformieren(dfs)
    speichern(master)
