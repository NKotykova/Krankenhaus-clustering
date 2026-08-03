"""
0_cleanup.py
============
Bereinigt QS.Qualitätsindikator.csv, OHNE das Original zu verändern/
zu überschreiben. Das Ergebnis wird als separate Datei gespeichert.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

EINGABE_DATEI  = Path(__file__).parent / "data" / "raw" / "QS.Qualitätsindikator.csv"
AUSGABE_DATEI  = Path(__file__).parent / "data" / "raw" / "QS_Qualitätsindikator.csv"

EXCEL_DATUM_PATTERN = re.compile(
    r"^(Jan|Feb|Mrz|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez|"
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"[\s\-]?\d{1,4}$|^\d{1,2}[\s\-](Jan|Feb|Mrz|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)$",
    re.IGNORECASE,
)

INDIKATOREN_MASSTAB_X100 = [
    "56102", "56103", "56104", "56105", "56106", "56107", "56108",
    "56109", "56110", "56111", "56112", "56113", "56114", "56115", "56116",
]

INDIKATOR_RICHTUNG_OVERRIDE = {
    "2163":   ">=", "56100":  ">=", "56101":  ">=", "56102":  ">=",
    "56103":  ">=", "56104":  ">=", "56105":  ">=", "56106":  ">=",
    "56107":  ">=", "56108":  ">=", "56109":  ">=", "56110":  ">=",
    "56111":  ">=", "56112":  ">=", "56113":  ">=", "56114":  ">=",
    "56115":  ">=", "56116":  ">=", "56117":  ">=", "56118":  ">=",
    "102001": ">=", "132003": ">=",
}


def schritt1_deduplizieren(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt vollständig identische Zeilen (echte Duplikate)."""
    vorher = len(df)
    df_bereinigt = df.drop_duplicates()
    print(f"  [1] Deduplizierung: {vorher:,} -> {len(df_bereinigt):,} Zeilen "
          f"(-{vorher - len(df_bereinigt):,})")
    return df_bereinigt


def schritt2_ergebnis_reparieren(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repariert QSQI.Ergebnis-Werte, die durch Excel-Autokorrektur fälschlich
    als Datum interpretiert und gespeichert wurden (z.B. "Jan 15" statt "15").
    Wo möglich, wird der echte Wert aus FallzahlBeobachteteEreignisse /
    FallzahlGrundgesamtheit rekonstruiert. Andernfalls -> NaN.
    """
    df = df.copy()
    ergebnis_text = df["QSQI.Ergebnis"].astype(str).str.strip()
    ist_excel_datum = ergebnis_text.str.match(EXCEL_DATUM_PATTERN, na=False)

    beob  = pd.to_numeric(df["QSQI.FallzahlBeobachteteEreignisse"], errors="coerce")
    grund = pd.to_numeric(df["QSQI.FallzahlGrundgesamtheit"], errors="coerce")
    rekonstruierbar = ist_excel_datum & beob.notna() & grund.notna() & (grund > 0)

    ergebnis_repariert = pd.to_numeric(
        ergebnis_text.str.replace(",", ".", regex=False), errors="coerce"
    )
    ergebnis_repariert.loc[rekonstruierbar] = (
        beob.loc[rekonstruierbar] / grund.loc[rekonstruierbar] * 100
    )
    ergebnis_repariert.loc[ist_excel_datum & ~rekonstruierbar] = np.nan

    df["QSQI.Ergebnis_original"]      = df["QSQI.Ergebnis"]
    df["QSQI.Ergebnis"]               = ergebnis_repariert
    df["QSQI.Ergebnis_war_verfaelscht"] = ist_excel_datum

    n_verfaelscht  = ist_excel_datum.sum()
    n_rekonstr     = rekonstruierbar.sum()
    print(f"  [2] Excel-Datum in Ergebnis: {n_verfaelscht:,} Zeilen "
          f"({n_verfaelscht/len(df)*100:.1f}%), "
          f"rekonstruiert {n_rekonstr:,}, verloren (NaN) {n_verfaelscht - n_rekonstr:,}")
    return df


def schritt3_bundesdurchschnitt_masstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Korrigiert einen Skalenfehler bei bestimmten Indikatoren (56102-56116):
    QSQI.Bundesdurchschnitt liegt dort fälschlich im Bereich 0-1 statt 0-100.
    """
    df = df.copy()
    ind_nr = df["QSQI.Indikator"].astype(str).str.extract(r"^(\d+)")[0]
    bund   = pd.to_numeric(
        df["QSQI.Bundesdurchschnitt"].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )

    maske = ind_nr.isin(INDIKATOREN_MASSTAB_X100) & (bund <= 1) & bund.notna()
    n_betroffen = maske.sum()

    df["QSQI.Bundesdurchschnitt_original"] = df["QSQI.Bundesdurchschnitt"]
    bund_korrigiert = bund.copy()
    bund_korrigiert.loc[maske] = bund.loc[maske] * 100
    df["QSQI.Bundesdurchschnitt"] = bund_korrigiert

    print(f"  [3] Maßstab Bundesdurchschnitt (x100): {n_betroffen:,} Zeilen korrigiert")
    return df


def schritt4_operator_richtung(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bestimmt die Vergleichsrichtung (>= oder <=) für Zeilen, bei denen
    QSQI.Operator in den Rohdaten leer ist -- über einen recherchierten
    Override je Indikator, sonst über einen konservativen Standardwert ('<=').
    """
    df = df.copy()
    ind_nr = df["QSQI.Indikator"].astype(str).str.extract(r"^(\d+)")[0]
    hat_operator = df["QSQI.Operator"].notna()

    richtung = pd.Series(index=df.index, dtype=object)
    richtung[hat_operator] = df.loc[hat_operator, "QSQI.Operator"].astype(str).str.strip()

    fehlt = ~hat_operator
    override = ind_nr[fehlt].map(INDIKATOR_RICHTUNG_OVERRIDE)
    richtung[fehlt] = override.fillna("<=")

    df["QSQI.Operator_original"] = df["QSQI.Operator"]
    df["QSQI.Operator_bestimmt"] = richtung

    n_override = (fehlt & ind_nr.isin(INDIKATOR_RICHTUNG_OVERRIDE.keys())).sum()
    n_standard = (fehlt & ~ind_nr.isin(INDIKATOR_RICHTUNG_OVERRIDE.keys())).sum()
    print(f"  [4] Richtung für leere Operator-Werte: "
          f"{n_override:,} über Override, {n_standard:,} über Standard '<='")
    return df


def main():
    if not EINGABE_DATEI.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {EINGABE_DATEI.resolve()}")

    AUSGABE_DATEI.parent.mkdir(parents=True, exist_ok=True)

    print(f"Lese: {EINGABE_DATEI}")
    df = pd.read_csv(EINGABE_DATEI, sep=",", encoding="utf-8-sig", low_memory=False)
    print(f"Eingelesen: {len(df):,} Zeilen, {len(df.columns)} Spalten\n")

    df = schritt1_deduplizieren(df)
    df = schritt2_ergebnis_reparieren(df)
    df = schritt3_bundesdurchschnitt_masstab(df)
    df = schritt4_operator_richtung(df)

    df.to_csv(AUSGABE_DATEI, index=False, sep=",", encoding="utf-8-sig")

    print(f"\nGespeichert: {AUSGABE_DATEI}")
    print(f"Größe: {len(df):,} Zeilen, {len(df.columns)} Spalten")
    print(f"\nDie originale Eingabedatei wurde NICHT verändert.")


if __name__ == "__main__":
    main()
