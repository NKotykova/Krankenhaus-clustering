"""
01_extract.py
=============
Liest alle CSV-Dateien aus data/raw/ ein,
prüft die Spaltenstruktur und gibt ein Dictionary zurück.

Rückgabe: {Dateiname: DataFrame}

Ausführen:
    python etl/01_extract.py
"""

import pandas as pd
from pathlib import Path

# -- Einstellungen ------------------------------------------------------------
RAW_DIR  = Path(__file__).parent / "data" / "raw"   # Ordner mit den 86 CSV-Dateien
ENCODING = "utf-8-sig"        # Reale Dateien sind UTF-8 mit BOM, nicht cp1252
SEP      = ","                # Reale Dateien sind komma-separiert, nicht ';'

# Dateien die keine Analysedaten enthalten -- werden übersprungen
SKIP_FILES = {"Error.csv"}

# Erwartete Spalten je Datei -- zur Qualitätsprüfung nach dem Einlesen
EXPECTED_COLUMNS = {
    "SO.csv": [
        "SO.QBID", "SO.Betten", "SO.Latitude", "SO.Longitude",
        "KH.Träger.Art", "SO.Psychiatrie", "SO.Uni", "SO.Bundesland",
    ],
    "FA.csv": [
        "ABTID", "FA.QBID", "FA.Name", "FA.FZ.Voll",
    ],
    "AM.csv": [
        "ABTID", "AM.Key", "AM.VS.Link",
    ],
    "AM_Key.csv": [
        "AM.Key", "AM.Behandlungsmöglichkeit",
    ],
    "AM_VAVU.csv": [
        "AM.VS.Key", "AM.VS.Link",
    ],
    "AM_VAVU_Key.csv": [
        "AM.VS.Key", "AM.VS", "AM.VS.Gruppe", "AM.VS.Obergruppe",
    ],
    "QS.Qualitätsindikator.csv": [
        "SO.QBID", "QSQI.Ergebnis", "QSQI.Bundesdurchschnitt",
        "QSQI.Indikator", "QSQI.Operator", "QSQI.FallzahlGrundgesamtheit",
    ],
    "GIQI.csv": [
        "SO.QBID", "GIQI.Rate", "GIQI.Leistungsbereich",
        "GIQI.Zähler", "GIQI.Nenner",
    ],
    "MM.csv": [
        "MM.Key", "MM.Erbracht", "MM.Mindestmenge",
        "MM.Differenz", "MM.Leistungsbereich",
    ],
    "AMTS.csv": [
        "SO.QBID", "AMTS.AnzahlApotheker",
        "AMTS.VerantGremium", "AMTS.VerantGremiumBeteiligt",
    ],
    "AMTS_Massnahme.csv": [
        "AMTS.ID", "AMTSMassnahme.Key",
    ],
    "BM.csv": [
        "SO.QBID", "BM.Eingeführt",
        "BM.Patientenbefragungen.durchgeführt",
        "BM.Schriftliches.Konzept",
    ],
    "HM.csv": [
        "SO.QBID", "HM.Key", "HM.KISS.Key",
    ],
    "EF.csv": [
        "SO.QBID", "EF.Key",
    ],
    "RM.csv": [
        "SO.QBID", "RM.Key",
    ],
    "Notfallversorgung.csv": [
        "SO.QBID", "Notfallversorgung.TeilnahmeNotfallstufe",
    ],
    "Pflegepersonalregelung.csv": [
        "SO.QBID", "PPR.Erfüllungsgrad",
    ],
}
# -----------------------------------------------------------------------------


def csv_einlesen(dateipfad: Path) -> pd.DataFrame | None:
    """Liest eine CSV-Datei ein. Bei Fehler wird None zurückgegeben."""
    try:
        df = pd.read_csv(
            dateipfad,
            encoding=ENCODING,
            sep=SEP,            
            low_memory=False,
        )
        return df
    except Exception as fehler:
        print(f"  ✗ Fehler beim Lesen von {dateipfad.name}: {fehler}")
        return None


def spalten_pruefen(df: pd.DataFrame, dateiname: str) -> list[str]:
    """Prüft ob alle erwarteten Spalten vorhanden sind. Gibt fehlende Spalten zurück."""
    erwartet = EXPECTED_COLUMNS.get(dateiname, [])
    return [s for s in erwartet if s not in df.columns]


def alle_einlesen(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Hauptfunktion: Liest alle CSV-Dateien ein und prüft die Struktur.
    Gibt ein Dictionary {Dateiname: DataFrame} zurück.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"\n  Ordner '{raw_dir}' nicht gefunden.\n"
            f"  Bitte alle 86 CSV-Dateien in diesen Ordner legen.\n"
            f"  Erwarteter Pfad: {raw_dir.resolve()}"
        )

    csv_dateien = sorted(raw_dir.glob("*.csv"))
    if not csv_dateien:
        raise FileNotFoundError(f"Keine CSV-Dateien in '{raw_dir}' gefunden.")

    print(f"{len(csv_dateien)} CSV-Dateien gefunden\n")

    dataframes: dict[str, pd.DataFrame] = {}
    fehler_liste    = []
    warnungen_liste = []

    for dateipfad in csv_dateien:
        name = dateipfad.name

        # Technische Dateien überspringen
        if name in SKIP_FILES:
            print(f"  –  {name:<50} (übersprungen)")
            continue

        df = csv_einlesen(dateipfad)
        if df is None:
            fehler_liste.append(name)
            continue

        # Spalten prüfen
        fehlende = spalten_pruefen(df, name)
        if fehlende:
            meldung = f"fehlende Spalten: {fehlende}"
            warnungen_liste.append((name, meldung))
            print(f"  ⚠  {name:<50} {len(df):>6,} Zeilen  [{meldung}]")
        else:
            print(f"  ✓  {name:<50} {len(df):>6,} Zeilen")

        dataframes[name] = df

    # Zusammenfassung ausgeben
    print(f"\n{'─'*60}")
    print(f"  Eingelesen:  {len(dataframes)} Dateien")
    print(f"  Fehler:      {len(fehler_liste)}")
    print(f"  Warnungen:   {len(warnungen_liste)}")
    if fehler_liste:
        print(f"\n  Dateien mit Lesefehlern:")
        for f in fehler_liste:
            print(f"    {f}")
    if warnungen_liste:
        print(f"\n  Dateien mit fehlenden Spalten:")
        for name, meldung in warnungen_liste:
            print(f"    {name}: {meldung}")
    print(f"{'─'*60}\n")

    return dataframes


def so_zusammenfassung(so: pd.DataFrame) -> None:
    """Gibt eine Übersicht der Haupttabelle SO.csv aus."""
    print("── SO.csv (Haupttabelle) ────────────────────────────────")
    print(f"  Krankenhäuser gesamt:    {so['SO.QBID'].nunique():,}")
    print(f"  Doppelte QBIDs:          {so['SO.QBID'].duplicated().sum()}")
    print(f"  Fehlende Koordinaten:    {so['SO.Latitude'].isna().sum()}")
    print(f"  Universitätskliniken:    {(so['SO.Uni'] == 1).sum()}")
    print(f"  Psychiatrische Kliniken: {(so['SO.Psychiatrie'] == 'Ja').sum()}")
    print(f"\n  Trägerart:")
    for k, v in so["KH.Träger.Art"].value_counts().items():
        print(f"    {k:<22} {v:>5} ({v/len(so)*100:.1f}%)")
    print(f"\n  Bundesländer (Top 5):")
    for k, v in so["SO.Bundesland"].value_counts().head(5).items():
        print(f"    {k:<30} {v:>5}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    dfs = alle_einlesen()

    if "SO.csv" in dfs:
        so_zusammenfassung(dfs["SO.csv"])

    # Übersicht: welche Dateien haben SO.QBID als Schlüssel
    print("Geladene Dateien (QBID-Status):")
    for name in sorted(dfs.keys()):
        hat_qbid = "SO.QBID" in dfs[name].columns
        print(f"  {'[QBID]' if hat_qbid else '[----]'}  {name}")
