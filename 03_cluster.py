"""
03_cluster.py
=============
Qualitaetsbasiertes Clustering deutscher Krankenhaeuser.

WICHTIGE AENDERUNG gegenueber der ersten Version:
    Frueher wurde nach Anzahl der QS-Indikatoren stratifiziert (Spezialklinik/
    Mittlere/Maximalversorger), um Kliniken unterschiedlicher Groesse getrennt
    zu vergleichen. Analyse hat gezeigt: das Kernproblem war nicht die Groesse
    an sich, sondern zwei ueberlagerte Effekte --
        1. Stichprobenrauschen bei wenigen Indikatoren (behoben durch
           gruppenspezifisches Empirical-Bayes-Shrinkage in 02_transform.py)
        2. Unterschiedliche Basisraten je Fachbereich -- Orthopaedie schlaegt
           den Bundesdurchschnitt im Schnitt nur in 37% der Vergleiche,
           Onkologie in 61% (behoben durch Z-Normalisierung INNERHALB jeder
           Fachgruppe vor der gewichteten Zusammenfuehrung, ebenfalls in
           02_transform.py)
    Mit beiden Korrekturen im bereits berechneten feat_erfolgsquote ist die
    Stratifikation nach Groesse nicht mehr noetig -- der Score ist jetzt
    ueber Kliniken unterschiedlicher Groesse UND Fachrichtung vergleichbar.
    Empirischer Beleg: Korrelation |Score-0.5| vs. n_indikatoren fiel von
    -0.35...-0.45 (unkorrigiert) auf -0.08 (korrigiert), und die
    Top-10%-Verteilung ueber n_indikatoren-Quartile ist jetzt annaehernd
    gleichmaessig statt einseitig auf kleine Kliniken konzentriert.

Clustering:
    EIN K-Means k=3 Lauf ueber ALLE Kliniken mit feat_erfolgsquote
    (kein Stratifizieren, kein getrenntes Clustering je Groesse/Fachbereich).

    Wahl von k=3 -- ZWEI unabhaengige Gruende zugleich:
        1. Interpretierbarkeit: drei kurze, statistisch neutrale Kategorien
           lassen sich fuer die Endnutzerin des Dashboards leicht erklaeren.
        2. Empirisch bestaetigt: Elbow+Silhouette-Analyse (Funktion
           elbow_silhouette_analyse(), k=2..10) zeigt auf dem echten
           Datensatz Silhouette=0.617 bei k=3 -- das GLOBALE MAXIMUM im
           gesamten getesteten Bereich (k=5 z.B. nur 0.576, k=10 nur 0.607).
           k=3 ist damit nicht nur die einfachste, sondern auch die
           statistisch am besten separierte Wahl -- keine willkuerliche
           Kompromissentscheidung.
        Hinweis: eine erste Version dieses Skripts testete auch k=5 auf
        einem kleineren Zwischen-Datensatz, wo k=5 das bessere Silhouette
        zeigte. Auf dem vollstaendigen finalen Datensatz kehrte sich das
        um -- ein Beleg dafuer, warum Kennzahlen immer auf dem tatsaechlich
        finalen Datensatz erneut geprueft werden sollten, nicht nur auf
        einer Zwischenversion.

Cluster-Namen (bewusst neutral, angelehnt an IQTIG-Sprachgebrauch wie
"auffällig"/"unauffällig" -- Position relativ zum Bundesdurchschnitt,
keine wertende Qualitaetsaussage):
    Über Ø / Im Ø / Unter Ø
    + "Keine QS-Daten" fuer KH ohne Qualitaetsdaten

Validierung:
    Struktur/Personal-Features, Bundesland, Traegerart werden NICHT zum
    Clustering verwendet, sondern dienen der inhaltlichen Beschreibung der
    Cluster im Dashboard und der Pruefung, ob Cluster durch etwas anderes
    als Klinikgroesse erklaerbar sind.

Bekannte Limitation (bewusst dokumentiert, nicht behoben):
    Nach Groessen- und Fachbereichs-Korrektur bleibt ein systematischer,
    GERICHTETER Zusammenhang zwischen Klinikgroesse (SO.Betten) und
    feat_erfolgsquote bestehen -- auch INNERHALB einzelner Fachgruppen
    (z.B. corr(score, Betten) = -0.50 in Geburtshilfe, -0.41 in Kardiologie).
    Das ist kein Stichprobenrauschen (das waere symmetrisch, nicht gerichtet)
    und kein Artefakt der gruppenuebergreifenden Zusammenfuehrung (der Effekt
    existiert bereits pro Einzelgruppe). Wahrscheinlichste Erklaerung:
    unvollstaendige Risikoadjustierung -- grosse (oft universitaere/tertiaere)
    Zentren behandeln ueberdurchschnittlich komplexe, von kleineren Haeusern
    zuverwiesene Faelle; ein Vergleich gegen den Bundesdurchschnitt bildet das
    hoehere Grundrisiko nicht vollstaendig ab, wodurch grosse Haeuser trotz
    ggf. gleichwertiger oder besserer Versorgung systematisch schlechter
    abschneiden koennen. Dies ist ein bekanntes, in der Literatur zum
    Krankenhaus-Benchmarking dokumentiertes Problem, keine Schwaeche dieser
    Pipeline speziell.

    Traegerart und Bundesland wurden auf denselben Effekt hin geprueft
    (empirische Werte aus dem tatsaechlichen Lauf auf dem finalen Datensatz):
    - Traegerart: der rohe (nicht groessenkontrollierte) Befund zeigt "privat"
      etwas haeufiger im Cluster "Ueber Ø" (30.0%) als "freigemeinnuetzig"
      (27.5%) oder "oeffentlich" (25.3%) -- ein moderater, kein dramatischer
      Unterschied. Bei Kontrolle nach Groessen-Quartil (kleinstes bis groesstes
      Viertel nach SO.Betten) bleibt "privat" in JEDEM Quartil vorne oder
      gleichauf (29% / 32% / 36% / 22%) gegenueber "oeffentlich"
      (26% / 31% / 25% / 21%) -- die Richtung kehrt sich also NICHT um, wie in
      einer frueheren Zwischenversion dieser Analyse noch beobachtet. Der
      Effekt wird durch Groessenkontrolle etwas kleiner, bleibt aber bestehen.
    - Bundesland: die unkontrollierte Verteilung zeigt spuerbare Unterschiede
      im Cluster "Ueber Ø" (z.B. Bayern 32.8%, Hessen 31.2% vs.
      Baden-Wuerttemberg 21.8%, Nordrhein-Westfalen 22.1%). Anders als bei
      Traegerart wird diese Kontrolle in diesem Skript NICHT zusaetzlich
      nach Groessen-Quartil aufgeschluesselt -- ob der Bundesland-Effekt nach
      Groessenkontrolle bestehen bleibt, ist mit den aktuellen Ausgaben also
      nicht belegt, sondern offen. Moegliche echte regionale Unterschiede
      (Versorgungsstruktur, Landesrecht) sind plausibel, aber nicht mit dieser
      Analyse allein ursaechlich zu klaeren.

    Konsequenz fuer die Interpretation: Cluster-Labels sollten als "Position
    relativ zum bundesweiten Durchschnitt ohne vollstaendige Risikoadjustierung"
    verstanden werden, nicht als absolutes Qualitaetsurteil -- insbesondere bei
    grossen/komplexen Haeusern ist Vorsicht geboten.

Ausfuehren:
    python 03_cluster.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

K = 3
CLUSTER_NAMEN = {1: "Über Ø", 2: "Im Ø", 3: "Unter Ø"}
CLUSTER_REIHENFOLGE = ["Unter Ø", "Im Ø", "Über Ø"]

VALIDIERUNGS_FEATURES = [
    "feat_arztdichte",
    "feat_pflegedichte",
    "feat_ppr",
    "feat_ausstattung",
    "feat_ausbildung",
    "feat_notfall",
    "feat_amts",
    "feat_bm",
    "feat_hygiene",
    "feat_if_score",
    "feat_fehler",
    "feat_rm",
    "SO.Betten",
    "n_indikatoren",
    "n_fachgruppen",
]


def elbow_silhouette_analyse(df: pd.DataFrame, feature: str = "feat_erfolgsquote",
                              k_bereich=range(2, 11)) -> None:
    """
    Elbow (WCSS) + Silhouette ueber einen Bereich von k, um die Wahl von K
    empirisch zu pruefen -- analog zur Methodik im Gruppenprojekt (Folie
    'Elbow-Analyse: Optimale Cluster-Anzahl'), aber auf dem korrigierten
    (Fachgruppen-normierten, geshrinkten) feat_erfolgsquote statt der rohen
    Erfolgsquote.

    WICHTIG fuer die Interpretation: das hier gewaehlte K wird NICHT rein
    nach dem statistischen Optimum bestimmt, sondern nach Interpretierbarkeit
    fuer die Endnutzerin (kurze, neutrale Kategorien). Diese Funktion dient
    dazu, transparent zu zeigen, WAS das statistische Optimum waere, damit
    die Abweichung davon eine bewusste, dokumentierte Entscheidung ist.
    """
    X = StandardScaler().fit_transform(df[[feature]])

    print(f"\n  Elbow + Silhouette fuer k=2..{k_bereich[-1]} (Feature: {feature}):")
    print(f"  {'k':>3} {'WCSS':>10} {'Silhouette':>12}")
    for k in k_bereich:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        wcss = km.inertia_
        sil = silhouette_score(X, labels)
        markiert = "  <- gewaehlt" if k == K else ""
        print(f"  {k:>3} {wcss:>10.2f} {sil:>12.3f}{markiert}")



def kmeans_clustern(df: pd.DataFrame, k: int, feature: str = "feat_erfolgsquote") -> pd.DataFrame:
    """K-Means Clustering auf einer Feature-Spalte, benannt nach Score-Rang."""
    df = df.copy()
    X = df[[feature]].copy()
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    df["cluster_id"] = km.fit_predict(X_scaled)

    cluster_scores = df.groupby("cluster_id")[feature].mean()
    rang = cluster_scores.rank(ascending=False).astype(int)
    df["cluster_label"] = df["cluster_id"].map(rang).map(CLUSTER_NAMEN)

    sil = silhouette_score(X_scaled, df["cluster_id"])
    print(f"\n  K-Means k={k}, Silhouette={sil:.3f}")
    verteilung = df["cluster_label"].value_counts()
    for name in CLUSTER_REIHENFOLGE:
        if name in verteilung:
            print(f"    {name:<20} {verteilung[name]:>5} KH  "
                  f"(Ø score={df[df['cluster_label']==name][feature].mean():.3f})")

    return df[["cluster_label"]]


def cluster_validieren(df: pd.DataFrame) -> None:
    """
    Zeigt wie sich Struktur/Personal-Features zwischen Clustern unterscheiden.
    Prueft explizit, ob die Cluster noch durch Groesse (SO.Betten,
    n_indikatoren) erklaerbar sind, oder durch etwas anderes.
    """
    vorhanden = [f for f in VALIDIERUNGS_FEATURES if f in df.columns]
    cluster_namen = CLUSTER_REIHENFOLGE

    print(f"\n  Validierung (Struktur/Personal nach Cluster):")
    print(f"  {'Feature':<30}", end="")
    for name in cluster_namen:
        if name in df["cluster_label"].values:
            print(f"  {name:<18}", end="")
    print()
    print(f"  {'-'*95}")

    for feat in vorhanden:
        print(f"  {feat:<30}", end="")
        for name in cluster_namen:
            maske = df["cluster_label"] == name
            if maske.sum() > 0:
                wert = df[maske][feat].mean()
                print(f"  {wert:<18.3f}", end="")
        print()

    print(f"\n  Territorium (Bundesland, Top-5) -- Anteil je Cluster:")
    top_bl = df["SO.Bundesland"].value_counts().head(5).index
    ct = pd.crosstab(
        df[df["SO.Bundesland"].isin(top_bl)]["SO.Bundesland"],
        df["cluster_label"], normalize="index",
    )
    print((ct * 100).round(1))

    print(f"\n  Traegerart -- Anteil je Cluster:")
    ct2 = pd.crosstab(df["KH.Träger.Art"], df["cluster_label"], normalize="index")
    print((ct2 * 100).round(1))

    # -- Kontrolle: bleibt der Traegerart-/Bundesland-Effekt nach Groesse ------
    # bestehen, oder ist er nur ein Schatten von SO.Betten? (siehe Docstring
    # oben fuer die ausfuehrliche Erklaerung dieser bekannten Limitation)
    print(f"\n  Kontrolle: Anteil 'Über Ø' nach Traegerart, INNERHALB Groessen-Quartilen")
    print(f"  (prueft ob der Traegerart-Effekt echt ist oder nur ein Groessen-Schatten):")
    df_kontrolle = df.copy()
    df_kontrolle["betten_bucket"] = pd.qcut(df_kontrolle["SO.Betten"], q=4, duplicates="drop")
    for bucket in df_kontrolle["betten_bucket"].cat.categories:
        b = df_kontrolle[df_kontrolle["betten_bucket"] == bucket]
        ct_b = pd.crosstab(b["KH.Träger.Art"], b["cluster_label"] == "Über Ø", normalize="index")
        if True in ct_b.columns:
            zeile = ", ".join(f"{idx}={v*100:.0f}%" for idx, v in ct_b[True].items())
            print(f"    {str(bucket):<20} {zeile}")


def clustern(df: pd.DataFrame) -> pd.DataFrame:
    """Fuehrt EIN Clustering ueber alle Kliniken mit QS-Daten durch."""
    mit_qs  = df[df["feat_erfolgsquote"].notna()].copy()
    ohne_qs = df[df["feat_erfolgsquote"].isna()].copy()

    print(f"  Mit QS-Daten:    {len(mit_qs):,} KH")
    print(f"  Ohne QS-Daten:   {len(ohne_qs):,} KH")

    elbow_silhouette_analyse(mit_qs)

    ergebnis = df.copy()
    ergebnis["cluster_label"] = "Keine QS-Daten"

    labels = kmeans_clustern(mit_qs, k=K)
    ergebnis.loc[mit_qs.index, "cluster_label"] = labels["cluster_label"].values

    cluster_validieren(mit_qs.assign(cluster_label=labels["cluster_label"].values))

    return ergebnis


if __name__ == "__main__":
    master_pfad = PROCESSED_DIR / "master.parquet"
    if not master_pfad.exists():
        raise FileNotFoundError("Bitte zuerst 02_transform.py ausfuehren.")

    df = pd.read_parquet(master_pfad)
    df["SO.Betten"] = pd.to_numeric(df["SO.Betten"], errors="coerce")
    print(f"Geladen: {len(df):,} Krankenhaeuser, {len(df.columns)} Spalten\n")

    print("── Clustering ───────────────────────────────────────────")
    df_final = clustern(df)

    print(f"\n── Gesamtergebnis ───────────────────────────────────────")
    print(f"\n  {'Cluster':<25} {'Anzahl':>7}")
    print(f"  {'-'*35}")
    for cluster in CLUSTER_REIHENFOLGE[::-1] + ["Keine QS-Daten"]:
        n = (df_final["cluster_label"] == cluster).sum()
        if n > 0:
            print(f"  {cluster:<25} {n:>7}")

    ausgabe = PROCESSED_DIR / "master_clustered.parquet"
    df_final.to_parquet(ausgabe, index=False)
    print(f"\n── Gespeichert ──────────────────────────────────────────")
    print(f"  {ausgabe}")
    print(f"  {len(df_final):,} Krankenhaeuser, {len(df_final.columns)} Spalten")
