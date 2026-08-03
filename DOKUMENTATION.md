# Dokumentation: app_analyst.py

Diese Dokumentation erklärt jeden Teil des Programms in einfacher Sprache.
Kein Fachjargon ohne Erklärung. Für jeden Abschnitt: **Was passiert hier?**
und **Warum ist das so gemacht?**

**Ergebnis der Prüfung vorab:** Der Code wurde getestet (Syntax-Prüfung und
Ausführung aller drei Seiten). Es gibt **keine Fehler**, die das Programm
zum Absturz bringen. Es gibt nur harmlose Warnungen (z. B. eine veraltete
Streamlit-Funktion, die noch funktioniert, aber irgendwann ersetzt werden
sollte). Diese sind am Ende der Dokumentation aufgelistet.

---

## 1. Kopf des Programms (Zeilen 1–34)

**Was passiert hier?**
Das Programm beschreibt sich selbst (ein Kommentar, kein Code). Danach
werden Werkzeuge geladen, die das Programm braucht:
- `streamlit` — baut die Webseite (Knöpfe, Tabellen, Diagramme).
- `pandas` — rechnet mit den Krankenhaus-Daten (wie eine sehr große
  Excel-Tabelle).
- `plotly` — zeichnet die Diagramme und die Karte.
- `sklearn` (scikit-learn) — macht die Cluster-Berechnung (K-Means) und
  prüft, wie gut die Cluster sind (Silhouette).

Danach wird der Pfad zur Datendatei festgelegt:
`DATEI = Path("data/processed/master_clustered.parquet")`

**Warum so gemacht?**
Diese Datei enthält bereits alle berechneten Ergebnisse (aus den Skripten
`01_extract.py` bis `03_cluster.py`). Das Dashboard liest nur diese fertige
Datei — es rechnet die Cluster nicht selbst neu aus.

---

## 2. Design-Farben und feste Begriffe (Zeilen 36–90)

**Was passiert hier?**
Hier stehen alle Farben und Namen, die im ganzen Programm benutzt werden,
an einer Stelle gesammelt:
- `COLOR` — die Grundfarben (dunkelblau für Text, grün/blau/braun für die
  drei Cluster, ein Gold-Ton nur für "Befund"-Hinweise).
- `CLUSTER_FARBEN` — welche Farbe zu welchem Cluster gehört.
- `NEBEN_FARBEN` / `GROESSEN_FARBEN` — eigene Farben für alles, was
  **kein** Cluster ist (Größenklasse, Trägerart), damit man es nicht
  verwechselt.
- `FACHBEREICHE` — die 6 medizinischen Fachbereiche, die verglichen
  werden (z. B. Kardiologie), zusammen mit dem Namen der Datenspalte.
- `FEATURE_LABELS` — die 12 Struktur-Merkmale (z. B. Arztdichte) mit
  ihrem lesbaren Namen.
- `QI_INDIKATOREN_GESAMT = 199` — eine feste Zahl. Sie kommt **nicht**
  aus einer Berechnung im Dashboard, sondern wurde vorher in
  `02_transform.py` ermittelt und hier nur als bekannter Wert eingetragen
  (weil die Rohdaten dem Dashboard nicht vorliegen).
- `DE_CENTER` / `DE_ZOOM` — wo die Karte zentriert ist und wie nah
  hineingezoomt wird.

**Warum so gemacht?**
Wenn man später eine Farbe ändern will, muss man sie nur an **einer**
Stelle ändern, nicht in 20 verschiedenen Zeilen im Code suchen.

---

## 3. CSS — das Erscheinungsbild (Zeilen 92–235)

**Was passiert hier?**
Das ist kein Python-Code, sondern **CSS** (die Sprache, die das Aussehen
von Webseiten steuert) — hier als Text in Python eingebettet. Es legt fest:
- Welche Schriftart benutzt wird (`Inter`, von Google geladen).
- Wie groß der Text an welcher Stelle ist.
- Wie die Karten (`.kpi`, `.befund`, `.fach-card`) aussehen: weißer
  Hintergrund, runde Ecken, ein Rahmen.
- Wie die Knöpfe in der Seitenleiste (Sidebar) aussehen — inklusive der
  Regel, dass ein **aktiver** Cluster-Knopf in seiner Cluster-Farbe
  ausgefüllt ist, ein **inaktiver** nur einen farbigen Rand hat.

**Warum so gemacht?**
Ohne dieses CSS sähe das Dashboard aus wie ein Standard-Streamlit-Programm
(graue Standardknöpfe, kleine Standardschrift). Das CSS macht daraus ein
Layout, das aussieht wie ein professioneller Bericht.

---

## 4. Daten laden (Zeilen 242–306)

### `daten_laden()`
**Was passiert hier?**
1. Liest die fertige Datendatei ein.
2. Wandelt einige Text-Spalten (Breitengrad, Längengrad, Betten) in echte
   Zahlen um — falls sie als Text mit Komma gespeichert wurden.
3. Füllt fehlende Cluster-Werte mit `"Keine QS-Daten"` — das sind Kliniken,
   die keine Qualitätsindikatoren gemeldet haben.
4. Teilt alle Kliniken nach Bettenzahl in 4 gleich große Gruppen
   (Größenklassen: Klein / Mittel / Groß / Sehr groß).
5. Berechnet für jeden der 6 Fachbereiche einen **Prozentwert**: Wie gut
   ist diese Klinik im Vergleich zu allen anderen Kliniken in genau
   diesem Fachbereich? (Nicht im Vergleich zu allen Kliniken insgesamt.)

**Warum so gemacht?**
Punkt 5 ist wichtig: Ein Krankenhaus, das nur Geburtshilfe macht, soll
nicht mit einem verglichen werden, das nur Kardiologie macht — die
"normalen" Werte sind in jedem Fachbereich unterschiedlich. Deshalb wird
hier für jeden Fachbereich eine eigene Vergleichsgruppe gebildet
(statistisch: Z-Wert, dann in eine Prozentzahl von 0–100 % umgerechnet).

Der Befehl `@st.cache_data` über der Funktion bedeutet: Streamlit merkt
sich das Ergebnis. Beim nächsten Klick wird die Datei nicht erneut von der
Festplatte gelesen — das macht das Dashboard schneller.

### `fachbereich_statistik()`
**Was passiert hier?**
Berechnet für jeden Fachbereich: Wie stark streuen die Werte über alle
Kliniken (Standardabweichung)? Und: Wie viele Qualitätsindikatoren hat
eine "typische" Klinik in diesem Fachbereich (Median)?

**Warum so gemacht?**
Diese Zahlen werden später gebraucht, um in der Klinik-Profil-Seite zu
erklären, **warum** eine Prozentzahl so ist, wie sie ist (siehe Abschnitt
9 unten).

### `groessen_grenzen()`
**Was passiert hier?**
Berechnet, bei wie vielen Betten genau die Grenze zwischen "Klein" und
"Mittel" liegt (und so weiter) — als echte Zahl, nicht nur als Name.

**Warum so gemacht?**
Damit in der Seitenleiste beim Überfahren mit der Maus über den Knopf
"Klein" eine echte Zahl steht (z. B. "17–106 Betten"), nicht nur das Wort
"klein".

### `elbow_silhouette_berechnen()`
**Was passiert hier?**
Rechnet für k = 2 bis 10 (also 2 bis 10 mögliche Cluster-Anzahlen) jeweils
eine K-Means-Gruppierung und misst zwei Werte: WCSS (wie kompakt sind die
Gruppen) und Silhouette (wie gut trennen sich die Gruppen voneinander).

**Warum so gemacht?**
Das ist der Beweis für "Befund 01": k=3 ist nicht geraten, sondern
das Ergebnis mit dem besten Silhouette-Wert im ganzen getesteten Bereich.

---

## 5. Bausteine — kleine Helfer-Funktionen (Zeilen 309–390)

Diese Funktionen bauen wiederkehrende, kleine HTML-Bausteine. Sie werden
später in allen drei Seiten immer wieder aufgerufen, statt den gleichen
HTML-Code jedes Mal neu zu schreiben.

- **`hex_zu_rgba()`** — wandelt eine Farbe wie `#4C9A4E` in eine
  durchsichtige Version um (für helle Hintergründe hinter Text).
- **`befund_karte()`** — baut eine der drei "Befund"-Karten. Der
  ausführliche Text steht nicht direkt auf der Karte, sondern im
  `title`-Attribut — das erzeugt den kleinen Hinweis-Kasten, der beim
  Überfahren mit der Maus erscheint (Hover-Tooltip).
- **`kpi_karte()`** — baut eine Kachel mit einer großen Zahl rechts und
  einem Titel/Untertitel links (z. B. "Fachbereiche — 6").
- **`info_karte()`** — eine einfachere Kachel-Variante, ohne die große
  Zahl rechts (für Text-Werte wie "Bundesland: Bayern").
- **`cluster_badge()`** — baut das farbige Label, das den Cluster einer
  Klinik zeigt (z. B. "Über Ø" in Grün). Für "Keine QS-Daten" wird der
  Text automatisch zu "Keine QS-Daten (Ergebnisqualität)" erweitert — das
  macht klar, dass nur die Ergebnis-Indikatoren fehlen, nicht alle Daten.
- **`fach_farbe()`** — entscheidet: Ist der Prozentwert eines Fachbereichs
  grün (≥ 60 %), gelb-braun (40–60 %) oder rot (< 40 %)?

---

## 6. Erklärtexte für die Seitenleiste (Zeilen 368–433)

- **`CLUSTER_HELP`** — feste Erklärtexte für die drei Cluster-Knöpfe
  (was "Über Ø" / "Im Ø" / "Unter Ø" bedeutet).
- **`groesse_help_text()`** — baut den Erklärtext für die
  Größenklasse-Knöpfe, mit den echten Betten-Zahlen aus
  `groessen_grenzen()`.

### `sidebar_filter()`
**Was passiert hier?**
Baut die komplette Seitenleiste mit den Filter-Knöpfen:
1. Drei Knöpfe für die Cluster (mit Anzahl, z. B. "491 · Unter Ø").
   Klick auf einen Knopf schaltet diesen Cluster ein/aus.
2. Vier Knöpfe für die Größenklasse (ohne Anzahl).
3. Ein Knopf "Filter zurücksetzen", der beide Filter wieder auf "alle
   ausgewählt" stellt.

**Warum so gemacht?**
Die Filter werden in `st.session_state` gespeichert — das ist Streamlits
Art, sich Dinge zwischen einem Klick und dem nächsten zu merken (sonst
würde bei jedem Klick alles vergessen).

Ein technisches Detail: `st.sidebar.container(key=f"filterbtn_...")` gibt
jedem Knopf-Container einen eindeutigen Namen. Nur dadurch kann das CSS
weiter oben jeden Knopf einzeln in seiner Cluster-Farbe einfärben.

---

## 7. Seite 1: Übersicht (Zeilen 439–593)

**Reihenfolge auf der Seite, von oben nach unten:**

1. **Kopfzeile** — Titel und Datenstand.
2. **Berechnungen im Hintergrund** (noch nicht sichtbar): Wie viele
   Kliniken sind nach den aktuellen Filtern übrig? Wie hoch ist die
   durchschnittliche Erfolgsquote? — Diese Zahlen werden gleich gebraucht.
3. **Zwei Spalten, halbe Breite jede** (`col_map` und `col_side`):
   - **Rechte Spalte** (`col_side`, wird zuerst befüllt):
     - Die drei Befund-Karten, übereinander.
     - "Umfang der Analyse": 6 Kacheln in 3 Zweier-Reihen.
   - **Linke Spalte** (`col_map`): die Deutschlandkarte.
   - **Rechte Spalte, zweiter Teil**: die Kachel "Klinken mit QS-Daten",
     danach das Diagramm "Top / Flop 5 Kliniken".

**Warum in dieser Reihenfolge?**
Wichtig: Man kann in Streamlit dieselbe Spalte **mehrmals** befüllen
(hier `col_side` zweimal). Das Programm zeichnet zuerst alles für
`col_side`, dann alles für `col_map`, dann noch einmal etwas für
`col_side` — Streamlit setzt es trotzdem in der richtigen Spalte
übereinander. So kann die Karte in der Mitte des Codes stehen, obwohl sie
optisch "zwischen" zwei Teilen der rechten Spalte liegt.

**Die Karte im Detail:**
- Für jeden Cluster (oder jede Größenklasse, je nach Auswahl) wird eine
  eigene Punktwolke auf die Karte gezeichnet (`go.Scattermapbox`).
- `bounds` (Zeile 531) begrenzt, wie weit man die Karte wegzoomen/
  verschieben kann — verhindert, dass man "verloren geht".
- `zoom` und `center` bestimmen, was beim ersten Laden zu sehen ist.
- Klickt man auf einen Punkt, öffnet ein Knopf "Profil öffnen" das
  Klinik-Profil dieser Klinik (über `st.session_state`).

**Top/Flop-Diagramm im Detail:**
- Nimmt die 5 besten und 5 schlechtesten Kliniken (nach Erfolgsquote).
- Zeichnet für jede einen Balken, farbig nach Cluster.
- `textposition="outside"` sorgt dafür, dass die Prozent-Zahl **neben**
  dem Balken steht, nicht darin — sonst würde die Zahl bei einem Balken,
  der schon fast 100 % lang ist, über den Rand hinausragen und
  abgeschnitten werden.
- Klick auf einen Balken öffnet ebenfalls das Klinik-Profil.

---

## 8. Seite 2: Cluster-Analyse (Zeilen 601–760)

Diese Seite liefert die **Beweise** hinter den drei Befunden.

**Zeile aus zwei Diagrammen (halbe Breite jedes):**
- **Links — "Zu Befund 01":** Das Elbow/Silhouette-Diagramm. Zeigt für
  k=2 bis 10 den Silhouette-Wert. Eine senkrechte Linie markiert k=3.
- **Rechts — "Zu Befund 03":** Zeigt, wie sich der Anteil "Über Ø" nach
  Trägerart (privat/öffentlich/freigemeinnützig) verändert, **wenn man
  nach Klinikgröße aufteilt** (4 Größen-Viertel auf der X-Achse). Das
  beweist: Der scheinbare Unterschied zwischen Trägerarten wird kleiner,
  wenn man nach Größe kontrolliert.

**Tabelle "Zu Befund 02":**
Zeigt alle 12 Struktur-Merkmale, jeweils als Durchschnitt für die drei
Cluster. Der beste Wert einer Zeile wird grün markiert, der schlechteste
rot.

**Auswahlmenü + Kastendiagramm:**
Man kann ein Merkmal aus der Liste wählen. Darunter erscheint ein
Kastendiagramm (Boxplot), das die **volle Verteilung** dieses Merkmals je
Cluster zeigt — nicht nur den Durchschnitt aus der Tabelle, sondern auch,
wie stark die Werte innerhalb eines Clusters streuen.

**Klinikgröße je Cluster (Kastendiagramm):**
Zeigt die Bettenzahl je Cluster. Die Y-Achse ist bei 1.000 Betten
abgeschnitten (mit Hinweistext), weil einzelne sehr große Kliniken
(bis über 3.000 Betten) sonst das ganze Diagramm zusammendrücken würden.
Auf jeder Box steht der Median als Zahl.

**"Wie homogen sind die Cluster wirklich?"**
Zwei Diagramme:
- **Links (schmal):** Streuung innerhalb jedes Clusters
  (Variationskoeffizient = Standardabweichung geteilt durch Mittelwert).
  Ein hoher Wert bedeutet: Die Kliniken in diesem Cluster sind sehr
  unterschiedlich, obwohl sie im selben Cluster sind.
- **Rechts (breit):** Wie viel Prozent der Kliniken in jedem Cluster
  haben überhaupt Daten zu jedem der 6 Fachbereiche?

Am Ende: eine gelbe Hinweisbox, die diese beiden Diagramme in Worte
übersetzt.

---

## 9. Seite 3: Klinik-Profil (Zeilen 767–901)

**Suchfeld:**
Ein Auswahlmenü mit allen Klinik-Namen. Wenn man vorher auf der Karte
oder im Top/Flop-Diagramm auf eine Klinik geklickt hat, ist diese hier
schon vorausgewählt (`st.session_state.profil_kh`).

**Obere Zeile (zwei Spalten):**
- **Links:** das Cluster-Label, dann 5 Info-Kacheln (Bundesland, Betten,
  Träger, Größenklasse, Datengrundlage). Falls die Klinik zu den größten
  25 % gehört UND "Unter Ø" ist (oder Transplantationen durchführt),
  erscheint eine gelbe Warnbox mit Verweis auf Befund 02.
- **Rechts:** ein Tacho-Diagramm (Gauge), das zeigt, wo die
  Gesamt-Erfolgsquote dieser Klinik im Vergleich zum Bundesdurchschnitt
  liegt (0–100 %).

**"Qualität je Fachbereich" — die 6 Kacheln:**
Für jeden der 6 Fachbereiche eine Kachel mit Prozentwert. Wichtig: Dieser
Prozentwert vergleicht die Klinik **nur** mit anderen Kliniken im
**selben** Fachbereich — nicht mit allen Kliniken.

Beim Überfahren mit der Maus erscheint ein Hinweistext mit drei
Informationen:
1. Die Position in Standardabweichungen (Z-Wert).
2. Ob dieser Fachbereich eng oder breit gestreut ist (aus
   `fachbereich_statistik()`) — wichtig, weil bei enger Streuung schon
   ein kleiner Unterschied wie ein großer Prozentsprung wirkt.
3. Wie viele Indikatoren diese Klinik in diesem Fachbereich gemeldet hat,
   verglichen mit der typischen (Median-)Anzahl.

Hat eine Klinik in einem Fachbereich gar keine Daten, wird die Kachel
grau und zeigt nur "–".

**Tabelle "Struktur im Vergleich":**
Zeigt alle 12 Struktur-Merkmale für: diese Klinik, den Durchschnitt ihres
Clusters, den Gesamtdurchschnitt. Der Wert dieser Klinik wird grün, wenn
er mehr als 2 Prozentpunkte über dem Gesamtdurchschnitt liegt, rot, wenn
er mehr als 2 Prozentpunkte darunter liegt, sonst grau.

---

## 10. Hauptfunktion `main()` (Zeilen 908–949)

**Was passiert hier?**
Das ist der Startpunkt des ganzen Programms. Reihenfolge:
1. Seiteneinstellungen setzen (breiter Bildschirm, Seitentitel).
2. Das CSS von oben einfügen.
3. Prüfen, ob die Datendatei überhaupt existiert — falls nicht, eine
   Fehlermeldung zeigen und aufhören.
4. Die Daten laden.
5. Merken, welche Seite gerade aktiv ist (Standard: "Übersicht").
6. Die Navigation in der Seitenleiste zeichnen (die drei Seiten-Knöpfe).
7. Falls man **nicht** auf der Klinik-Profil-Seite ist: die Filter-Knöpfe
   zeichnen und die aktuelle Anzahl je Cluster berechnen.
8. Je nachdem, welche Seite ausgewählt ist, die passende Funktion
   aufrufen (`seite_uebersicht`, `seite_cluster_analyse` oder
   `seite_profil`).

**Warum sind die Filter auf der Klinik-Profil-Seite deaktiviert?**
Weil man dort ohnehin nach einer einzelnen, konkreten Klinik sucht — ein
Cluster- oder Größenfilter würde hier nichts sinnvoll einschränken.

---

## Gefundene Warnungen (keine echten Fehler)

Beim Testen sind folgende **harmlose** Meldungen aufgetaucht:

1. **`use_container_width` ist veraltet.** Streamlit plant, diesen Namen
   nach 2025 durch `width='stretch'` zu ersetzen. Funktioniert aktuell
   noch einwandfrei — muss aber irgendwann einmal überall ersetzt werden.
2. **Leeres `label` bei `st.radio`.** Die Navigation nutzt bewusst ein
   leeres Label (der Titel steht schon separat darüber). Streamlit
   empfiehlt aus Gründen der Barrierefreiheit ein sichtbares Label — hier
   bewusst weggelassen und mit `label_visibility="collapsed"` versteckt.
3. **"Pandas DataFrame hash failed."** Beim Zwischenspeichern
   (`@st.cache_data`) einer Tabelle mit einer Listen-Spalte
   (`fachbereiche_liste`) kann Streamlit diese Spalte nicht direkt
   vergleichen und weicht auf eine langsamere Methode aus (Pickling).
   Das Programm funktioniert trotzdem korrekt — es ist nur minimal
   langsamer beim ersten Laden.

Keine dieser drei Meldungen führt zu einem Absturz oder einem falschen
Ergebnis.
