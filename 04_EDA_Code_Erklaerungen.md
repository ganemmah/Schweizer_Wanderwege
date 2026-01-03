# Code-Erklärungen: Exploratory Data Analysis (EDA) - Schweizer Wanderwege

**Zweck dieses Dokuments:** Detaillierte Erklärung jeder Codezeile für Dokumentation und Video-Präsentation

---

## 1. Setup & Daten laden

### 1.1 Library Imports

```python
import pandas as pd
```
**Erklärung:** Importiert die Pandas-Bibliothek unter dem Alias "pd". Pandas ist die wichtigste Bibliothek für Datenanalyse in Python und ermöglicht das Arbeiten mit tabellarischen Daten (DataFrames).

```python
import numpy as np
```
**Erklärung:** Importiert NumPy unter dem Alias "np". NumPy wird für numerische Berechnungen verwendet, z.B. für mathematische Operationen und Arrays.

```python
import matplotlib.pyplot as plt
```
**Erklärung:** Importiert das pyplot-Modul von Matplotlib als "plt". Matplotlib ist die Standard-Bibliothek für Visualisierungen in Python. Mit pyplot können wir Diagramme erstellen.

```python
import seaborn as sns
```
**Erklärung:** Importiert Seaborn als "sns". Seaborn baut auf Matplotlib auf und bietet schönere, statistische Visualisierungen mit weniger Code.

```python
from sqlalchemy import create_engine
```
**Erklärung:** Importiert die Funktion `create_engine` aus SQLAlchemy. Diese Funktion ermöglicht die Verbindung zu Datenbanken (in unserem Fall MySQL).

---

### 1.2 Visualization Settings

```python
plt.style.use('seaborn-v0_8-darkgrid')
```
**Erklärung:** Setzt den globalen Stil für alle Matplotlib-Plots auf "seaborn-v0_8-darkgrid". Dies gibt den Diagrammen ein professionelles Aussehen mit Gitternetzlinien im Hintergrund.

```python
sns.set_palette('Set2')
```
**Erklärung:** Definiert die Standard-Farbpalette für Seaborn-Plots. "Set2" ist eine vordefinierte Farbpalette mit harmonischen, gut unterscheidbaren Farben.

```python
plt.rcParams['figure.figsize'] = (12, 6)
```
**Erklärung:** Setzt die Standard-Grösse für alle Diagramme auf 12 Zoll Breite und 6 Zoll Höhe. Dies macht die Plots gross genug, um Details zu erkennen.

```python
plt.rcParams['font.size'] = 11
```
**Erklärung:** Setzt die Standard-Schriftgrösse für alle Text-Elemente in Diagrammen auf 11 Punkte.

---

### 1.3 Display Settings

```python
pd.set_option('display.max_columns', None)
```
**Erklärung:** Konfiguriert Pandas so, dass ALLE Spalten eines DataFrames angezeigt werden (nicht nur eine Auswahl). `None` bedeutet "keine Begrenzung".

```python
pd.set_option('display.width', 150)
```
**Erklärung:** Setzt die maximale Breite der Ausgabe auf 150 Zeichen. Dies verhindert, dass breite Tabellen umbrechen.

```python
pd.set_option('display.precision', 2)
```
**Erklärung:** Zeigt numerische Werte mit 2 Dezimalstellen an (z.B. 3.14 statt 3.14159265).

---

### 1.4 Warnings unterdrücken

```python
import warnings
warnings.filterwarnings("ignore")
```
**Erklärung:** Importiert das warnings-Modul und unterdrückt alle Warnmeldungen. Dies macht die Ausgabe sauberer, sollte aber nur in finalen Notebooks verwendet werden (während der Entwicklung sind Warnungen hilfreich!).

```python
print("Libraries imported successfully")
```
**Erklärung:** Gibt eine Bestätigungsmeldung aus, dass alle Bibliotheken erfolgreich geladen wurden.

---

### 1.5 Datenbank-Verbindung

```python
DB_USER = "root"
```
**Erklärung:** Definiert den Benutzernamen für die MySQL-Datenbank-Verbindung als Variable. "root" ist der Standard-Administrator-Account.

```python
DB_PASSWORD = "password"
```
**Erklärung:** Definiert das Passwort für die Datenbank-Verbindung. In Produktionsumgebungen sollte dies NIEMALS im Code stehen, sondern aus Umgebungsvariablen gelesen werden!

```python
DB_HOST = "localhost"
```
**Erklärung:** Definiert den Server, auf dem die Datenbank läuft. "localhost" bedeutet, dass die Datenbank auf dem gleichen Computer läuft (über Docker).

```python
DB_PORT = "3306"
```
**Erklärung:** Definiert den Port, über den MySQL erreichbar ist. 3306 ist der Standard-Port für MySQL-Datenbanken.

```python
DB_NAME = "wanderwege_db"
```
**Erklärung:** Definiert den Namen der Datenbank, mit der wir uns verbinden wollen.

```python
connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
```
**Erklärung:** Erstellt einen Connection String (Verbindungszeichenkette) im Format: `mysql+pymysql://user:password@host:port/database`. Der f-String (f"...") ermöglicht das Einfügen von Variablen mit {}.

**Format-Aufschlüsselung:**
- `mysql+pymysql://` = Datenbank-Typ und Treiber
- `root:password@` = Benutzername:Passwort
- `localhost:3306` = Server:Port
- `/wanderwege_db` = Datenbankname

```python
engine = create_engine(connection_string)
```
**Erklärung:** Erstellt eine SQLAlchemy-Engine (Verbindungsobjekt) zur MySQL-Datenbank. Diese Engine wird später verwendet, um SQL-Abfragen auszuführen.

---

### 1.6 Daten laden

```python
query = "SELECT * FROM wanderwege"
```
**Erklärung:** Definiert eine SQL-Abfrage als String. "SELECT * FROM wanderwege" bedeutet: "Wähle ALLE Spalten (*) aus der Tabelle 'wanderwege' aus".

```python
df = pd.read_sql(query, con=engine)
```
**Erklärung:** Führt die SQL-Abfrage aus und lädt das Ergebnis in einen Pandas DataFrame namens "df". 
- `query` = die SQL-Abfrage
- `con=engine` = die Datenbankverbindung

```python
print(f"Loaded {len(df)} hiking routes from MySQL database")
```
**Erklärung:** Gibt die Anzahl der geladenen Zeilen aus. `len(df)` zählt die Anzahl der Zeilen im DataFrame.

```python
print(f"Columns: {df.shape[1]}")
```
**Erklärung:** Gibt die Anzahl der Spalten aus. `df.shape` gibt ein Tupel (Zeilen, Spalten) zurück, `df.shape[1]` greift auf die Spalten-Anzahl zu.

```python
df.head()
```
**Erklärung:** Zeigt die ersten 5 Zeilen des DataFrames an. Dies gibt einen schnellen Überblick über die Datenstruktur.

---

## 2. Non-Graphical EDA

### 2.1 Deskriptive Statistiken

```python
numeric_cols = ['duration_min', 'distance_km', 'ascent_m', 'descent_m']
```
**Erklärung:** Erstellt eine Liste mit den Namen aller numerischen Spalten. Dies vermeidet Code-Wiederholungen und macht den Code wartbarer.

```python
print("="*80)
```
**Erklärung:** Gibt 80 Gleichheitszeichen aus, um eine visuelle Trennlinie zu erstellen. Der `*` Operator wiederholt einen String.

```python
print("DESKRIPTIVE STATISTIKEN - Numerische Variablen")
print("="*80)
```
**Erklärung:** Gibt eine Überschrift mit Trennlinien oben und unten aus.

```python
desc_stats = df[numeric_cols].describe().T
```
**Erklärung:** 
- `df[numeric_cols]` = Wählt nur die numerischen Spalten aus
- `.describe()` = Berechnet deskriptive Statistiken (count, mean, std, min, 25%, 50%, 75%, max)
- `.T` = Transponiert die Tabelle (dreht Zeilen und Spalten um) für bessere Lesbarkeit

```python
desc_stats['median'] = df[numeric_cols].median()
```
**Erklärung:** Fügt eine neue Spalte "median" zur Statistik-Tabelle hinzu. `.median()` berechnet den Median (Mittelwert der sortierten Daten) für jede numerische Spalte.

```python
desc_stats = desc_stats[['count', 'mean', 'median', 'std', 'min', 'max']]
```
**Erklärung:** Wählt nur die wichtigsten Statistiken aus und ordnet sie in einer bestimmten Reihenfolge an. Dies macht die Tabelle übersichtlicher.

```python
print(desc_stats)
```
**Erklärung:** Gibt die Statistik-Tabelle aus.

```python
print("\n")
```
**Erklärung:** Gibt eine Leerzeile aus. `\n` ist das Zeichen für einen Zeilenumbruch (newline).

```python
print("INTERPRETATION:")
```
**Erklärung:** Gibt eine Überschrift für die Interpretation aus.

```python
print(f"- Durchschnittliche Wanderdauer: {df['duration_min'].mean():.0f} Minuten ({df['duration_min'].mean()/60:.1f} Stunden)")
```
**Erklärung:** Berechnet und gibt die durchschnittliche Wanderdauer aus.
- `df['duration_min'].mean()` = Berechnet den Durchschnitt der Dauer-Spalte
- `:.0f` = Formatierung: 0 Dezimalstellen, f=floating point number
- `/60` = Umrechnung von Minuten in Stunden
- `:.1f` = Formatierung: 1 Dezimalstelle

```python
print(f"- Durchschnittliche Distanz: {df['distance_km'].mean():.1f} km")
```
**Erklärung:** Berechnet und gibt die durchschnittliche Distanz mit 1 Dezimalstelle aus.

```python
print(f"- Durchschnittlicher Aufstieg: {df['ascent_m'].mean():.0f} m")
```
**Erklärung:** Berechnet und gibt den durchschnittlichen Aufstieg ohne Dezimalstellen aus.

```python
print(f"- Durchschnittlicher Abstieg: {df['descent_m'].mean():.0f} m")
```
**Erklärung:** Berechnet und gibt den durchschnittlichen Abstieg ohne Dezimalstellen aus.

---

### 2.2 Top 5 Kantone

```python
canton_counts = df['canton'].value_counts().head(5)
```
**Erklärung:** 
- `df['canton']` = Wählt die Kanton-Spalte aus
- `.value_counts()` = Zählt, wie oft jeder Kanton vorkommt (erstellt eine Häufigkeitstabelle)
- `.head(5)` = Nimmt nur die ersten 5 (die häufigsten)

```python
canton_percentage = (canton_counts / len(df) * 100).round(1)
```
**Erklärung:** Berechnet die Prozentsätze:
- `canton_counts / len(df)` = Teilt jede Anzahl durch die Gesamtzahl (ergibt Anteil 0-1)
- `* 100` = Umrechnung in Prozent (0-100)
- `.round(1)` = Rundet auf 1 Dezimalstelle

```python
canton_df = pd.DataFrame({
    'Anzahl Wanderwege': canton_counts,
    'Prozent (%)': canton_percentage
})
```
**Erklärung:** Erstellt einen neuen DataFrame mit zwei Spalten aus den berechneten Daten. Das Dictionary definiert Spaltennamen (keys) und Werte (values).

```python
print(canton_df)
```
**Erklärung:** Gibt die Tabelle mit Top 5 Kantonen aus.

```python
print(f"\n Die Top 5 Kantone repräsentieren {canton_percentage.sum():.1f}% aller Wanderwege")
```
**Erklärung:** Berechnet und gibt aus, wie viel Prozent der Wanderwege in den Top 5 Kantonen liegen.
- `canton_percentage.sum()` = Addiert alle 5 Prozentsätze
- `:.1f` = 1 Dezimalstelle

---

### 2.3 Korrelationsmatrix

```python
correlation_matrix = df[numeric_cols].corr()
```
**Erklärung:** 
- Wählt nur numerische Spalten aus
- `.corr()` = Berechnet die Pearson-Korrelationskoeffizienten zwischen allen Spaltenpaaren
- Ergebnis: Eine Matrix mit Werten zwischen -1 (perfekte negative Korrelation) und +1 (perfekte positive Korrelation)

```python
print(correlation_matrix.round(3))
```
**Erklärung:** Gibt die Korrelationsmatrix mit 3 Dezimalstellen aus.

```python
corr_pairs = []
```
**Erklärung:** Erstellt eine leere Liste, um Korrelations-Paare zu speichern.

```python
for i in range(len(correlation_matrix.columns)):
```
**Erklärung:** Startet eine Schleife über alle Spalten-Indizes (0, 1, 2, ...).

```python
    for j in range(i+1, len(correlation_matrix.columns)):
```
**Erklärung:** Innere Schleife, die nur über Spalten NACH der aktuellen Spalte iteriert. Dies vermeidet Duplikate (z.B. beide: A<->B und B<->A).

```python
        corr_pairs.append((
            correlation_matrix.columns[i],
            correlation_matrix.columns[j],
            correlation_matrix.iloc[i, j]
        ))
```
**Erklärung:** Fügt ein Tupel zur Liste hinzu mit:
- Spaltenname i
- Spaltenname j  
- Korrelationswert zwischen i und j
`.iloc[i, j]` greift auf die Zelle an Position (Zeile i, Spalte j) zu.

```python
corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
```
**Erklärung:** Sortiert die Korrelations-Paare:
- `sorted()` = Sortier-Funktion
- `key=lambda x: abs(x[2])` = Sortierkriterium: absoluter Wert des 3. Elements (Korrelation)
- `reverse=True` = Absteigende Sortierung (höchste zuerst)

```python
print("\nStärkste Korrelationen:")
for var1, var2, corr_val in corr_pairs_sorted[:3]:
    print(f"- {var1} <-> {var2}: {corr_val:.3f}")
```
**Erklärung:** 
- `corr_pairs_sorted[:3]` = Nimmt die ersten 3 Paare
- Schleife entpackt jedes Tupel in var1, var2, corr_val
- Gibt jedes Paar mit 3 Dezimalstellen aus
- `<->` ist ein Unicode-Zeichen für Doppelpfeil

---

### 2.4 Missing Values Check

```python
missing_data = pd.DataFrame({
    'Anzahl Missing': df.isnull().sum(),
    'Prozent (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
```
**Erklärung:** Erstellt einen DataFrame mit Missing-Value-Statistiken:
- `df.isnull()` = Erstellt Boolean-Matrix (True bei fehlenden Werten)
- `.sum()` = Zählt True-Werte pro Spalte
- Division und Multiplikation berechnet Prozentsätze

```python
missing_data = missing_data[missing_data['Anzahl Missing'] > 0].sort_values('Anzahl Missing', ascending=False)
```
**Erklärung:** 
- `missing_data['Anzahl Missing'] > 0` = Filtert nur Spalten MIT fehlenden Werten
- `.sort_values()` = Sortiert nach Anzahl, absteigend

```python
if len(missing_data) > 0:
    print(missing_data)
    print(f"\nSpalten mit fehlenden Werten: {len(missing_data)}")
else:
    print("Keine fehlenden Werte gefunden!")
```
**Erklärung:** If-Else-Statement:
- WENN es fehlende Werte gibt → zeige Tabelle
- SONST → zeige Erfolgsmeldung

---

### 2.5 Crosstab

```python
crosstab = pd.crosstab(df['difficulty_level'], df['physical_demand'], margins=True)
```
**Erklärung:** Erstellt eine Kreuztabelle (Kontingenztafel):
- Zeilen = Schwierigkeitsgrad
- Spalten = Physische Anforderung
- Zellen = Anzahl Wanderungen mit dieser Kombination
- `margins=True` = Fügt Zeilen- und Spaltensummen hinzu

```python
print(crosstab)
```
**Erklärung:** Gibt die Kreuztabelle aus.

```python
crosstab_pct = pd.crosstab(df['difficulty_level'], df['physical_demand'], normalize='index') * 100
```
**Erklärung:** Erstellt prozentuale Kreuztabelle:
- `normalize='index'` = Normalisiert zeilenweise (jede Zeile summiert zu 100%)
- `* 100` = Umrechnung in Prozent

```python
print(crosstab_pct.round(1))
```
**Erklärung:** Gibt die prozentuale Tabelle mit 1 Dezimalstelle aus.

---

## 3. Graphical EDA

### 3.1 Histogramm: Wanderdauer

```python
plt.figure(figsize=(12, 5))
```
**Erklärung:** Erstellt eine neue Figur (Zeichenfläche) mit Breite 12 und Höhe 5 Zoll.

```python
plt.subplot(1, 2, 1)
```
**Erklärung:** Erstellt ein Subplot-Raster und aktiviert das erste Subplot:
- `1` = 1 Zeile
- `2` = 2 Spalten
- `1` = Aktiviere Position 1 (links)

```python
plt.hist(df['duration_min'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
```
**Erklärung:** Erstellt ein Histogramm:
- `df['duration_min']` = Daten
- `bins=30` = Teilt Daten in 30 Balken auf
- `color='skyblue'` = Füllfarbe
- `edgecolor='black'` = Randfarbe der Balken
- `alpha=0.7` = Transparenz (0=transparent, 1=undurchsichtig)

```python
plt.axvline(df['duration_min'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["duration_min"].mean():.0f} min')
```
**Erklärung:** Zeichnet eine vertikale Linie beim Mittelwert:
- `df['duration_min'].mean()` = x-Position (Mittelwert)
- `color='red'` = Farbe
- `linestyle='--'` = Gestrichelte Linie
- `linewidth=2` = Liniendicke
- `label=...` = Beschriftung für Legende

```python
plt.axvline(df['duration_min'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["duration_min"].median():.0f} min')
```
**Erklärung:** Zeichnet eine grüne vertikale Linie beim Median.

```python
plt.xlabel('Dauer (Minuten)', fontsize=12)
```
**Erklärung:** Setzt die Beschriftung der x-Achse mit Schriftgrösse 12.

```python
plt.ylabel('Häufigkeit', fontsize=12)
```
**Erklärung:** Setzt die Beschriftung der y-Achse.

```python
plt.title('Verteilung der Wanderdauer', fontsize=14, fontweight='bold')
```
**Erklärung:** Setzt den Titel:
- `fontsize=14` = Grösser als normale Schrift
- `fontweight='bold'` = Fettdruck

```python
plt.legend()
```
**Erklärung:** Zeigt die Legende an (mit den labels von axvline).

```python
plt.grid(axis='y', alpha=0.3)
```
**Erklärung:** Fügt Gitternetzlinien hinzu:
- `axis='y'` = Nur horizontale Linien
- `alpha=0.3` = Leicht transparent (dezent)

```python
plt.subplot(1, 2, 2)
```
**Erklärung:** Aktiviert das zweite Subplot (rechts).

```python
plt.hist(df['duration_min']/60, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
```
**Erklärung:** Erstellt zweites Histogramm mit Dauer in STUNDEN (durch 60 geteilt).

```python
plt.tight_layout()
```
**Erklärung:** Optimiert automatisch die Abstände zwischen Subplots, damit sich nichts überlappt.

```python
plt.show()
```
**Erklärung:** Zeigt die fertige Grafik an.

```python
print(f"Die meisten Wanderungen dauern zwischen {df['duration_min'].quantile(0.25)/60:.1f} und {df['duration_min'].quantile(0.75)/60:.1f} Stunden.")
```
**Erklärung:** Berechnet und gibt das Interquartil-Range aus:
- `.quantile(0.25)` = 25%-Percentil (unteres Quartil)
- `.quantile(0.75)` = 75%-Percentil (oberes Quartil)
- Umrechnung in Stunden

---

### 3.2 Boxplot: Distanz nach Schwierigkeitsgrad

```python
plt.figure(figsize=(12, 6))
```
**Erklärung:** Neue Figur erstellen.

```python
difficulty_order = sorted(df['difficulty_level'].unique())
```
**Erklärung:** 
- `df['difficulty_level'].unique()` = Alle eindeutigen Schwierigkeitsgrade
- `sorted()` = Sortiert sie alphabetisch (T1, T2, T3, ...)

```python
sns.boxplot(data=df, x='difficulty_level', y='distance_km', order=difficulty_order, palette='Set2')
```
**Erklärung:** Erstellt einen Seaborn-Boxplot:
- `data=df` = Datenquelle
- `x='difficulty_level'` = Kategorische Variable auf x-Achse
- `y='distance_km'` = Numerische Variable auf y-Achse
- `order=difficulty_order` = Sortierreihenfolge
- `palette='Set2'` = Farbschema

```python
medians = df.groupby('difficulty_level')['distance_km'].median().reindex(difficulty_order)
```
**Erklärung:** Berechnet Mediane für jede Schwierigkeitsstufe:
- `groupby('difficulty_level')` = Gruppiert nach Schwierigkeit
- `['distance_km'].median()` = Berechnet Median der Distanz pro Gruppe
- `.reindex(difficulty_order)` = Sortiert in gewünschter Reihenfolge

```python
for i, (level, median) in enumerate(medians.items()):
    plt.text(i, median, f'{median:.1f} km', ha='center', va='bottom', fontweight='bold', fontsize=10)
```
**Erklärung:** Schleife über alle Mediane:
- `enumerate(medians.items())` = Gibt Index und (level, median)-Paare zurück
- `plt.text(i, median, ...)` = Platziert Text an Position (x=i, y=median)
- `ha='center'` = horizontal alignment (Textausrichtung)
- `va='bottom'` = vertical alignment (Text über dem Punkt)

---

### 3.3 Scatterplot: Distanz vs. Dauer

```python
difficulty_colors = {level: i for i, level in enumerate(sorted(df['difficulty_level'].unique()))}
```
**Erklärung:** Dictionary Comprehension - erstellt ein Mapping:
- Schwierigkeitsgrad → Zahl (für Farbcodierung)
- Beispiel: {'T1': 0, 'T2': 1, 'T3': 2, ...}

```python
colors = df['difficulty_level'].map(difficulty_colors)
```
**Erklärung:** Wandelt jeden Schwierigkeitsgrad in eine Zahl um (für Farben):
- `.map()` = Ersetzt jeden Wert durch den entsprechenden Wert aus dem Dictionary

```python
scatter = plt.scatter(df['distance_km'], df['duration_min'], c=colors, cmap='viridis',
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
```
**Erklärung:** Erstellt Scatterplot:
- `df['distance_km']` = x-Werte
- `df['duration_min']` = y-Werte
- `c=colors` = Farben basierend auf Schwierigkeit
- `cmap='viridis'` = Colormap (Farbschema)
- `s=50` = Punktgrösse
- `edgecolors='black'` = Schwarzer Rand um jeden Punkt

```python
cbar = plt.colorbar(scatter, ticks=list(difficulty_colors.values()))
```
**Erklärung:** Fügt eine Farbskala (Colorbar) hinzu:
- `ticks=...` = Positioniert Beschriftungen bei den Zahlen

```python
cbar.ax.set_yticklabels(list(difficulty_colors.keys()))
```
**Erklärung:** Ersetzt die Zahlen in der Colorbar durch die tatsächlichen Schwierigkeitsgrade (T1, T2, ...).

```python
cbar.set_label('Schwierigkeitsgrad', rotation=270, labelpad=20, fontsize=11)
```
**Erklärung:** Beschriftet die Colorbar:
- `rotation=270` = Dreht Text um 270° (vertikal, lesbar von rechts)
- `labelpad=20` = Abstand zwischen Label und Colorbar

```python
z = np.polyfit(df['distance_km'].dropna(), df.loc[df['distance_km'].notna(), 'duration_min'], 1)
```
**Erklärung:** Berechnet lineare Regression (Trendlinie):
- `np.polyfit()` = Findet beste Polynomfunktion
- `.dropna()` = Entfernt NaN-Werte
- `1` = Grad 1 (linear: y = mx + b)
- Ergebnis: Array [m, b] (Steigung und Y-Achsenabschnitt)

```python
p = np.poly1d(z)
```
**Erklärung:** Erstellt ein Polynom-Objekt aus den Koeffizienten, das wie eine Funktion aufgerufen werden kann.

```python
plt.plot(df['distance_km'].sort_values(), p(df['distance_km'].sort_values()),
         "r--", linewidth=2, alpha=0.8, label=f'Trendlinie: y={z[0]:.1f}x+{z[1]:.1f}')
```
**Erklärung:** Zeichnet die Trendlinie:
- `.sort_values()` = Sortiert x-Werte für glatte Linie
- `p(...)` = Berechnet y-Werte mit der Polynomfunktion
- `"r--"` = Rot, gestrichelt
- `label=...` = Zeigt die Gleichung in der Legende

---

### 3.4 Korrelations-Heatmap

```python
plt.figure(figsize=(10, 8))
```
**Erklärung:** Quadratische Figur für Heatmap.

```python
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
```
**Erklärung:** Erstellt eine Heatmap:
- `annot=True` = Zeigt Zahlen in den Zellen
- `fmt='.3f'` = Format: 3 Dezimalstellen
- `cmap='coolwarm'` = Farbschema (blau=negativ, rot=positiv)
- `center=0` = 0 wird weiss dargestellt
- `square=True` = Quadratische Zellen
- `linewidths=1` = Trennlinien zwischen Zellen
- `cbar_kws={"shrink": 0.8}` = Colorbar auf 80% Höhe

---

### 3.5 Barplot: Top 5 Kantone

```python
top5_cantons = df['canton'].value_counts().head(5)
```
**Erklärung:** Wie vorher - zählt Kantone und nimmt Top 5.

```python
bars = plt.bar(range(len(top5_cantons)), top5_cantons.values,
               color='steelblue', edgecolor='black', alpha=0.8)
```
**Erklärung:** Erstellt Balkendiagramm:
- `range(len(top5_cantons))` = x-Positionen: [0, 1, 2, 3, 4]
- `top5_cantons.values` = Höhen der Balken
- Speichert Balken-Objekte in `bars` für spätere Referenz

```python
plt.xticks(range(len(top5_cantons)), top5_cantons.index, fontsize=12)
```
**Erklärung:** Setzt x-Achsen-Beschriftungen:
- `range(len(top5_cantons))` = Positionen
- `top5_cantons.index` = Kantonnamen

```python
for i, (bar, value) in enumerate(zip(bars, top5_cantons.values)):
```
**Erklärung:** Schleife über Balken und Werte:
- `zip(bars, top5_cantons.values)` = Kombiniert Listen paarweise
- `enumerate()` = Gibt zusätzlich den Index

```python
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
```
**Erklärung:** Platziert Wert über jedem Balken:
- `bar.get_x() + bar.get_width()/2` = Mitte des Balkens (x)
- `bar.get_height() + 2` = Knapp über dem Balken (y)

---

### 3.6 Stacked Barplot

```python
top5_cantons = df['canton'].value_counts().head(5).index
```
**Erklärung:** Diesmal nur die NAMEN (index), nicht die Anzahlen.

```python
df_top5 = df[df['canton'].isin(top5_cantons)]
```
**Erklärung:** Filtert DataFrame auf nur Top 5 Kantone:
- `.isin(top5_cantons)` = Prüft ob Kanton in der Liste ist
- Ergebnis: Boolean-Maske für Filterung

```python
crosstab_difficulty = pd.crosstab(df_top5['canton'], df_top5['difficulty_level'])
```
**Erklärung:** Kreuztabelle für die Top 5 Kantone:
- Zeilen = Kantone
- Spalten = Schwierigkeitsgrade
- Werte = Anzahl

```python
difficulty_cols = sorted([col for col in crosstab_difficulty.columns])
```
**Erklärung:** List Comprehension - sortiert die Spaltennamen.

```python
crosstab_difficulty = crosstab_difficulty[difficulty_cols]
```
**Erklärung:** Ordnet die Spalten in sortierter Reihenfolge an.

```python
crosstab_difficulty.plot(kind='bar', stacked=True, figsize=(12, 6),
                         colormap='viridis', edgecolor='black', linewidth=0.5)
```
**Erklärung:** Erstellt gestapeltes Balkendiagramm:
- `kind='bar'` = Balkendiagramm
- `stacked=True` = Balken werden übereinander gestapelt
- `colormap='viridis'` = Verschiedene Farben für Schwierigkeitsgrade

```python
plt.legend(title='Schwierigkeitsgrad', bbox_to_anchor=(1.05, 1), loc='upper left')
```
**Erklärung:** Positioniert Legende:
- `title=...` = Titel der Legende
- `bbox_to_anchor=(1.05, 1)` = Position ausserhalb des Plots (rechts oben)
- `loc='upper left'` = Ankerpunkt der Legende-Box

```python
plt.xticks(rotation=0)
```
**Erklärung:** Dreht x-Achsen-Beschriftungen auf 0° (horizontal).

---

## 4. Zusammenfassung

```python
print("="*80)
print("ZUSAMMENFASSUNG - Wichtigste Erkenntnisse aus der EDA")
print("="*80)
```
**Erklärung:** Überschrift mit Trennlinien.

```python
print()
```
**Erklärung:** Leerzeile.

```python
print("DATENÜBERSICHT:")
print(f"- Anzahl Wanderwege: {len(df)}")
```
**Erklärung:** Gibt die Gesamtzahl der Wanderwege aus. Die Leerzeichen am Anfang erzeugen Einrückung.

```python
print(f"- Anzahl Kantone: {df['canton'].nunique()}")
```
**Erklärung:** `.nunique()` = Zählt die Anzahl eindeutiger (unique) Werte.

```python
print(f"- Schwierigkeitsgrade: {sorted(df['difficulty_level'].unique())}")
```
**Erklärung:** Zeigt alle vorhandenen Schwierigkeitsgrade sortiert an.

```python
print(f"- Stärkste Korrelation: {corr_pairs_sorted[0][0]} <-> {corr_pairs_sorted[0][1]} ({corr_pairs_sorted[0][2]:.3f})")
```
**Erklärung:** Greift auf das erste Element der sortierten Korrelations-Liste zu:
- `[0]` = Erstes Tupel
- `[0]`, `[1]`, `[2]` = Erstes, zweites, drittes Element des Tupels

```python
print(f"- Durchschnittliche Wanderzeit: {df['duration_min'].mean()/60:.1f} Stunden")
```
**Erklärung:** Berechnet Durchschnitt und rechnet in Stunden um.

```python
print(f"- Max. Aufstieg: {df['ascent_m'].max():.0f} m")
```
**Erklärung:** `.max()` = Findet den Maximalwert der Spalte.

```python
print(f"- Korrelation Aufstieg <-> Abstieg: {df[['ascent_m', 'descent_m']].corr().iloc[0,1]:.3f}")
```
**Erklärung:** 
- `df[['ascent_m', 'descent_m']]` = Wählt beide Spalten aus
- `.corr()` = Berechnet Korrelationsmatrix (2x2)
- `.iloc[0,1]` = Greift auf Zelle Zeile 0, Spalte 1 zu (Korrelation zwischen den beiden)

```python
print(f"- Top 5 Kantone: {', '.join(top5_cantons.tolist())}")
```
**Erklärung:** 
- `.tolist()` = Wandelt Series in Python-Liste um
- `', '.join(...)` = Verbindet Liste zu einem String mit Kommas

```python
print(f"- Führender Kanton: {top5_cantons[0]} ({canton_counts.iloc[0]} Wanderwege)")
```
**Erklärung:** 
- `top5_cantons[0]` = Erster Kanton
- `canton_counts.iloc[0]` = Anzahl des ersten Kantons

---

## Footer: System-Information

```python
import os
import platform
import socket
from platform import python_version
from datetime import datetime
```
**Erklärung:** Importiert Module für System-Informationen:
- `os` = Betriebssystem-Funktionen
- `platform` = Plattform-Informationen
- `socket` = Netzwerk-Funktionen (hier nicht verwendet)
- `python_version` = Funktion für Python-Version
- `datetime` = Datum/Zeit-Funktionen

```python
print('-----------------------------------')
```
**Erklärung:** Trennlinie.

```python
print(os.name.upper())
```
**Erklärung:** 
- `os.name` = Name des Betriebssystems ('nt' für Windows, 'posix' für Linux/Mac)
- `.upper()` = Wandelt in Grossbuchstaben um

```python
print(platform.system(), '|', platform.release())
```
**Erklärung:** 
- `platform.system()` = Betriebssystem-Name (z.B. "Windows", "Linux")
- `platform.release()` = Version/Release (z.B. "10", "11")
- `'|'` = Trennzeichen

```python
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
```
**Erklärung:** 
- `datetime.now()` = Aktuelles Datum und Zeit
- `.strftime()` = Formatiert Datum als String
- `"%Y-%m-%d %H:%M:%S"` = Format: Jahr-Monat-Tag Stunde:Minute:Sekunde

```python
print('Python Version:', python_version())
```
**Erklärung:** Gibt die installierte Python-Version aus (z.B. "3.11.5").

---

## Zusammenfassung der wichtigsten Konzepte:

### Pandas-Operationen:
- `.head()` = Erste Zeilen
- `.describe()` = Statistiken
- `.value_counts()` = Häufigkeiten
- `.groupby()` = Gruppierung
- `.corr()` = Korrelation
- `.isnull()` = Fehlende Werte
- `.unique()` = Eindeutige Werte
- `.nunique()` = Anzahl eindeutiger Werte

### Matplotlib/Seaborn:
- `plt.figure()` = Neue Grafik
- `plt.subplot()` = Mehrere Plots in einer Figur
- `plt.hist()` = Histogramm
- `plt.scatter()` = Scatterplot
- `sns.boxplot()` = Boxplot
- `sns.heatmap()` = Heatmap
- `plt.xlabel/ylabel/title()` = Beschriftungen

### Python-Grundlagen:
- f-Strings: `f"Text {variable}"` = String-Formatierung
- List Comprehension: `[x for x in liste]` = Kompakte Listen-Erstellung
- Lambda: `lambda x: x**2` = Anonyme Funktion
- `.round()`, `.mean()`, `.median()` = Mathematische Operationen

---

**Ende der Code-Erklärungen**

