# Code Explanations: Exploratory Data Analysis (EDA) - Swiss Hiking Trails

**Purpose of this Document:** Detailed explanation of every line of code for documentation and video presentation

---

## 1. Setup & Load Data

### 1.1 Library Imports

```python
import pandas as pd
```
**Explanation:** Imports the Pandas library under the alias "pd". Pandas is the most important library for data analysis in Python and enables working with tabular data (DataFrames).

```python
import numpy as np
```
**Explanation:** Imports NumPy under the alias "np". NumPy is used for numerical calculations, e.g., for mathematical operations and arrays.

```python
import matplotlib.pyplot as plt
```
**Explanation:** Imports the pyplot module from Matplotlib as "plt". Matplotlib is the standard library for visualizations in Python. With pyplot we can create diagrams.

```python
import seaborn as sns
```
**Explanation:** Imports Seaborn as "sns". Seaborn builds on Matplotlib and offers prettier statistical visualizations with less code.

```python
from sqlalchemy import create_engine
```
**Explanation:** Imports the `create_engine` function from SQLAlchemy. This function enables connection to databases (in our case MySQL).

---

### 1.2 Visualization Settings

```python
plt.style.use('seaborn-v0_8-darkgrid')
```
**Explanation:** Sets the global style for all Matplotlib plots to "seaborn-v0_8-darkgrid". This gives diagrams a professional appearance with grid lines in the background.

```python
sns.set_palette('Set2')
```
**Explanation:** Defines the default color palette for Seaborn plots. "Set2" is a predefined color palette with harmonious, easily distinguishable colors.

```python
plt.rcParams['figure.figsize'] = (12, 6)
```
**Explanation:** Sets the default size for all diagrams to 12 inches width and 6 inches height. This makes the plots large enough to recognize details.

```python
plt.rcParams['font.size'] = 11
```
**Explanation:** Sets the default font size for all text elements in diagrams to 11 points.

---

### 1.3 Display Settings

```python
pd.set_option('display.max_columns', None)
```
**Explanation:** Configures Pandas so that ALL columns of a DataFrame are displayed (not just a selection). `None` means "no limit".

```python
pd.set_option('display.width', 150)
```
**Explanation:** Sets the maximum width of output to 150 characters. This prevents wide tables from wrapping.

```python
pd.set_option('display.precision', 2)
```
**Explanation:** Displays numeric values with 2 decimal places (e.g., 3.14 instead of 3.14159265).

---

### 1.4 Suppress Warnings

```python
import warnings
warnings.filterwarnings("ignore")
```
**Explanation:** Imports the warnings module and suppresses all warning messages. This makes the output cleaner, but should only be used in final notebooks (during development warnings are helpful!).

```python
print("Libraries imported successfully")
```
**Explanation:** Outputs a confirmation message that all libraries were loaded successfully.

---

### 1.5 Database Connection

```python
DB_USER = "root"
```
**Explanation:** Defines the username for the MySQL database connection as a variable. "root" is the default administrator account.

```python
DB_PASSWORD = "password"
```
**Explanation:** Defines the password for the database connection. In production environments this should NEVER be in the code, but read from environment variables!

```python
DB_HOST = "localhost"
```
**Explanation:** Defines the server where the database runs. "localhost" means the database runs on the same computer (via Docker).

```python
DB_PORT = "3306"
```
**Explanation:** Defines the port through which MySQL is accessible. 3306 is the standard port for MySQL databases.

```python
DB_NAME = "hiking_routes_db"
```
**Explanation:** Defines the name of the database we want to connect to.

```python
connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
```
**Explanation:** Creates a connection string in the format: `mysql+pymysql://user:password@host:port/database`. The f-string (f"...") enables inserting variables with {}.

**Format Breakdown:**
- `mysql+pymysql://` = Database type and driver
- `root:password@` = Username:Password
- `localhost:3306` = Server:Port
- `/hiking_routes_db` = Database name

```python
engine = create_engine(connection_string)
```
**Explanation:** Creates a SQLAlchemy engine (connection object) to the MySQL database. This engine will later be used to execute SQL queries.

---

### 1.6 Load Data

```python
query = "SELECT * FROM hiking_routes"
```
**Explanation:** Defines an SQL query as a string. "SELECT * FROM wanderwege" means: "Select ALL columns (*) from the table 'wanderwege'".

```python
df = pd.read_sql(query, con=engine)
```
**Explanation:** Executes the SQL query and loads the result into a Pandas DataFrame named "df". 
- `query` = the SQL query
- `con=engine` = the database connection

```python
print(f"Loaded {len(df)} hiking routes from MySQL database")
```
**Explanation:** Outputs the number of loaded rows. `len(df)` counts the number of rows in the DataFrame.

```python
print(f"Columns: {df.shape[1]}")
```
**Explanation:** Outputs the number of columns. `df.shape` returns a tuple (rows, columns), `df.shape[1]` accesses the column count.

```python
df.head()
```
**Explanation:** Displays the first 5 rows of the DataFrame. This gives a quick overview of the data structure.

---

## 2. Non-Graphical EDA

### 2.1 Descriptive Statistics

```python
numeric_cols = ['duration_min', 'distance_km', 'ascent_m', 'descent_m']
```
**Explanation:** Creates a list with the names of all numeric columns. This avoids code repetition and makes the code more maintainable.

```python
print("="*80)
```
**Explanation:** Outputs 80 equal signs to create a visual separator line. The `*` operator repeats a string.

```python
print("DESCRIPTIVE STATISTICS - Numeric Variables")
print("="*80)
```
**Explanation:** Outputs a heading with separator lines above and below.

```python
desc_stats = df[numeric_cols].describe().T
```
**Explanation:** 
- `df[numeric_cols]` = Selects only the numeric columns
- `.describe()` = Calculates descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
- `.T` = Transposes the table (rotates rows and columns) for better readability

```python
desc_stats['median'] = df[numeric_cols].median()
```
**Explanation:** Adds a new column "median" to the statistics table. `.median()` calculates the median (middle value of sorted data) for each numeric column.

```python
desc_stats = desc_stats[['count', 'mean', 'median', 'std', 'min', 'max']]
```
**Explanation:** Selects only the most important statistics and arranges them in a specific order. This makes the table more clear.

```python
print(desc_stats)
```
**Explanation:** Outputs the statistics table.

```python
print("\n")
```
**Explanation:** Outputs a blank line. `\n` is the character for a line break (newline).

```python
print("INTERPRETATION:")
```
**Explanation:** Outputs a heading for the interpretation.

```python
print(f"- Average hiking duration: {df['duration_min'].mean():.0f} minutes ({df['duration_min'].mean()/60:.1f} hours)")
```
**Explanation:** Calculates and outputs the average hiking duration.
- `df['duration_min'].mean()` = Calculates the average of the duration column
- `:.0f` = Formatting: 0 decimal places, f=floating point number
- `/60` = Conversion from minutes to hours
- `:.1f` = Formatting: 1 decimal place

```python
print(f"- Average distance: {df['distance_km'].mean():.1f} km")
```
**Explanation:** Calculates and outputs the average distance with 1 decimal place.

```python
print(f"- Average ascent: {df['ascent_m'].mean():.0f} m")
```
**Explanation:** Calculates and outputs the average ascent without decimal places.

```python
print(f"- Average descent: {df['descent_m'].mean():.0f} m")
```
**Explanation:** Calculates and outputs the average descent without decimal places.

---

### 2.2 Top 5 Cantons

```python
canton_counts = df['canton'].value_counts().head(5)
```
**Explanation:** 
- `df['canton']` = Selects the canton column
- `.value_counts()` = Counts how often each canton appears (creates a frequency table)
- `.head(5)` = Takes only the first 5 (the most frequent)

```python
canton_percentage = (canton_counts / len(df) * 100).round(1)
```
**Explanation:** Calculates the percentages:
- `canton_counts / len(df)` = Divides each count by the total (yields proportion 0-1)
- `* 100` = Conversion to percent (0-100)
- `.round(1)` = Rounds to 1 decimal place

```python
canton_df = pd.DataFrame({
    'Number of Trails': canton_counts,
    'Percent (%)': canton_percentage
})
```
**Explanation:** Creates a new DataFrame with two columns from the calculated data. The dictionary defines column names (keys) and values (values).

```python
print(canton_df)
```
**Explanation:** Outputs the table with Top 5 cantons.

```python
print(f"\n The Top 5 cantons represent {canton_percentage.sum():.1f}% of all hiking trails")
```
**Explanation:** Calculates and outputs how many percent of hiking trails are in the Top 5 cantons.
- `canton_percentage.sum()` = Adds all 5 percentages
- `:.1f` = 1 decimal place

---

### 2.3 Correlation Matrix

```python
correlation_matrix = df[numeric_cols].corr()
```
**Explanation:** 
- Selects only numeric columns
- `.corr()` = Calculates Pearson correlation coefficients between all column pairs
- Result: A matrix with values between -1 (perfect negative correlation) and +1 (perfect positive correlation)

```python
print(correlation_matrix.round(3))
```
**Explanation:** Outputs the correlation matrix with 3 decimal places.

```python
corr_pairs = []
```
**Explanation:** Creates an empty list to store correlation pairs.

```python
for i in range(len(correlation_matrix.columns)):
```
**Explanation:** Starts a loop over all column indices (0, 1, 2, ...).

```python
    for j in range(i+1, len(correlation_matrix.columns)):
```
**Explanation:** Inner loop that only iterates over columns AFTER the current column. This avoids duplicates (e.g., both: A<->B and B<->A).

```python
        corr_pairs.append((
            correlation_matrix.columns[i],
            correlation_matrix.columns[j],
            correlation_matrix.iloc[i, j]
        ))
```
**Explanation:** Adds a tuple to the list with:
- Column name i
- Column name j  
- Correlation value between i and j
`.iloc[i, j]` accesses the cell at position (row i, column j).

```python
corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
```
**Explanation:** Sorts the correlation pairs:
- `sorted()` = Sorting function
- `key=lambda x: abs(x[2])` = Sorting criterion: absolute value of the 3rd element (correlation)
- `reverse=True` = Descending sort (highest first)

```python
print("\nStrongest Correlations:")
for var1, var2, corr_val in corr_pairs_sorted[:3]:
    print(f"• {var1} <-> {var2}: {corr_val:.3f}")
```
**Explanation:** 
- `corr_pairs_sorted[:3]` = Takes the first 3 pairs
- Loop unpacks each tuple into var1, var2, corr_val
- Outputs each pair with 3 decimal places
- `<->` is a Unicode character for double arrow

---

### 2.4 Missing Values Check

```python
missing_data = pd.DataFrame({
    'Number Missing': df.isnull().sum(),
    'Percent (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
```
**Explanation:** Creates a DataFrame with missing value statistics:
- `df.isnull()` = Creates Boolean matrix (True for missing values)
- `.sum()` = Counts True values per column
- Division and multiplication calculates percentages

```python
missing_data = missing_data[missing_data['Number Missing'] > 0].sort_values('Number Missing', ascending=False)
```
**Explanation:** 
- `missing_data['Number Missing'] > 0` = Filters only columns WITH missing values
- `.sort_values()` = Sorts by count, descending

```python
if len(missing_data) > 0:
    print(missing_data)
    print(f"\nColumns with missing values: {len(missing_data)}")
else:
    print("No missing values found!")
```
**Explanation:** If-Else statement:
- IF there are missing values → show table
- ELSE → show success message

---

### 2.5 Crosstab

```python
crosstab = pd.crosstab(df['difficulty_level'], df['physical_demand'], margins=True)
```
**Explanation:** Creates a cross table (contingency table):
- Rows = Difficulty level
- Columns = Physical demand
- Cells = Number of hikes with this combination
- `margins=True` = Adds row and column totals

```python
print(crosstab)
```
**Explanation:** Outputs the cross table.

```python
crosstab_pct = pd.crosstab(df['difficulty_level'], df['physical_demand'], normalize='index') * 100
```
**Explanation:** Creates percentage cross table:
- `normalize='index'` = Normalizes row-wise (each row sums to 100%)
- `* 100` = Conversion to percent

```python
print(crosstab_pct.round(1))
```
**Explanation:** Outputs the percentage table with 1 decimal place.

---

## 3. Graphical EDA

### 3.1 Histogram: Hiking Duration

```python
plt.figure(figsize=(12, 5))
```
**Explanation:** Creates a new figure (canvas) with width 12 and height 5 inches.

```python
plt.subplot(1, 2, 1)
```
**Explanation:** Creates a subplot grid and activates the first subplot:
- `1` = 1 row
- `2` = 2 columns
- `1` = Activate position 1 (left)

```python
plt.hist(df['duration_min'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
```
**Explanation:** Creates a histogram:
- `df['duration_min']` = Data
- `bins=30` = Divides data into 30 bars
- `color='skyblue'` = Fill color
- `edgecolor='black'` = Border color of bars
- `alpha=0.7` = Transparency (0=transparent, 1=opaque)

```python
plt.axvline(df['duration_min'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["duration_min"].mean():.0f} min')
```
**Explanation:** Draws a vertical line at the mean:
- `df['duration_min'].mean()` = x-position (mean)
- `color='red'` = Color
- `linestyle='--'` = Dashed line
- `linewidth=2` = Line thickness
- `label=...` = Label for legend

```python
plt.axvline(df['duration_min'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["duration_min"].median():.0f} min')
```
**Explanation:** Draws a green vertical line at the median.

```python
plt.xlabel('Duration (Minutes)', fontsize=12)
```
**Explanation:** Sets the x-axis label with font size 12.

```python
plt.ylabel('Frequency', fontsize=12)
```
**Explanation:** Sets the y-axis label.

```python
plt.title('Distribution of Hiking Duration', fontsize=14, fontweight='bold')
```
**Explanation:** Sets the title:
- `fontsize=14` = Larger than normal font
- `fontweight='bold'` = Bold

```python
plt.legend()
```
**Explanation:** Shows the legend (with the labels from axvline).

```python
plt.grid(axis='y', alpha=0.3)
```
**Explanation:** Adds grid lines:
- `axis='y'` = Only horizontal lines
- `alpha=0.3` = Slightly transparent (subtle)

```python
plt.subplot(1, 2, 2)
```
**Explanation:** Activates the second subplot (right).

```python
plt.hist(df['duration_min']/60, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
```
**Explanation:** Creates second histogram with duration in HOURS (divided by 60).

```python
plt.tight_layout()
```
**Explanation:** Automatically optimizes spacing between subplots so nothing overlaps.

```python
plt.show()
```
**Explanation:** Displays the finished graphic.

```python
print(f"Most hikes last between {df['duration_min'].quantile(0.25)/60:.1f} and {df['duration_min'].quantile(0.75)/60:.1f} hours.")
```
**Explanation:** Calculates and outputs the interquartile range:
- `.quantile(0.25)` = 25%-percentile (lower quartile)
- `.quantile(0.75)` = 75%-percentile (upper quartile)
- Conversion to hours

---

### 3.2 Boxplot: Distance by Difficulty Level

```python
plt.figure(figsize=(12, 6))
```
**Explanation:** Create new figure.

```python
difficulty_order = sorted(df['difficulty_level'].unique())
```
**Explanation:** 
- `df['difficulty_level'].unique()` = All unique difficulty levels
- `sorted()` = Sorts them alphabetically (T1, T2, T3, ...)

```python
sns.boxplot(data=df, x='difficulty_level', y='distance_km', order=difficulty_order, palette='Set2')
```
**Explanation:** Creates a Seaborn boxplot:
- `data=df` = Data source
- `x='difficulty_level'` = Categorical variable on x-axis
- `y='distance_km'` = Numeric variable on y-axis
- `order=difficulty_order` = Sort order
- `palette='Set2'` = Color scheme

```python
medians = df.groupby('difficulty_level')['distance_km'].median().reindex(difficulty_order)
```
**Explanation:** Calculates medians for each difficulty level:
- `groupby('difficulty_level')` = Groups by difficulty
- `['distance_km'].median()` = Calculates median of distance per group
- `.reindex(difficulty_order)` = Sorts in desired order

```python
for i, (level, median) in enumerate(medians.items()):
    plt.text(i, median, f'{median:.1f} km', ha='center', va='bottom', fontweight='bold', fontsize=10)
```
**Explanation:** Loop over all medians:
- `enumerate(medians.items())` = Returns index and (level, median) pairs
- `plt.text(i, median, ...)` = Places text at position (x=i, y=median)
- `ha='center'` = horizontal alignment (text alignment)
- `va='bottom'` = vertical alignment (text above the point)

---

### 3.3 Scatterplot: Distance vs. Duration

```python
difficulty_colors = {level: i for i, level in enumerate(sorted(df['difficulty_level'].unique()))}
```
**Explanation:** Dictionary comprehension - creates a mapping:
- Difficulty level → Number (for color coding)
- Example: {'T1': 0, 'T2': 1, 'T3': 2, ...}

```python
colors = df['difficulty_level'].map(difficulty_colors)
```
**Explanation:** Converts each difficulty level to a number (for colors):
- `.map()` = Replaces each value with the corresponding value from the dictionary

```python
scatter = plt.scatter(df['distance_km'], df['duration_min'], c=colors, cmap='viridis',
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
```
**Explanation:** Creates scatterplot:
- `df['distance_km']` = x-values
- `df['duration_min']` = y-values
- `c=colors` = Colors based on difficulty
- `cmap='viridis'` = Colormap (color scheme)
- `s=50` = Point size
- `edgecolors='black'` = Black border around each point

```python
cbar = plt.colorbar(scatter, ticks=list(difficulty_colors.values()))
```
**Explanation:** Adds a color scale (colorbar):
- `ticks=...` = Positions labels at the numbers

```python
cbar.ax.set_yticklabels(list(difficulty_colors.keys()))
```
**Explanation:** Replaces the numbers in the colorbar with the actual difficulty levels (T1, T2, ...).

```python
cbar.set_label('Difficulty Level', rotation=270, labelpad=20, fontsize=11)
```
**Explanation:** Labels the colorbar:
- `rotation=270` = Rotates text by 270° (vertical, readable from right)
- `labelpad=20` = Distance between label and colorbar

```python
z = np.polyfit(df['distance_km'].dropna(), df.loc[df['distance_km'].notna(), 'duration_min'], 1)
```
**Explanation:** Calculates linear regression (trendline):
- `np.polyfit()` = Finds best polynomial function
- `.dropna()` = Removes NaN values
- `1` = Degree 1 (linear: y = mx + b)
- Result: Array [m, b] (slope and y-intercept)

```python
p = np.poly1d(z)
```
**Explanation:** Creates a polynomial object from the coefficients that can be called like a function.

```python
plt.plot(df['distance_km'].sort_values(), p(df['distance_km'].sort_values()),
         "r--", linewidth=2, alpha=0.8, label=f'Trendline: y={z[0]:.1f}x+{z[1]:.1f}')
```
**Explanation:** Draws the trendline:
- `.sort_values()` = Sorts x-values for smooth line
- `p(...)` = Calculates y-values with the polynomial function
- `"r--"` = Red, dashed
- `label=...` = Shows the equation in the legend

---

### 3.4 Correlation Heatmap

```python
plt.figure(figsize=(10, 8))
```
**Explanation:** Square figure for heatmap.

```python
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
```
**Explanation:** Creates a heatmap:
- `annot=True` = Shows numbers in cells
- `fmt='.3f'` = Format: 3 decimal places
- `cmap='coolwarm'` = Color scheme (blue=negative, red=positive)
- `center=0` = 0 is displayed as white
- `square=True` = Square cells
- `linewidths=1` = Separator lines between cells
- `cbar_kws={"shrink": 0.8}` = Colorbar at 80% height

---

### 3.5 Barplot: Top 5 Cantons

```python
top5_cantons = df['canton'].value_counts().head(5)
```
**Explanation:** As before - counts cantons and takes Top 5.

```python
bars = plt.bar(range(len(top5_cantons)), top5_cantons.values,
               color='steelblue', edgecolor='black', alpha=0.8)
```
**Explanation:** Creates bar chart:
- `range(len(top5_cantons))` = x-positions: [0, 1, 2, 3, 4]
- `top5_cantons.values` = Heights of bars
- Stores bar objects in `bars` for later reference

```python
plt.xticks(range(len(top5_cantons)), top5_cantons.index, fontsize=12)
```
**Explanation:** Sets x-axis labels:
- `range(len(top5_cantons))` = Positions
- `top5_cantons.index` = Canton names

```python
for i, (bar, value) in enumerate(zip(bars, top5_cantons.values)):
```
**Explanation:** Loop over bars and values:
- `zip(bars, top5_cantons.values)` = Combines lists pairwise
- `enumerate()` = Additionally provides the index

```python
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
```
**Explanation:** Places value above each bar:
- `bar.get_x() + bar.get_width()/2` = Center of bar (x)
- `bar.get_height() + 2` = Just above the bar (y)

---

### 3.6 Stacked Barplot

```python
top5_cantons = df['canton'].value_counts().head(5).index
```
**Explanation:** This time only the NAMES (index), not the counts.

```python
df_top5 = df[df['canton'].isin(top5_cantons)]
```
**Explanation:** Filters DataFrame to only Top 5 cantons:
- `.isin(top5_cantons)` = Checks if canton is in the list
- Result: Boolean mask for filtering

```python
crosstab_difficulty = pd.crosstab(df_top5['canton'], df_top5['difficulty_level'])
```
**Explanation:** Cross table for the Top 5 cantons:
- Rows = Cantons
- Columns = Difficulty levels
- Values = Count

```python
difficulty_cols = sorted([col for col in crosstab_difficulty.columns])
```
**Explanation:** List comprehension - sorts the column names.

```python
crosstab_difficulty = crosstab_difficulty[difficulty_cols]
```
**Explanation:** Arranges the columns in sorted order.

```python
crosstab_difficulty.plot(kind='bar', stacked=True, figsize=(12, 6),
                         colormap='viridis', edgecolor='black', linewidth=0.5)
```
**Explanation:** Creates stacked bar chart:
- `kind='bar'` = Bar chart
- `stacked=True` = Bars are stacked on top of each other
- `colormap='viridis'` = Different colors for difficulty levels

```python
plt.legend(title='Difficulty Level', bbox_to_anchor=(1.05, 1), loc='upper left')
```
**Explanation:** Positions legend:
- `title=...` = Legend title
- `bbox_to_anchor=(1.05, 1)` = Position outside the plot (upper right)
- `loc='upper left'` = Anchor point of legend box

```python
plt.xticks(rotation=0)
```
**Erklärung:** Dreht x-Achsen-Beschriftungen auf 0° (horizontal).

---

## 4. Summary

```python
print("="*80)
print("SUMMARY - Key Findings from EDA")
print("="*80)
```
**Explanation:** Heading with separator lines.

```python
print()
```
**Explanation:** Blank line.

```python
print("DATA OVERVIEW:")
print(f"- Number of hiking trails: {len(df)}")
```
**Explanation:** Outputs the total number of hiking trails. The spaces at the beginning create indentation.

```python
print(f"- Number of cantons: {df['canton'].nunique()}")
```
**Explanation:** `.nunique()` = Counts the number of unique values.

```python
print(f"- Difficulty levels: {sorted(df['difficulty_level'].unique())}")
```
**Explanation:** Zeigt alle vorhandenen Schwierigkeitsgrade sortiert an.

```python
print(f"- Strongest correlation: {corr_pairs_sorted[0][0]} <-> {corr_pairs_sorted[0][1]} ({corr_pairs_sorted[0][2]:.3f})")
```
**Explanation:** Greift auf das erste Element der sortierten Korrelations-Liste zu:
- `[0]` = Erstes Tupel
- `[0]`, `[1]`, `[2]` = Erstes, zweites, drittes Element des Tupels

```python
print(f"- Average hiking time: {df['duration_min'].mean()/60:.1f} hours")
```
**Explanation:** Berechnet Durchschnitt und rechnet in Stunden um.

```python
print(f"- Max. ascent: {df['ascent_m'].max():.0f} m")
```
**Explanation:** `.max()` = Findet den Maximalwert der Spalte.

```python
print(f"- Correlation Ascent <-> Descent: {df[['ascent_m', 'descent_m']].corr().iloc[0,1]:.3f}")
```
**Explanation:** 
- `df[['ascent_m', 'descent_m']]` = Wählt beide Spalten aus
- `.corr()` = Berechnet Korrelationsmatrix (2x2)
- `.iloc[0,1]` = Greift auf Zelle Zeile 0, Spalte 1 zu (Korrelation zwischen den beiden)

```python
print(f"- Top 5 Cantons: {', '.join(top5_cantons.tolist())}")
```
**Explanation:** 
- `.tolist()` = Wandelt Series in Python-Liste um
- `', '.join(...)` = Verbindet Liste zu einem String mit Kommas

```python
print(f"- Leading Canton: {top5_cantons[0]} ({canton_counts.iloc[0]} hiking trails)")
```
**Explanation:** 
- `top5_cantons[0]` = Erster Kanton
- `canton_counts.iloc[0]` = Anzahl des ersten Kantons

---

## Footer: System Information

```python
import os
import platform
import socket
from platform import python_version
from datetime import datetime
```
**Explanation:** Imports modules for system information:
- `os` = Operating system functions
- `platform` = Platform information
- `socket` = Network functions (not used here)
- `python_version` = Function for Python version
- `datetime` = Date/time functions

```python
print('-----------------------------------')
```
**Explanation:** Trennlinie.

```python
print(os.name.upper())
```
**Explanation:** 
- `os.name` = Name des Betriebssystems ('nt' für Windows, 'posix' für Linux/Mac)
- `.upper()` = Wandelt in Grossbuchstaben um

```python
print(platform.system(), '|', platform.release())
```
**Explanation:** 
- `platform.system()` = Betriebssystem-Name (z.B. "Windows", "Linux")
- `platform.release()` = Version/Release (z.B. "10", "11")
- `'|'` = Trennzeichen

```python
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
```
**Explanation:** 
- `datetime.now()` = Aktuelles Datum und Zeit
- `.strftime()` = Formatiert Datum als String
- `"%Y-%m-%d %H:%M:%S"` = Format: Jahr-Monat-Tag Stunde:Minute:Sekunde

```python
print('Python Version:', python_version())
```
**Explanation:** Gibt die installierte Python-Version aus (z.B. "3.11.5").

---

## Summary of Key Concepts:

### Pandas Operations:
- `.head()` = First rows
- `.describe()` = Statistics
- `.value_counts()` = Frequencies
- `.groupby()` = Grouping
- `.corr()` = Correlation
- `.isnull()` = Missing values
- `.unique()` = Unique values
- `.nunique()` = Count of unique values

### Matplotlib/Seaborn:
- `plt.figure()` = New graphic
- `plt.subplot()` = Multiple plots in one figure
- `plt.hist()` = Histogram
- `plt.scatter()` = Scatterplot
- `sns.boxplot()` = Boxplot
- `sns.heatmap()` = Heatmap
- `plt.xlabel/ylabel/title()` = Labels

### Python Basics:
- f-Strings: `f"Text {variable}"` = String formatting
- List Comprehension: `[x for x in list]` = Compact list creation
- Lambda: `lambda x: x**2` = Anonymous function
- `.round()`, `.mean()`, `.median()` = Mathematical operations

---

**End of Code Explanations**
