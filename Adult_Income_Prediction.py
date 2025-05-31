# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Vorhersage des Einkommens einer Person anhand demografischer Merkmale
# MAGIC
# MAGIC ### Problemdefinition
# MAGIC Ziel des Projekts ist es, vorherzusagen, ob eine Person ein Einkommen von mehr als 50.000 pro Jahr hat, unter Verwendung eines demografischen Datensatzes. Diese Art von Problem ist eine binäre Klassifikation.
# MAGIC

# COMMAND ----------

import pandas as pd

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv("adult.data", header=None, names=column_names, na_values=" ?", skipinitialspace=True)
df.head()


# COMMAND ----------

# Zeilen mit fehlenden Werten (NaN) entfernen
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Zielspalte ("income") in 0 und 1 umwandeln
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Datenbereinigung und -vorverarbeitung
# MAGIC In diesem Schritt entfernen wir fehlende Werte, konvertieren Text gegebenenfalls in ein numerisches Format und bereiten die Daten für das Training des Modells vor.
# MAGIC

# COMMAND ----------

df.info()
df.isnull().sum()

df['income'].unique()


# COMMAND ----------

# Herausfinden, welche Spalten vom Typ Text sind und eindeutige Werte in kategorialen Spalten überprüfen
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"Spalte {col}: {df[col].unique()}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Explorative Datenanalyse (EDA)
# MAGIC Wir analysieren statistische Verteilungen, Beziehungen zwischen Variablen, Korrelationen und Visualisierungen, um die Daten besser zu verstehen.
# MAGIC

# COMMAND ----------

df.describe()


# COMMAND ----------



df['income'].value_counts()
df['education'].value_counts()



# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='income', data=df) #wie viele Personen ein Einkommen <50k oder >50k haben
plt.title("Einkommensverteilung (>50K vs. <=50K)")
plt.show()


# COMMAND ----------

sns.boxplot(x='income', y='age', data=df) #Altersverteilung nach Einkommen
plt.title("Altersverteilung nach Einkommen")
plt.show()


# COMMAND ----------

sns.boxplot(x='income', y='hours-per-week', data=df)
plt.title("Wöchentliche Arbeitsstunden vs. Einkommen")
plt.show()


# COMMAND ----------

# Nur für numerische Spalten
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm') #Korrelationsmatrix
plt.title("Korrelationsmatrix")
plt.show()


# COMMAND ----------

sns.countplot(x='education', hue='income', data=df) #Korrelation zwischen Bildung und Einkommen
plt.xticks(rotation=90)
plt.title("Bildung und Einkommen")
plt.show()


# COMMAND ----------

sns.countplot(x='workclass', hue='income', data=df) #Korrelation zwischen Arbeitgebertyp und Einkommen
plt.xticks(rotation=90)
plt.title("Arbeitgebertyp und Einkommen")
plt.show()


# COMMAND ----------

X = df.drop('income', axis=1)  # alle Spalten außer 'income'
y = df['income']               # Zielvariable

# COMMAND ----------

X = pd.get_dummies(X, drop_first=True) #One-Hot-Encoding | von kategorialen Variablen in numerisches Format

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Modellierung und Training
# MAGIC Wir verwenden Algorithmen des maschinellen Lernens, um Modelle mit den verarbeiteten Daten zu trainieren, und bewerten dann deren Leistung anhand relevanter Metriken.
# MAGIC

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# wir haben 80% der Daten für das Training und 20% für das Testen zugewiesen

# COMMAND ----------

print("Größe des Trainingsdatensatzes:", X_train.shape)
print("Größe des Testdatensatzes:", X_test.shape)


# COMMAND ----------

# Variablen X und y trennen
X = df.drop('income', axis=1)
y = df['income']

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Kategoriale Spalten in numerische Variablen umwandeln
X = pd.get_dummies(X, drop_first=True)

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistisches Regressionsmodell erstellen
model = LogisticRegression(max_iter=1000)

# COMMAND ----------

# Modell mit Trainingsdaten trainieren
model.fit(X_train, y_train)


# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Grundlegende Bewertung
print("Klassifikationsbericht:\n", classification_report(y_test, y_pred)) # Precision, Recall, F1-Score
print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
print("Genauigkeit:", accuracy_score(y_test, y_pred))


# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# Parameterliste definieren
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}

# Testen verschiedener Parameterkombinationen mittels Kreuzvalidierung
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Beste Parameter anzeigen
print("Beste Parameter:", grid.best_params_)


# COMMAND ----------

# Optimiertes Modell neu trainieren
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

# Erneut validieren
y_pred_best = best_model.predict(X_test)

print("Klassifikationsbericht (optimiert):\n", classification_report(y_test, y_pred_best))
print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred_best))
print("Genauigkeit:", accuracy_score(y_test, y_pred_best))


# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)


# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Leistungsvergleich der Modelle

def evaluate_model(name, y_true, y_pred):
    print(f"🔹 {name}")
    print("Genauigkeit:", accuracy_score(y_true, y_pred))
    print("Präzision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred)) # Recall ist auch im Deutschen gebräuchlich
    print("F1-Score:", f1_score(y_true, y_pred))
    print("")

evaluate_model("Logistische Regression (optimiert)", y_test, y_pred_best)
evaluate_model("Entscheidungsbaum", y_test, y_pred_tree)
evaluate_model("Random Forest", y_test, y_pred_rf) # Random Forest ist auch im Deutschen gebräuchlich


# COMMAND ----------

# MAGIC %md
# MAGIC Wir haben Random Forest als endgültiges Modell ausgewählt und es mit dem Testdatensatz getestet.
# MAGIC Es erzielte einen F1-Score von 0.69, was auf eine ausgewogene Leistung hinweist.
# MAGIC Daher wird dieses Modell im nächsten Schritt für Analyse und Interpretation verwendet.

# COMMAND ----------

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test) # Visualisierung korrekter und inkorrekter Vorhersagen
plt.title("Konfusionsmatrix - Random Forest")
plt.show()


# COMMAND ----------

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(rf_model, X_test, y_test) # Leistung bei verschiedenen Schwellenwerten
plt.title("ROC-Kurve - Random Forest")
plt.show()


# COMMAND ----------

from sklearn.metrics import roc_auc_score

y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba_rf) # allgemeines Leistungsmaß
print("AUC-Score:", roc_auc)


# COMMAND ----------

import pandas as pd
import seaborn as sns

importances = rf_model.feature_importances_
features = pd.Series(importances, index=X.columns)
top_features = features.sort_values(ascending=False).head(10)  # Identifizierung, welche Attribute für die Vorhersage am wichtigsten sind

sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Wichtige Merkmale - Random Forest")
plt.xlabel("Wichtigkeit")
plt.show()