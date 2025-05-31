# Einkommensvorhersage anhand demografischer Merkmale

## Inhaltsverzeichnis
1. [Über das Projekt](#über-das-projekt)
2. [Verwendeter Datensatz](#verwendeter-datensatz)
3. [Projektstruktur](#projektstruktur)
4. [Verwendete Technologien](#verwendete-technologien)
5. [Installation und Ausführung](#installation-und-ausführung)
6. [Wichtigste Schritte](#wichtigste-schritte)
7. [Modellergebnisse (Beispiel)](#modellergebnisse-beispiel)

## Über das Projekt

Dieses Projekt zielt darauf ab, vorherzusagen, ob das jährliche Einkommen einer Person 50.000 US-Dollar übersteigt oder nicht. Hierfür werden demografische Daten aus dem "Adult Census Income" Datensatz verwendet. Es handelt sich um ein binäres Klassifikationsproblem, das mit verschiedenen Machine-Learning-Modellen gelöst wird.

Das primäre Skript (`einkommensvorhersage.py` oder der Name deines Notebooks) führt durch den gesamten Prozess von der Datenaufbereitung über die explorative Datenanalyse bis hin zum Training und der Evaluierung der Modelle.

## Verwendeter Datensatz

Der für dieses Projekt verwendete Datensatz ist der "Adult Census Income" Datensatz, der von der UCI Machine Learning Repository bezogen werden kann. Er enthält verschiedene anonymisierte Merkmale von Personen.
Die Datendatei `adult.data` wird für die Ausführung des Skripts benötigt.

**Merkmale (Beispiele):**
* Alter (`age`)
* Bildungsgrad (`education`)
* Beruf (`occupation`)
* Arbeitsstunden pro Woche (`hours-per-week`)
* Geschlecht (`sex`)
* Herkunftsland (`native-country`)
* Einkommen (`income`): Die Zielvariable (<=50K oder >50K)

## Projektstruktur
