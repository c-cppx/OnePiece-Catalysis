# OnePiece Cu/Ga Phase Tutorial Notebooks

Diese Notebook-Serie startet bei den HDF-Dateien aus OnePiece und zeigt Schritt
für Schritt, wie die Tabellen unter dem Bulk/Oberflächen-Multiplot entstehen.

## Reihenfolge

1. `00_hdf_onepiece_pandas_ase_intro.ipynb`
   - lädt eine HDF-Datei mit `pd.read_hdf(filename, key="df")`
   - erklärt `pandas.DataFrame`, OnePiece-Adapter und ASE-Strukturspalten
   - zeigt erste Filter-, Sortier- und Descriptor-Befehle

2. `01_bulk_phase_table_from_hdf.ipynb`
   - lädt `CuGabulk_oxide.hdf`
   - baut das Temperatur- und `pH2O/pH2`-Raster
   - evaluiert Bulk-Energieausdrücke
   - erzeugt `tutorial_bulk_transition_summary.csv`

3. `02_surface_phase_tables_from_hdf.ipynb`
   - lädt `CuGasurf_100.hdf`, `CuGasurf_110.hdf`, `CuGasurf_111.hdf`,
     `CuGasurf_211.hdf`
   - berechnet korrigierte Oberflächenenergien pro Fläche
   - erzeugt pro Miller-Index und kombiniert stabile Phasentabellen

4. `03_multiplot_transition_tables.ipynb`
   - kombiniert Bulk- und Oberflächen-Summaries
   - formt das gemeinsame Tabellenschema für den Multiplot
   - speichert `tutorial_bulk_surface_transition_phase_summary_extended.csv`

## Inputs

Die Notebooks erwarten die HDF-Dateien hier:

`/Users/dk2994/Desktop/Uni/Journal/Thesis/Notebooks/Surface Alloys`

Die Pfade sind oben in jedem Notebook in `DATA_ROOT` definiert.

## Outputs

Die Tutorial-Outputs werden nach

`/Users/dk2994/Desktop/git/PFUI/notebooks/phase_diagram_outputs`

geschrieben, damit sie neben den bereits erzeugten finalen Diagrammen liegen.
