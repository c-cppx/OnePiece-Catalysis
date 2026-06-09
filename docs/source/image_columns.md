# Image Columns

OnePiece Studio can show images when a DataFrame column contains image paths.

This is useful for:

- rendered ASE structure thumbnails
- DOS plots such as `totaldos.png`
- charge-density or geometry images
- publication-ready snapshots stored next to each calculation folder

## Add A Path Column

If each calculation row has a `Path` column, you can create an image column like this:

```python
from pathlib import Path

df["dos_image"] = df["Path"].map(lambda value: str(Path(value) / "totaldos.png"))
```

Then mark it as an image column:

```python
from onepiece_studio import DataFrameSource, OnePieceStudioConfig
from onepiece_studio.ui.streamlit_app import run_app

source = DataFrameSource(df, name="CuGa surfaces")
config = OnePieceStudioConfig(
    title="CuGa Surface Database",
    primary_key="Name",
    image_columns=["dos_image"],
    structure_columns=["struc"],
    asset_root=Path("/"),
)

run_app(source, config)
```

Values may be:

- absolute local paths such as `path/to/totaldos.png`
- relative local paths resolved against `asset_root`
- local file URLs
- `https://` URLs

## Generate ASE Structure Images

A useful next step is to render `struc` into a thumbnail folder and store the thumbnail path:

```python
from pathlib import Path
from ase.io import write

thumb_dir = Path("docs/source/_static/structure_thumbnails")
thumb_dir.mkdir(parents=True, exist_ok=True)

image_paths = []
for index, atoms in df["struc"].items():
    image_path = thumb_dir / f"structure_{index}.png"
    write(image_path, atoms, rotation="10x,20y,0z")
    image_paths.append(str(image_path))

df["structure_image"] = image_paths
```

Then set:

```python
OnePieceStudioConfig(image_columns=["structure_image"], structure_columns=["struc"])
```

The DataFrame remains the source of truth: the UI only needs to know which columns should be treated
as images.
