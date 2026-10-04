"""Verifique integridade, correspondência de métricas e artefatos entregues."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import nbformat
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    spec = importlib.util.spec_from_file_location("experiment", ROOT / "experiments/scripts/experiment.py")
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    experiment.checks()
    rows = list(csv.DictReader((ROOT / "experiments/results/metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 9 and len({(r["seed"], r["p"]) for r in rows}) == 9
    for row in rows:
        with np.load(ROOT / f"experiments/results/synthetic_seed{row['seed']}_p{row['p']}.npz") as data:
            assert np.isfinite(data["maps"]).all() and (data["maps"] >= 0).all()
            recalculated = experiment.evaluate(data["proposals"], data["gt"])
            for name, value in recalculated.items():
                assert abs(value-float(row[name])) < 1e-12, (row, name)
            assert abs(float(data["squared_update"][-1])-float(row["max_squared_update"])) < 1e-12
            assert int(row["iterations"]) == len(data["squared_update"])
            assert row["converged"] == "True"
            pairwise = experiment.iou_matrix(data["proposals"], data["proposals"])
            np.fill_diagonal(pairwise, 0)
            assert pairwise.max() <= experiment.CONFIG["nms_iou"]
    slides = json.loads((ROOT / "presentation/slides.json").read_text())
    pptx_path = ROOT / "presentation/TP558_Diffuse2Seg.pptx"
    with ZipFile(pptx_path) as archive:
        assert archive.testzip() is None
        files = set(archive.namelist())
        for name in files:
            if name.endswith((".xml", ".rels")):
                tree = ET.fromstring(archive.read(name))
                if name.endswith(".rels"):
                    base = "" if name == "_rels/.rels" else name.rsplit("/_rels/", 1)[0]
                    for relationship in tree:
                        if relationship.attrib.get("TargetMode") != "External":
                            target = relationship.attrib["Target"]
                            assert posixpath.normpath(posixpath.join(base, target)) in files, (name, target)
        slide_parts = [n for n in files if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        note_parts = [n for n in files if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)]
        assert len(slide_parts) == len(note_parts) == len(slides) == 36
        namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        for i, slide in enumerate(slides, 1):
            text = " ".join(ET.fromstring(archive.read(f"ppt/slides/slide{i}.xml")).itertext())
            assert slide["title"] in text
            note = ET.fromstring(archive.read(f"ppt/notesSlides/notesSlide{i}.xml"))
            notes_text = " ".join(t.text or "" for t in note.findall(".//a:t", namespace))
            assert slide["notes"] in notes_text
    pdf = ROOT / "presentation/TP558_Diffuse2Seg.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    assert int(re.search(r"Pages:\s+(\d+)", info).group(1)) == 36
    notebook_counts = {}
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells and all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
        notebook_counts[path.name] = len(code_cells)
    assert len(notebook_counts) == 2
    media = {}
    for path in sorted((ROOT / "data/outputs/manim").glob("*.mp4")):
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                           "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
                          capture_output=True, text=True, check=True).stdout)
        stream = probe["streams"][0]
        assert (stream["width"], stream["height"]) == (1280, 720)
        assert stream["r_frame_rate"] == "30/1"
        duration = float(probe["format"]["duration"])
        assert duration > 0
        with Image.open(path.with_suffix(".gif")) as gif:
            assert gif.size == (800, 450) and gif.n_frames > 1
        media[path.stem] = {"seconds": duration, "resolution": "1280x720", "fps": 30}
    assert len(media) == 4
    # Folhas de contato permitem revisar todos os slides sem alterar o PDF.
    for page in range(3):
        contact = Image.new("RGB", (1350, 1100), "#dde4ed")
        draw = ImageDraw.Draw(contact)
        for k in range(12):
            index = page*12+k+1
            source = ROOT / f"presentation/slides/slide_{index:02d}.png"
            with Image.open(source) as image:
                image.thumbnail((440, 248), Image.Resampling.LANCZOS)
                x, y = (k % 3)*450+5, (k//3)*275+20
                contact.paste(image, (x, y))
                draw.text((x, y-15), f"Slide {index:02d}", fill="#172033")
        contact.save(ROOT / f"presentation/slides/contact_{page+1}.png")
    files = [ROOT / "README.md", ROOT / "notes/analysis.md", ROOT / "notes/experimental_report.md",
             ROOT / "notes/research_application.md", pptx_path, pdf,
             ROOT / "experiments/results/metrics.csv", ROOT / "experiments/results/protocol.json"]
    files.extend(sorted((ROOT / "notebooks").glob("*.ipynb")))
    result = {"checks": "passed", "slides": len(slides), "notebook_code_cells": notebook_counts,
              "experiments": len(rows), "media": media,
              "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT / "experiments/results/validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["checks", "slides", "notebook_code_cells", "experiments", "media"]}, indent=2))


if __name__ == "__main__":
    main()
