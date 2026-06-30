from pathlib import Path

import numpy as np

from narcoticsense.spectroscopy import import_spectrum


def test_import_simple_xy_csv(tmp_path: Path):
    p = tmp_path / "simple.csv"
    p.write_text("x,y\n3,0.1\n2,0.2\n1,0.3\n")
    result = import_spectrum(p, modality="uv-vis", sample_id="s1")
    assert result.x_column == "x"
    assert result.y_column == "y"
    assert result.spectrum.modality == "uv-vis"
    assert np.allclose(result.spectrum.x, [1, 2, 3])


def test_import_vendor_style_with_title_row():
    text = "Sample ABC,,\nWavelength (nm),Abs,\n800,-0.1,\n799,-0.2,\n798,-0.3,\n"
    result = import_spectrum(text.encode(), modality="fluorescence")
    assert result.header_row == 1
    assert result.x_column == "Wavelength (nm)"
    assert result.y_column == "Abs"
    assert np.all(np.diff(result.spectrum.x) > 0)
    assert any("descending" in msg for msg in result.messages)
