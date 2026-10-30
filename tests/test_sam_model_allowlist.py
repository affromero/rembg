from pathlib import Path
from typing import BinaryIO

import pooch
import pytest

from rembg.sessions.sam import SamSession


@pytest.mark.parametrize(
    "model",
    [
        "../../evil",
        "sam_vit_b_01ec64#/../../../evil",
        "sam_vit_b_01ec64/../../../evil",
        "/etc/passwd",
        "sam_vit_b_01ec64\x00evil",
        "",
        None,
        [],
        {},
        1,
    ],
)
def test_invalid_names_cannot_download_or_create_cache_files(
    model: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    monkeypatch.setenv("U2NET_HOME", str(cache))

    def refuse_network(*args: object, **kwargs: object) -> None:
        pytest.fail("An invalid SAM name reached the download boundary")

    monkeypatch.setattr(pooch, "retrieve", refuse_network)
    with pytest.raises(ValueError, match="sam_model"):
        SamSession.download_models(sam_model=model)
    assert not cache.exists()


@pytest.mark.parametrize(
    "model", [None, "sam_vit_b_01ec64", "sam_vit_l_0b3195", "sam_vit_h_4b8939"]
)
@pytest.mark.parametrize("quant", [False, True])
def test_published_checkpoints_keep_confined_pinned_downloads(
    model: str | None,
    quant: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    monkeypatch.delenv("MODEL_CHECKSUM_DISABLED", raising=False)
    downloads: list[tuple[str, str | None]] = []

    def retrieve(
        url: str,
        known_hash: str | None,
        *,
        fname: str,
        path: str,
        progressbar: bool,
    ) -> str:
        assert url == (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/" + fname
        )
        assert known_hash is not None and known_hash.startswith("md5:")
        target = Path(path) / fname
        assert target.parent == tmp_path
        target.write_bytes(b"model-data")
        downloads.append((fname, known_hash))
        return str(target)

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    if model is None:
        encoder, decoder = SamSession.download_models(sam_quant=quant)
        model = "sam_vit_b_01ec64"
    else:
        encoder, decoder = SamSession.download_models(sam_model=model, sam_quant=quant)
    suffix = ".quant.onnx" if quant else ".onnx"
    assert Path(encoder) == tmp_path / f"{model}.encoder{suffix}"
    assert Path(decoder) == tmp_path / f"{model}.decoder{suffix}"
    assert Path(encoder).read_bytes() == b"model-data"
    assert Path(decoder).read_bytes() == b"model-data"
    if model == "sam_vit_h_4b8939" and not quant:
        assert (
            tmp_path / f"{model}.encoder_data.bin"
        ).read_bytes() == b"model-data" * 3
        assert {name for name, _ in downloads} == {
            f"{model}.encoder.onnx",
            f"{model}.decoder.onnx",
            *(f"{model}.encoder_data.{index}.bin" for index in (1, 2, 3)),
        }
    else:
        assert {name for name, _ in downloads} == {
            Path(encoder).name,
            Path(decoder).name,
        }


def test_untrusted_download_bytes_fail_checksum_before_entering_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    monkeypatch.delenv("MODEL_CHECKSUM_DISABLED", raising=False)

    def download(
        self: pooch.HTTPDownloader,
        url: str,
        output_file: str | BinaryIO,
        pooch: object = None,
        check_only: bool = False,
    ) -> None:
        if isinstance(output_file, str):
            Path(output_file).write_bytes(b"untrusted-onnx-bytes")
        else:
            output_file.write(b"untrusted-onnx-bytes")

    monkeypatch.setattr(pooch.HTTPDownloader, "__call__", download)
    with pytest.raises(ValueError, match="hash"):
        SamSession.download_models()
    assert not (tmp_path / "sam_vit_b_01ec64.encoder.onnx").exists()
