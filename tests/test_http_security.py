import asyncio
import io
import json
import socket
from pathlib import Path

import aiohappyeyeballs
import aiohttp.web
import httpx
import numpy as np
import onnx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from onnx import TensorProto, helper
from PIL import Image

from rembg import new_session
from rembg.commands.s_command import s_command
from rembg.http_security import (
    PublicResolver,
    fetch_public_image,
    validate_http_model_options,
    validate_public_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/image",
        "http://localhost@8.8.8.8/",
        "http://127.0.0.1",
        "http://127.1",
        "http://2130706433",
        "http://0127.0.0.1",
        "http://10.0.0.1",
        "http://169.254.169.254",
        "http://100.64.0.1",
        "http://224.0.0.1",
        "http://[::1]",
        "http://[::ffff:127.0.0.1]",
        "http://[2002:7f00:1::]",
        "http://[2002:a00:1::]",
        "http://[fe80::1%25eth0]",
    ],
)
def test_image_urls_reject_private_and_ambiguous_addresses(url):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_public_https_urls_preserve_hostname_and_path():
    url = validate_public_url("https://example.com/image.png?size=20")
    assert url.host == "example.com"
    assert url.path == "/image.png"
    assert url.query["size"] == "20"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address", ["127.0.0.1", "::1", "::ffff:127.0.0.1", "10.0.0.1", "169.254.169.254"]
)
async def test_dns_cannot_supply_private_addresses(monkeypatch, address):
    def resolve(host, port, *args, **kwargs):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        endpoint = (
            (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
        )
        return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", endpoint)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    resolver = PublicResolver()
    try:
        with pytest.raises(ValueError, match="non-public"):
            await resolver.resolve("images.example", 80, socket.AF_UNSPEC)
    finally:
        await resolver.close()


@pytest_asyncio.fixture
async def public_image_server(monkeypatch):
    requested = []
    app = aiohttp.web.Application()

    async def handle(request):
        requested.append(request.path)
        if request.path == "/redirect-private":
            raise aiohttp.web.HTTPFound("http://169.254.169.254/private")
        if request.path == "/redirect-public":
            raise aiohttp.web.HTTPFound("/image")
        if request.path == "/loop":
            raise aiohttp.web.HTTPFound("/loop")
        return aiohttp.web.Response(body=b"public-image", content_type="image/png")

    app.router.add_get("/{path:.*}", handle)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    site = aiohttp.web.SockSite(runner, listener)
    await site.start()

    addresses = {"images.example": "8.8.8.8"}

    def resolve(host, destination_port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (addresses[host], destination_port),
            )
        ]

    async def connect(**kwargs):
        connection = socket.socket()
        connection.setblocking(False)
        await asyncio.get_running_loop().sock_connect(connection, ("127.0.0.1", port))
        return connection

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(aiohappyeyeballs, "start_connection", connect)
    yield requested, addresses
    await runner.cleanup()


@pytest.mark.asyncio
async def test_public_image_fetching_and_relative_redirects_work(public_image_server):
    assert (
        await fetch_public_image("http://images.example/redirect-public")
        == b"public-image"
    )
    requested, addresses = public_image_server
    assert requested == ["/redirect-public", "/image"]


@pytest.mark.asyncio
async def test_redirects_cannot_reach_metadata_services(public_image_server):
    with pytest.raises(ValueError, match="non-public"):
        await fetch_public_image("http://images.example/redirect-private")
    requested, addresses = public_image_server
    assert requested == ["/redirect-private"]


@pytest.mark.asyncio
async def test_redirect_loops_fail_explicitly(public_image_server):
    with pytest.raises(ValueError, match="redirect"):
        await fetch_public_image("http://images.example/loop")


@pytest.mark.asyncio
async def test_dns_rebinding_cannot_reuse_a_previous_public_resolution(
    public_image_server,
):
    requested, addresses = public_image_server
    assert await fetch_public_image("http://images.example/image") == b"public-image"
    addresses["images.example"] = "127.0.0.1"
    with pytest.raises(ValueError, match="non-public"):
        await fetch_public_image("http://images.example/private")
    assert requested == ["/image"]


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("U2NET_HOME", str(root))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["mask"], axes=[1], keepdims=1)],
        "segmentation",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, "height", "width"]
            )
        ],
        [
            helper.make_tensor_value_info(
                "mask", TensorProto.FLOAT, [1, 1, "height", "width"]
            )
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    inside = root / "weights.onnx"
    outside = tmp_path / "outside.onnx"
    onnx.save(model, inside)
    onnx.save(model, outside)
    return root, inside, outside


def test_http_model_paths_reject_traversal_sibling_paths_and_symlinks(model_files):
    root, inside, outside = model_files
    link = root / "escaped.onnx"
    link.symlink_to(outside)
    for value in [
        str(outside),
        str(root / ".." / "outside.onnx"),
        str(link),
        str(root),
    ]:
        with pytest.raises(ValueError):
            validate_http_model_options({"model_path": value})
    options = validate_http_model_options({"model_path": str(inside), "other": 1})
    assert options == {"model_path": str(inside.resolve()), "other": 1}


@pytest.mark.parametrize(
    "options", [[], "weights.onnx", {"model_path": None}, {"model_path": ""}]
)
def test_http_model_options_require_valid_json_objects(options, model_files):
    with pytest.raises(ValueError):
        validate_http_model_options(options)


@pytest.mark.parametrize("model_name", ["u2net_custom", "dis_custom", "ben_custom"])
def test_trusted_library_custom_models_preserve_cpu_inference(model_name, model_files):
    root, inside, outside = model_files
    session = new_session(
        model_name, model_path=str(outside), providers=["CPUExecutionProvider"]
    )
    pixels = np.tile(np.arange(24, dtype=np.uint8)[None, :, None], (20, 1, 3))
    masks = session.predict(Image.fromarray(pixels))
    assert masks[0].size == (24, 20)
    assert np.asarray(masks[0]).max() > np.asarray(masks[0]).min()


@pytest.fixture
def http_app(monkeypatch, request):
    apps = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: apps.append(app))
    if hasattr(request, "param"):
        monkeypatch.setenv("REMBG_CORS_ORIGINS", json.dumps(request.param))
    else:
        monkeypatch.delenv("REMBG_CORS_ORIGINS", raising=False)
    s_command.callback(port=7000, host="127.0.0.1", log_level="error", threads=1)
    return apps[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["u2net_custom", "dis_custom", "ben_custom"])
async def test_http_custom_models_allow_confined_weights_and_reject_external_weights(
    http_app, model_files, model_name
):
    root, inside, outside = model_files
    pixels = np.tile(np.arange(24, dtype=np.uint8)[None, :, None], (20, 1, 3))
    image = io.BytesIO()
    Image.fromarray(pixels).save(image, format="PNG")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        for path, expected_status in [(inside, 200), (outside, 400)]:
            response = await client.post(
                "/api/remove",
                params={"extras": json.dumps({"model_path": str(path)})},
                data={"model": model_name, "om": "true"},
                files={"file": ("input.png", image.getvalue(), "image/png")},
            )
            assert response.status_code == expected_status, (
                response.text[:1000]
                if expected_status != 200
                else response.content[:1000]
            )
            if expected_status == 200:
                assert Image.open(io.BytesIO(response.content)).size == (24, 20)


@pytest.mark.asyncio
async def test_http_api_rejects_local_url_before_fetching(http_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/api/remove", params={"url": "http://127.0.0.1/private"}
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_cross_origin_access_is_denied_by_default(http_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        response = await client.options(
            "/api/remove",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_http_external_model_paths_do_not_reveal_file_existence(
    http_app, model_files
):
    root, inside, outside = model_files
    inaccessible = outside.parent / "inaccessible.onnx"
    inaccessible.write_bytes(outside.read_bytes())
    inaccessible.chmod(0)
    details = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        for path in [
            outside,
            outside.parent / "missing.onnx",
            inaccessible,
            outside.parent / "absent" / "file.onnx",
        ]:
            response = await client.post(
                "/api/remove",
                params={"extras": json.dumps({"model_path": str(path)})},
                data={"model": "u2net_custom"},
                files={"file": ("input.png", b"unused", "image/png")},
            )
            assert response.status_code == 400
            details.append(response.json()["detail"])
    assert len(set(details)) == 1
    assert all(str(outside.parent) not in detail for detail in details)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_app", [["https://trusted.example"]], indirect=True)
async def test_explicit_cors_origins_allow_only_configured_clients(http_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        for origin, status in [
            ("https://trusted.example", 200),
            ("https://untrusted.example", 400),
        ]:
            response = await client.options(
                "/api/remove",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.status_code == status
            if status == 200:
                assert response.headers["access-control-allow-origin"] == origin
                assert "access-control-allow-credentials" not in response.headers


def test_wildcard_cors_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("REMBG_CORS_ORIGINS", '["*"]')
    with pytest.raises(ValueError, match="explicit origins"):
        s_command.callback(port=7000, host="127.0.0.1", log_level="error", threads=1)


@pytest.mark.parametrize("model_name", ["u2net_custom", "dis_custom", "ben_custom"])
def test_browser_custom_models_reject_external_weights_and_preserve_confined_inference(
    monkeypatch, model_files, tmp_path, model_name
):
    import gradio

    root, inside, outside = model_files
    callbacks = []
    original_interface = gradio.Interface

    def interface(function, *args, **kwargs):
        callbacks.append(function)
        return original_interface(function, *args, **kwargs)

    monkeypatch.setattr(gradio, "Interface", interface)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REMBG_CORS_ORIGINS", raising=False)
    s_command.callback(port=7000, host="127.0.0.1", log_level="error", threads=1)
    pixels = np.tile(np.arange(24, dtype=np.uint8)[None, :, None], (20, 1, 3))
    image = tmp_path / "input.png"
    Image.fromarray(pixels).save(image)
    with pytest.raises(ValueError, match="models directory"):
        callbacks[0](
            str(image),
            model_name,
            False,
            240,
            10,
            10,
            True,
            False,
            json.dumps({"model_path": str(outside)}),
        )
    output = callbacks[0](
        str(image),
        model_name,
        False,
        240,
        10,
        10,
        True,
        False,
        json.dumps({"model_path": str(inside)}),
    )
    assert Image.open(output).size == (24, 20)


@pytest.mark.asyncio
async def test_http_model_options_select_the_requested_weights(http_app, model_files):
    root, inside, outside = model_files
    inverted_path = root / "inverted.onnx"
    inverted = onnx.load(inside)
    inverted.graph.node.append(helper.make_node("Neg", ["mask"], ["inverted"]))
    inverted.graph.output[0].name = "inverted"
    onnx.save(inverted, inverted_path)
    pixels = np.tile(np.arange(24, dtype=np.uint8)[None, :, None], (20, 1, 3))
    image = io.BytesIO()
    Image.fromarray(pixels).save(image, format="PNG")
    masks = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        for path in [inside, inverted_path, inside]:
            response = await client.post(
                "/api/remove",
                params={"extras": json.dumps({"model_path": str(path)})},
                data={"model": "u2net_custom", "om": "true"},
                files={"file": ("input.png", image.getvalue(), "image/png")},
            )
            assert response.status_code == 200
            masks.append(np.asarray(Image.open(io.BytesIO(response.content))))
    np.testing.assert_array_equal(masks[0], masks[2])
    assert not np.array_equal(masks[0], masks[1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sam_model", ["sam_vit_b_01ec64#/../../../evil", "../../evil", None, [], {}]
)
async def test_http_rejects_injected_sam_names_without_creating_models(
    http_app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sam_model: object,
) -> None:
    cache = tmp_path / "models"
    monkeypatch.setenv("U2NET_HOME", str(cache))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=http_app), base_url="http://testserver"
    ) as client:
        for _ in range(2):
            response = await client.post(
                "/api/remove",
                params={"extras": json.dumps({"sam_model": sam_model})},
                data={"model": "sam"},
                files={"file": ("input.png", b"unused", "image/png")},
            )
            assert response.status_code == 400
            assert "sam_model" in response.json()["detail"]
    assert not cache.exists()
