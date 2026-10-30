"""Security boundaries for the public HTTP and browser interfaces."""

import ipaddress
import os
import socket
from pathlib import Path

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.helpers import is_ip_address
from aiohttp.resolver import ThreadedResolver
from yarl import URL

from .sessions.base import BaseSession
from .sessions.sam import validate_sam_model


def validate_public_address(address):
    ip = ipaddress.ip_address(address)
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    if getattr(ip, "sixtofour", None) is not None:
        validate_public_address(ip.sixtofour)
    if (
        not ip.is_global
        or ip.is_multicast
        or ip.is_reserved
        or getattr(ip, "is_site_local", False)
    ):
        raise ValueError("Requests to non-public addresses are forbidden")


def validate_public_url(value):
    url = URL(value)
    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("An HTTP or HTTPS URL with a hostname is required")
    if url.user is not None or url.password is not None or "%" in url.host:
        raise ValueError("URL credentials and scoped addresses are forbidden")
    if is_ip_address(url.host):
        validate_public_address(url.host)
    return url


class PublicResolver(AbstractResolver):
    def __init__(self):
        self.resolver = ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await self.resolver.resolve(host, port, family)
        if not records:
            raise ValueError("The hostname did not resolve to a public address")
        for record in records:
            validate_public_address(record["host"])
        return records

    async def close(self):
        await self.resolver.close()


async def fetch_public_image(value):
    url = validate_public_url(value)
    resolver = PublicResolver()
    try:
        connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(
            connector=connector, trust_env=False, timeout=timeout
        ) as session:
            for attempt in range(6):
                url = validate_public_url(url)
                async with session.get(url, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or attempt == 5:
                            raise ValueError("Invalid or excessive image redirects")
                        url = url.join(URL(location))
                        continue
                    response.raise_for_status()
                    return await response.read()
    finally:
        await resolver.close()


def validate_http_model_options(options):
    if not isinstance(options, dict):
        raise ValueError("Model options must be a JSON object")
    options = options.copy()
    if "sam_model" in options:
        options["sam_model"] = validate_sam_model(options["sam_model"])
    if "model_path" not in options:
        return options
    model_path = options["model_path"]
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a nonempty path")
    invalid_path = "model_path must be a file inside the models directory"
    try:
        root = Path(BaseSession.u2net_home()).resolve(strict=True)
        target = Path(os.path.expanduser(model_path)).resolve(strict=False)
        if not target.is_relative_to(root):
            raise ValueError(invalid_path)
        target = target.resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(invalid_path)
    except (OSError, RuntimeError) as error:
        raise ValueError(invalid_path) from error
    options["model_path"] = str(target)
    return options
