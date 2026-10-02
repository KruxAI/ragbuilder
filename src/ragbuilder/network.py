"""Bounded public-web downloads with pinned DNS and redirect validation.

This boundary applies to document and prompt downloads, not user-configured LLM
or database services, which may legitimately run on a private network.
"""

import ipaddress
import socket
import time
from urllib.parse import urljoin, urlsplit

import certifi
import requests
import urllib3

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


def public_destination(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Document URLs must be HTTP(S) URLs without credentials")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        raise ValueError("Document URLs must use port 80 or 443")
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    ips = [address[4][0] for address in addresses]
    if not ips or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast for ip in ips):
        raise ValueError("Document URLs must resolve exclusively to public IP addresses")
    return parsed, port, ips[0]


def public_request(url, *, method="GET", headers=None, **kwargs):
    """Return a requests-compatible response without proxies or a second DNS lookup."""
    if method not in {"GET", "HEAD"}:
        raise ValueError("Only GET and HEAD are supported")
    started = time.monotonic()
    for _ in range(6):
        parsed, port, address = public_destination(url)
        pool_options = {"host": address, "port": port, "timeout": urllib3.Timeout(connect=5, read=15)}
        if parsed.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                **pool_options, server_hostname=parsed.hostname,
                assert_hostname=parsed.hostname, cert_reqs="CERT_REQUIRED", ca_certs=certifi.where(),
            )
        else:
            pool = urllib3.HTTPConnectionPool(**pool_options)
        request_headers = {"User-Agent": "ragbuilder", "Accept-Encoding": "identity"}
        request_headers.update(headers or {})
        request_headers["Host"] = parsed.netloc
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        raw = None
        try:
            raw = pool.urlopen(method, path, headers=request_headers, redirect=False, retries=False, preload_content=False)
            if raw.status in {301, 302, 303, 307, 308}:
                location = raw.headers.get("Location")
                if not location:
                    raise ValueError("Redirect has no location")
                url = urljoin(url, location)
                continue
            chunks = []
            size = 0
            if method != "HEAD":
                for chunk in raw.stream(65536, decode_content=True):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES or time.monotonic() - started > 60:
                        raise ValueError("Document download exceeds the size or time limit")
                    chunks.append(chunk)
            response = requests.Response()
            response.status_code = raw.status
            response.headers = requests.structures.CaseInsensitiveDict(raw.headers)
            response.url = url
            response._content = b"".join(chunks)
            response._content_consumed = True
            response.encoding = requests.utils.get_encoding_from_headers(response.headers)
            return response
        finally:
            if raw is not None:
                raw.close()
            pool.close()
    raise ValueError("Too many redirects")


def public_get(url, **kwargs):
    return public_request(url, method="GET", **kwargs)


def public_head(url, **kwargs):
    return public_request(url, method="HEAD", **kwargs)


def read_csv(source, **kwargs):
    """Load evaluation CSVs through the same URL boundary as documents."""
    from io import BytesIO
    from pathlib import Path
    import pandas as pd

    if urlsplit(str(source)).scheme in {"http", "https"}:
        response = public_get(str(source))
        response.raise_for_status()
        return pd.read_csv(BytesIO(response.content), **kwargs)
    # A Path prevents pandas/fsspec from interpreting other URL protocols.
    return pd.read_csv(Path(source), **kwargs)


class PublicWebLoader:
    """Load web text without allowing a loader to bypass the download boundary."""

    def __init__(self, web_path=None, *, web_paths=None, **kwargs):
        paths = web_paths if web_paths is not None else web_path
        self.paths = [paths] if isinstance(paths, str) else list(paths or [])

    def lazy_load(self):
        from bs4 import BeautifulSoup
        from langchain_core.documents import Document

        for url in self.paths:
            response = public_get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            yield Document(page_content=soup.get_text(), metadata={"source": url})

    def load(self):
        return list(self.lazy_load())
