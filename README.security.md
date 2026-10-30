# rembg 2.0.69 security backports

This fork retains the rembg 2.0.69 library and CLI interfaces and identifies
the patched package as `2.0.69+security.2`.

GHSA-55v6-g8pm-pw4c is addressed at the HTTP image-fetch boundary. The fetcher
validates HTTP and HTTPS URLs, rejects private and ambiguous IP literals, and
uses a validating resolver for the connection itself. Redirects are bounded
and every destination is checked. Proxy environment variables are ignored.
IPv4-mapped and 6to4 addresses receive the same public-address checks.

CORS permits no cross-origin requests by default. Set `REMBG_CORS_ORIGINS`
to a JSON list of explicit origins when browser clients need access.
Credentialed CORS and wildcard origins are disabled.

GHSA-3wqj-33cg-xc48 is addressed at both HTTP and browser model-selection
boundaries. Custom model paths must resolve to regular files inside
`U2NET_HOME`, including symlink resolution. Outside and invalid paths return
the same public error, without disclosing file existence or filesystem errors.
The server's session cache distinguishes model options.

Trusted library and CLI callers can continue using custom model paths outside
the HTTP model directory. ONNX Runtime still receives a filename, preserving
models that use external tensor data.

Upstream references are
`07ad0d493057bddf821dcc3e2410eb7e065257c0` and
`7c76d3cdc5757ffbda6a76664b24cfbecdb80273`.
These backports also close redirect, DNS rebinding, alternate numeric-address,
symlink and file-existence gaps in the proposed upstream changes.

The security tests use an actual HTTP server with DNS and network connections
controlled at system boundaries. Custom model tests run small ONNX models on
CPU through all three supported custom sessions and through the HTTP API.

GHSA-qr6x-7v33-6qmg is addressed by allowing only the three published SAM
checkpoint names before constructing download URLs or cache paths. HTTP
requests with invalid SAM names return 400. Every SAM model and sidecar
download uses the checksum from upstream commit
`b2758c6b39bb2b02df90a03128110339476984d3`. Library and CLI callers retain
the existing operator-controlled `MODEL_CHECKSUM_DISABLED` behavior.
