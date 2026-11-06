# BitTM wheels for the assignment

Both wheels target 64-bit Windows and CPython 3.12. Run all commands from the assignment root. The original training build is in this directory. The optional rule-shortening build is in simplify/. Both report version 0.1.1, so use the path and SHA-256 to distinguish them.

| Build | Source commit | Wheel SHA-256 |
|---|---|---|
| Training | 29deafe26bb3fb0e431f476f757bb8b26ba24ea9 | c5c7f30d204963a98446c155fa517622140a86f8c088734d21eaac6f77b84dc1 |
| Optional simplify | d660d4933714882f2c5c827b3d38ef69058819e3 | 7e21c348fd6fb8f4a2465c34bd5d905467479155cb1a476d3ec5e90044d08f00 |

For rule-shortening experiments create a separate environment.

```powershell
uv venv --python 3.12 .venv-rules
uv pip install --python .venv-rules\Scripts\python.exe -r requirements-windows.txt
uv pip install --reinstall --no-deps --python .venv-rules\Scripts\python.exe .\vendor\simplify\bittm-0.1.1-cp312-cp312-win_amd64.whl
$env:SCIKIT_LEARN_DATA = (Join-Path (Get-Location) "data")
.\.venv-rules\Scripts\python.exe measure_bittm_tools.py
```
