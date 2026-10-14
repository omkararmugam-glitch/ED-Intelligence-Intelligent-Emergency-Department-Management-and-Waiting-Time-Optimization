"""Entry point for `AppTest.from_file`, so the upload page can be run headlessly.

`AppTest.from_function` re-encodes the function's source with the machine's locale
encoding, which fails on Windows for any source containing an em dash or an icon
character. Pointing AppTest at a file avoids that round trip entirely.
"""

from app.upload_page import page_upload

page_upload()
