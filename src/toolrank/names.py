"""Tool names the agent APIs accept, derived from toolrank's tool ids.

Tool ids are ``<source>/<name>`` and may hold any character (``github/issues/create``,
``swagger-petstore/find pet by id``). Anthropic's Messages API wants ``^[a-zA-Z0-9_-]{1,128}$``
and OpenAI's Responses API ``^[a-zA-Z0-9_-]+$``; 64 characters are safe on both. ``api_name``
keeps an id readable when it can — each ``/`` becomes ``__`` — and hashes it otherwise:

- readable: the id is runs of ``[A-Za-z0-9-]`` joined by single ``_`` or ``/``, at most 64
  characters once flattened; such a name has ``_`` runs of one (``_``) or two (``/``) only, so it
  stands for exactly one id;
- hashed: anything else — the flattened id with other characters as ``_``, cut to 49 characters,
  then ``___`` and 12 hex characters of the id's sha256; a readable name never has three ``_`` in
  a row, so the two kinds cannot meet.

The name is a pure function of the id: a tool keeps it when other tools come and go, which a
conversation that has already loaded the tool relies on.
"""

from __future__ import annotations

import hashlib
import re

MAX_LEN = 64
API_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_READABLE = re.compile(r"[A-Za-z0-9-]+(?:[_/][A-Za-z0-9-]+)*")
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def api_name(tool_id: str) -> str:
    flat = tool_id.replace("/", "__")
    if _READABLE.fullmatch(tool_id) and len(flat) <= MAX_LEN:
        return flat
    digest = hashlib.sha256(tool_id.encode("utf-8")).hexdigest()[:12]
    return _UNSAFE.sub("_", flat)[:49] + "___" + digest
