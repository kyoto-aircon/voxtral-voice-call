def list_account_voices(*, base_url: str, api_key: str, timeout: int = 10) -> list:
    """GET /v1/audio/voices?limit=100 — every voice this account can use (presets + Studio clones).

    Returns ``[{"id", "name", "kind"}]``; raises on HTTP error (caller decides fallback).
    """
    url = base_url.rstrip("/") + "/v1/audio/voices?limit=100"
    req = urllib.request.Request(url, headers=_headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    out = []
    for voice in payload.get("data") or []:
        if isinstance(voice, dict) and voice.get("id"):
            out.append({
                "id": str(voice["id"]),
                "name": str(voice.get("name") or voice["id"]),
                "kind": str(voice.get("kind") or ""),
            })
    return out
