def test_resolve_voice_id_accepts_ids_and_labels():
    from voxtral_voice_call.tts_provider import resolve_voice_id, VOXTRAL3_VOICES
    assert resolve_voice_id("voice_1788") == "voice_1788"
    assert resolve_voice_id("David (en-GB)") == "voice_1788"
    assert resolve_voice_id("en_paul_neutral") == "en_paul_neutral"
    assert resolve_voice_id("") == "en_paul_neutral"
    assert resolve_voice_id("custom-clone-xyz") == "custom-clone-xyz"
    assert len(VOXTRAL3_VOICES) >= 35
