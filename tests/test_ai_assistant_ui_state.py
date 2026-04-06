from reserving_app.services.ai_assistant import resolve_ai_request_state


def test_resolve_ai_request_state_uses_preset_and_autosubmits():
    question, submit = resolve_ai_request_state("", "Summarise results", False)
    assert question == "Summarise results"
    assert submit is True


def test_resolve_ai_request_state_manual_submit_requires_question():
    question, submit = resolve_ai_request_state("Explain uncertainty", None, True)
    assert question == "Explain uncertainty"
    assert submit is True
