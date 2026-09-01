"""Headless end-to-end smoke test for streamlit_app.py using Streamlit's AppTest.

Usage:
    venv\\Scripts\\python.exe _smoke_test.py [question]
"""

import sys

from streamlit.testing.v1 import AppTest

QUESTION = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What does Section 8(1)(j) of the RTI Act 2005 protect?"
)


def main():
    at = AppTest.from_file("streamlit_app.py", default_timeout=300)
    at.run()
    assert not at.exception, f"Initial run raised: {at.exception}"
    print(f"[OK] app booted; exceptions={len(at.exception)}")

    # Try the sidebar example-button path first (deterministic), else chat_input.
    if len(sys.argv) <= 1 and len(at.button) > 0:
        print(f"[INFO] clicking first example button: {at.button[0].label}")
        at.button[0].click()
        at.run()
    else:
        # Fall back to driving the chat input directly.
        at.chat_input[0].set_value(QUESTION)
        at.run()

    assert not at.exception, f"Run after question raised: {at.exception}"

    msgs = at.session_state["messages"]
    print(f"[OK] session has {len(msgs)} message(s) after question")
    assert msgs[-1]["role"] == "assistant", "last message is not assistant"

    data = msgs[-1]["data"]
    print(f"[OK] refused={data['refused']} verified={data['verified']} "
          f"attempts={data['retrieval_attempts']} graph={data['graph_triggered']}")
    print(f"[OK] passages={len(data['passages'])}")
    print(f"[OK] answer preview: {data['answer'][:120]!r}")

    assert data["answer"], "empty answer"
    if not data["refused"]:
        assert data["verified"] is not None, "verified is None for non-refused answer"

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()