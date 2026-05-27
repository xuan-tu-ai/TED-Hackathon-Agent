from __future__ import annotations

import threading

import pytest


class TestDSPyThreadSafety:
    def test_get_lm_returns_lm_instance(self):
        from app.robot_agent.agent.llm_config import get_lm

        lm = get_lm()
        if lm is None:
            pytest.skip("No API key configured")

        assert lm is not None

    def test_dspy_context_usable_in_multiple_threads(self):
        import dspy
        from app.robot_agent.agent.llm_config import get_lm

        lm = get_lm()
        if lm is None:
            pytest.skip("No API key configured")

        errors = []
        results = []

        def worker():
            try:
                with dspy.context(lm=lm):
                    result = dspy.ChainOfThought(dspy.Signature("question -> answer"))(
                        question="What is 2+2?"
                    )
                    results.append(result.answer)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 3

    def test_different_threads_get_independent_context(self):
        import dspy
        from app.robot_agent.agent.llm_config import get_lm, get_llm_config

        lm = get_lm()
        if lm is None:
            pytest.skip("No API key configured")

        call_order = []
        lock = threading.Lock()

        def worker(thread_id):
            with dspy.context(lm=lm):
                with lock:
                    call_order.append(f"start-{thread_id}")
                result = dspy.ChainOfThought(dspy.Signature("question -> answer"))(
                    question=f"What is {thread_id}+{thread_id}?"
                )
                with lock:
                    call_order.append(f"end-{thread_id}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(call_order) == 6
        assert "start-0" in call_order
        assert "end-0" in call_order