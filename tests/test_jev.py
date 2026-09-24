"""Jev decision layer without the network: state shape, option masking, parsing, policy, client fallback."""

import json
import threading
import time

import httpx
import pytest

from tyto_voice.jev import (
    ACTIONS,
    ADAPT_QUIETLY,
    ASK_AFTER_SENTENCE,
    ASK_NOW,
    STAY_SILENT,
    TYPESAFE_BASE_URL,
    JevJudge,
    Situation,
    apply_policy,
    bucket_seconds,
    build_questions,
    build_state,
    fallback,
    keywords,
    parse_answer,
)


def sit(**over) -> Situation:
    base = dict(
        cause="noise", label="Noise", problem="loud background noise", severity="severe",
        lasting_s=11.0, trend="getting worse", agent_speaking=True, caller_speaking=False,
        agent_last_words="your order number is four seven two nine", caller_last_words="yes",
        since_ask_s=None, times_asked=0,
    )
    base.update(over)
    return Situation(**base)


def body(choice, confidence, fixing=0.05, detail=0.05) -> dict:
    return {
        "model": "typesafe-ai/jev",
        "answers": {
            "action": {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": {choice: confidence}},
            "caller_fixing": {"type": "noul", "noul": fixing},
            "detail_in_flight": {"type": "noul", "noul": detail},
        },
        "usage": {"input_tokens": 600, "output_tokens": 90},
    }


def leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)
    else:
        yield obj


# -- state and questions ---------------------------------------------------- #


def test_state_is_named_buckets_without_raw_numbers():
    state = build_state(sit())
    assert not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in leaves(state))
    assert state["audio_problem"]["severity"] == "severe"
    assert state["audio_problem"]["lasting"] == "about 10 seconds"
    assert state["audio_problem"]["trend"] == "getting worse"
    assert state["history"]["user_last_asked_to_fix_this"] == "never"
    assert state["history"]["times_asked_so_far"] == "never"
    assert "four seven two nine" in state["call"]["agent_last_words"]
    json.dumps(state)  # serialisable as sent


def test_bucket_seconds():
    assert bucket_seconds(None) == "never"
    assert bucket_seconds(1, now_word="just started") == "just started"
    assert bucket_seconds(5) == "a few seconds"
    assert bucket_seconds(12) == "about 10 seconds"
    assert bucket_seconds(21) == "about 20 seconds"
    assert bucket_seconds(70) == "about a minute"
    assert bucket_seconds(500) == "several minutes"


def test_questions_mask_finish_sentence_when_agent_silent():
    q = build_questions(sit(agent_speaking=False))
    assert set(q["action"]["criteria"]) == {ASK_NOW, ADAPT_QUIETLY, STAY_SILENT}
    assert set(build_questions(sit(agent_speaking=True))["action"]["criteria"]) == set(ACTIONS)
    assert {q[k]["type"] for k in q} == {"choice", "noul"}
    assert set(q) == {"action", "caller_fixing", "detail_in_flight"}


# -- parsing and policy ----------------------------------------------------- #


def test_parse_answer_reads_the_typesafe_shape():
    d = parse_answer(body(ASK_AFTER_SENTENCE, 0.83, fixing=0.08, detail=0.96), latency_ms=300)
    assert d.action == ASK_AFTER_SENTENCE and d.confidence == 0.83
    assert d.caller_fixing == 0.08 and d.detail_in_flight == 0.96
    assert d.latency_ms == 300 and d.source == "jev"


def test_parse_rejects_unknown_action():
    with pytest.raises(ValueError):
        parse_answer(body("handover", 0.9))


def test_policy_low_confidence_falls_back_to_the_rule():
    d = apply_policy(parse_answer(body(STAY_SILENT, 0.10)), sit())
    assert d.action == ASK_NOW and "low confidence" in d.reason


def test_policy_caller_already_fixing_stays_silent():
    d = apply_policy(parse_answer(body(ASK_NOW, 0.9, fixing=0.8)), sit(caller_last_words="hang on, closing the window"))
    assert d.action == STAY_SILENT


def test_policy_mid_detail_waits_for_the_sentence_only_while_speaking():
    assert apply_policy(parse_answer(body(ASK_NOW, 0.9, detail=0.9)), sit(agent_speaking=True)).action == ASK_AFTER_SENTENCE
    assert apply_policy(parse_answer(body(ASK_NOW, 0.9, detail=0.9)), sit(agent_speaking=False)).action == ASK_NOW


def test_policy_leaves_a_fallback_alone():
    d = apply_policy(fallback("ReadTimeout"), sit())
    assert d.action == ASK_NOW and d.source == "fallback" and d.reason == "ReadTimeout"


def test_keywords_pick_distinctive_words_of_a_nudge():
    words = keywords("Sorry, there is a lot of background noise. Could you move somewhere quieter?")
    assert {"background", "quieter", "somewhere"} <= words and "noise" not in words


# -- client ----------------------------------------------------------------- #


def test_judge_posts_state_and_questions_to_the_gateway():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json=body(ASK_NOW, 0.9))

    judge = JevJudge("vck_test", transport=httpx.MockTransport(handler))
    d = judge.evaluate(sit(agent_speaking=False))
    assert seen["url"] == "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    assert seen["auth"] == "Bearer vck_test"
    assert seen["json"]["model"] == "typesafe-ai/jev"
    assert set(seen["json"]["questions"]) == {"action", "caller_fixing", "detail_in_flight"}
    assert seen["json"]["state"]["call"]["agent_is_speaking"] is False
    assert d.action == ASK_NOW and d.source == "jev" and d.latency_ms >= 0
    judge.close()


def test_judge_direct_typesafe_base_picks_jev_latest():
    judge = JevJudge("ts_key", base_url=TYPESAFE_BASE_URL, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    assert judge.model == "jev-latest"
    judge.close()


def test_judge_falls_back_on_timeout_and_reports_it():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    logs = []
    judge = JevJudge("k", transport=httpx.MockTransport(handler), on_log=lambda k, t: logs.append((k, t)))
    d = judge.evaluate(sit())
    assert d.source == "fallback" and d.action == ASK_NOW and "ReadTimeout" in d.reason
    assert any(k == "jev.fallback" for k, _ in logs)
    judge.close()


def test_judge_ask_is_non_blocking_and_single_flight():
    gate, done, got = threading.Event(), threading.Event(), []

    def handler(request):
        gate.wait(2)
        return httpx.Response(200, json=body(ASK_NOW, 0.9))

    judge = JevJudge("k", transport=httpx.MockTransport(handler))
    assert judge.ask(sit(), lambda d, s: (got.append((d, s)), done.set())) is True
    assert judge.ask(sit(), lambda d, s: None) is False  # one in flight at a time
    assert judge.busy
    gate.set()
    assert done.wait(3)
    assert got[0][0].action == ASK_NOW and got[0][1].cause == "noise"
    for _ in range(50):  # busy clears right after the callback
        if not judge.busy:
            break
        time.sleep(0.02)
    assert not judge.busy
    judge.close()
