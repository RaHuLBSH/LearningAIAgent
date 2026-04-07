import logging
import os
import re
import time
import uuid
from typing import Dict, Any, List, Tuple, Optional

from .state import LearningState, DeckCard, QuizState, QuizAnswered

logger = logging.getLogger(__name__)

LLM_ERROR_RATE_LIMIT = "[[LLM_ERROR:RATE_LIMIT]]"
LLM_ERROR_OTHER = "[[LLM_ERROR:OTHER]]"


def _extract_topic_from_user_text(text: str) -> str:
    m = re.search(r"^(?:teach me|explain|i want to learn)\s+(.+)$", text.strip(), re.IGNORECASE)
    topic = (m.group(1) if m else text).strip()
    return topic or "General knowledge"


def _is_topic_switch_message(text: str) -> bool:
    stripped = (text or "").strip()
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in ["change topic", "switch topic", "new topic"]):
        return True
    if re.match(r"^teach me\s+.+$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^explain\s+.+$", stripped, re.IGNORECASE):
        return lowered not in {"explain simpler", "explain simple"}
    return False


def call_llm(prompt: str) -> str:
    """
    Wrapper around the LLM.

    Provider selection:
    - If `GROQ_API_KEY` is set, uses Groq via the OpenAI-compatible endpoint.
    - Else if `OPENAI_API_KEY` is set, uses OpenAI.
    - Else falls back to a simple mock so the POC still runs.
    """
    try:
        from openai import OpenAI  # type: ignore

        # optional: these exception classes exist in the OpenAI SDK; use getattr-safe import pattern
        try:
            from openai import RateLimitError  # type: ignore
        except Exception:  # noqa: BLE001
            RateLimitError = Exception  # type: ignore[misc,assignment]

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        else:
            client = OpenAI(api_key=openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # basic retry/backoff for 429s
        backoffs = [0.5, 1.0, 2.0]
        for i, sleep_s in enumerate([0.0] + backoffs):
            if sleep_s:
                time.sleep(sleep_s)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as exc:  # type: ignore[misc]
                logger.warning("Rate limited by LLM provider (attempt %s/%s): %s", i + 1, len(backoffs) + 1, exc)
                if i == len(backoffs):
                    return LLM_ERROR_RATE_LIMIT
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call failed (attempt %s/%s): %s", i + 1, len(backoffs) + 1, exc)
                if i == len(backoffs):
                    return LLM_ERROR_OTHER
                continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM unavailable, using mock. %s", exc)
        # very simple mock behaviour for offline/demo usage
        # Try to keep mock responses topic-aware (avoid DP-specific defaults).
        topic_match = re.search(r"(?:Main topic|Topic|Subtopic)\s*:\s*(.+)", prompt, re.IGNORECASE)
        inferred_topic = topic_match.group(1).strip() if topic_match else "the topic"

        if "multiple-choice quiz" in prompt.lower() or "a) <option>" in prompt.lower():
            return (
                f"Question: Which option best relates to {inferred_topic}?\n"
                "A) A core definition\n"
                "B) A common misconception\n"
                "C) An unrelated concept\n"
                "D) A random fact\n"
                "Answer: A"
            )
        if "structured learning card" in prompt.lower() or "title:" in prompt.lower():
            return (
                f"TITLE: {inferred_topic} (Overview)\n"
                "SUMMARY: Here is a simple, beginner-friendly overview.\n"
                "KEY_POINTS:\n"
                "- What it is\n"
                "- Why it matters\n"
                "- A practical example\n"
                "EXAMPLE:\n"
                "A small real-world example goes here.\n"
                "COMMON_MISTAKES:\n"
                "- Confusing related terms\n"
                "- Skipping the basics\n"
                "END"
            )
        if "evaluate the following answer" in prompt.lower():
            return "Feedback: I couldn't evaluate precisely in mock mode.\nResult: INCORRECT"
        return "This is a mock LLM response used for the POC."


# ---------- Prompt helpers ----------


def _learning_card_prompt(topic: str, subtopic: str, level: str, extra_instruction: str = "") -> str:
    simpler = "Use very simple language." if level == "simpler" else "Use clear, beginner-friendly language."
    deeper = "Add more intuition and one extra example." if level == "deeper" else "Keep it focused."
    return (
        "You are a helpful tutor. Write a structured learning card.\n"
        f"Topic: {topic}\n"
        f"Subtopic: {subtopic}\n"
        f"Style: {simpler} {deeper}\n"
        f"{extra_instruction}\n\n"
        "Return exactly this structure with these headings:\n"
        "TITLE: <short title>\n"
        "SUMMARY: <2-4 sentences>\n"
        "KEY_POINTS:\n"
        "- <bullet>\n"
        "- <bullet>\n"
        "- <bullet>\n"
        "EXAMPLE:\n"
        "<a small concrete example>\n"
        "COMMON_MISTAKES:\n"
        "- <bullet>\n"
        "- <bullet>\n"
        "END\n"
    )


def _deck_outline_prompt(topic: str, n: int) -> str:
    return (
        "Create a concise learning outline.\n"
        f"Topic: {topic}\n"
        f"Number of cards: {n}\n\n"
        "Return ONLY a numbered list of subtopics for learning cards.\n"
        "Rules:\n"
        "- Avoid generic 'summary'/'next steps' items.\n"
        "- Avoid meta headings like 'introduction' unless it is concrete.\n"
        "- Each subtopic should be concrete and teachable.\n"
        "- Use simple phrasing.\n"
    )


def _parse_numbered_list(text: str) -> List[str]:
    items: List[str] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*\d+\.\s*(.+)\s*$", line)
        if m:
            items.append(m.group(1).strip())
    return [i for i in items if i]


def _mcq_from_card_prompt(topic: str, card_text: str) -> str:
    return (
        "Create a high-quality multiple-choice question (MCQ) based ONLY on the learning card below.\n"
        f"Topic: {topic}\n\n"
        "Learning card:\n"
        f"{card_text}\n\n"
        "Return in exactly this format:\n"
        "Question: <one question>\n"
        "A) <option>\n"
        "B) <option>\n"
        "C) <option>\n"
        "D) <option>\n"
        "Answer: <A|B|C|D>\n"
        "Rules:\n"
        "- Options must be specific and non-generic.\n"
        "- Exactly one correct answer.\n"
        "- Avoid 'All of the above/None of the above'.\n"
    )


# ---------- Node implementations ----------


def router_node(state: LearningState) -> LearningState:
    """
    Strict router for deck-based flow.

    - If mode == quiz: only allow topic switch or A/B/C/D answers.
    - In learning/results: map typed intents to Learn more / Explain simpler / Ask me / Results.
    - Otherwise treat as a doubt about the current card.
    """
    user_msg = (state.get("latest_user_message") or "").strip()
    msg_lower = user_msg.lower()

    state["current_step"] = "router"
    state["new_topic"] = ""

    mode = state.get("mode", "learning")

    if any(k in msg_lower for k in ["explain simpler", "explain simple", "simplify", "too hard"]):
        state["interaction_type"] = "explain_simpler"
        state["current_step"] = "simplify_card"
        return state

    if any(k in msg_lower for k in ["tell me more", "learn more", "next", "continue"]):
        state["interaction_type"] = "learn_more"
        state["current_step"] = "render_card"
        return state

    if any(k in msg_lower for k in ["ask me", "quiz", "question"]):
        state["interaction_type"] = "ask_me"
        state["current_step"] = "quiz_prompt"
        return state

    if any(k in msg_lower for k in ["results", "score", "show results"]):
        state["interaction_type"] = "results"
        state["current_step"] = "results"
        return state

    # topic switching (allowed always)
    if _is_topic_switch_message(user_msg):
        m = re.search(r"(?:change topic to|switch topic to|new topic|teach me|explain)\s+(.*)", user_msg, re.IGNORECASE)
        state["interaction_type"] = "topic_switch"
        state["new_topic"] = (m.group(1).strip() if m and m.group(1) else _extract_topic_from_user_text(user_msg))
        state["current_step"] = "switch_topic"
        return state

    # quiz mode: only allow answers
    if mode == "quiz":
        m = re.search(r"\b([ABCD])\b", user_msg.upper())
        if m:
            state["interaction_type"] = "quiz_answer"
            state["current_step"] = "quiz_evaluate"
            return state
        state["interaction_type"] = "unknown"
        state["current_step"] = "end"
        state.setdefault("history", []).append(
            {"role": "assistant", "type": "system", "content": "Please answer with A/B/C/D, or change the topic."}
        )
        return state

    state["interaction_type"] = "doubt"
    state["current_step"] = "doubt_answer"
    return state


def intent_node(state: LearningState) -> LearningState:
    """
    Extract the learning topic from the user's message and initialize core fields.
    """
    logger.debug("Intent node, state: %s", state)
    user_msg = state.get("latest_user_message", "")

    # very simple extraction: look for 'teach me X' pattern, fallback to full message
    match = re.search(r"teach me (.+)", user_msg, re.IGNORECASE)
    topic = match.group(1).strip() if match else user_msg.strip()
    if not topic:
        topic = "Programming fundamentals"

    # Start a fresh learning journey for this topic (topic switch behaves the same)
    state["topic"] = topic
    state["learning_plan"] = []
    state["current_topic_index"] = 0
    state["attempts"] = 0
    state["score"] = 0
    state["pending_user_action"] = "none"
    state["explanation_level"] = "normal"
    state["quiz"] = {
        "type": "mcq",
        "question": "",
        "choices": [],
        "answer_key": None,
    }
    state["current_step"] = "plan"
    state["history"] = state.get("history", [])
    state["history"].append({"role": "user", "content": user_msg})
    return state


def planner_node(state: LearningState) -> LearningState:
    """
    Build a simple learning plan given a topic.
    """
    logger.debug("Planner node, state: %s", state)
    topic = state.get("topic", "Programming")

    # very lightweight rule-based planner as a fallback
    default_plan: List[str] = [
        f"Introduction to {topic}",
        f"Core concepts of {topic}",
        f"Common patterns in {topic}",
        f"Practice problems for {topic}",
        f"Summary and next steps for {topic}",
    ]

    state["learning_plan"] = default_plan
    state["current_topic_index"] = 0
    state["current_step"] = "teach"
    return state


def teach_node(state: LearningState) -> LearningState:
    """
    Explain the current topic/subtopic to the learner.
    """
    logger.debug("Teach node, state: %s", state)
    prompt = build_explanation_prompt(state)
    explanation = call_llm(prompt)

    if explanation in (LLM_ERROR_RATE_LIMIT, LLM_ERROR_OTHER):
        msg = (
            "I'm temporarily rate-limited by the LLM provider. Please wait a few seconds and try again."
            if explanation == LLM_ERROR_RATE_LIMIT
            else "I couldn't reach the LLM provider. Please retry in a moment."
        )
        state["history"] = state.get("history", [])
        state["history"].append({"role": "assistant", "type": "system", "content": msg})
        state["last_ai_message_type"] = "system"
        state["current_step"] = "end"
        return state

    state["history"] = state.get("history", [])
    state["history"].append({"role": "assistant", "type": "teaching", "content": explanation})
    state["last_ai_message_type"] = "teach"
    state["current_step"] = "quiz_build"
    return state


def _build_mcq_prompt(state: LearningState) -> str:
    topic = state.get("topic", "the topic")
    plan = state.get("learning_plan") or []
    idx = state.get("current_topic_index", 0)
    subtopic = plan[idx] if 0 <= idx < len(plan) else topic
    return (
        "Create a multiple-choice quiz.\n"
        f"Topic: {topic}\n"
        f"Subtopic: {subtopic}\n\n"
        "Return in exactly this format:\n"
        "Question: <one question>\n"
        "A) <option>\n"
        "B) <option>\n"
        "C) <option>\n"
        "D) <option>\n"
        "Answer: <A|B|C|D>\n"
        "Keep distractors plausible and only one correct answer."
    )


def _parse_mcq(text: str) -> Tuple[str, List[str], str] | None:
    q = re.search(r"^\s*(?:[-*]\s*)?(?:\*\*)?question(?:\*\*)?\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not q:
        return None
    question = q.group(1).strip()
    options: List[str] = []
    for letter in ["A", "B", "C", "D"]:
        m = re.search(
            rf"^\s*(?:[-*]\s*)?(?:\*\*)?{letter}(?:\*\*)?\s*[\).:-]\s*(.+?)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not m:
            return None
        options.append(m.group(1).strip())
    ans = re.search(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?answer(?:\*\*)?\s*:\s*([ABCD])(?:[\).:\s-].*)?$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not ans:
        return None
    return question, options, ans.group(1).upper()


def _find_next_unanswered_index(deck: List[DeckCard], answered: Dict[str, QuizAnswered], start: int = 0) -> Optional[int]:
    if not deck:
        return None

    safe_start = max(0, min(int(start or 0), len(deck) - 1))
    ordered_indexes = list(range(safe_start, len(deck))) + list(range(0, safe_start))
    for idx in ordered_indexes:
        card = deck[idx]
        cid = card.get("card_id", "")
        has_quiz = bool((card.get("quiz") or {}).get("question"))
        if has_quiz and cid and cid not in answered:
            return idx
    return None


def quiz_builder_node(state: LearningState) -> LearningState:
    """
    Build either an MCQ or short-answer quiz question and store it in state.

    This node is a \"pause point\". It sets `pending_user_action=\"awaiting_answer\"`
    so the next user message is treated as the quiz answer.
    """
    logger.debug("Quiz builder node, state: %s", state)

    # MCQ-only mode
    raw = call_llm(_build_mcq_prompt(state))
    if raw in (LLM_ERROR_RATE_LIMIT, LLM_ERROR_OTHER):
        msg = (
            "I'm rate-limited right now, so I can't generate a quiz. Please retry in a few seconds."
            if raw == LLM_ERROR_RATE_LIMIT
            else "I couldn't generate a quiz due to an LLM connectivity error. Please retry."
        )
        state["history"] = state.get("history", [])
        state["history"].append({"role": "assistant", "type": "system", "content": msg})
        state["last_ai_message_type"] = "system"
        state["pending_user_action"] = "none"
        state["current_step"] = "end"
        return state
    parsed = _parse_mcq(raw)
    if parsed:
        question, options, answer_key = parsed
    else:
        topic = state.get("topic", "the topic")
        plan = state.get("learning_plan") or []
        idx = state.get("current_topic_index", 0)
        subtopic = plan[idx] if 0 <= idx < len(plan) else topic
        question = f"Which option best matches a key idea in: {subtopic}?"
        options = [
            "A correct, relevant statement",
            "A common misconception",
            "An unrelated statement",
            "An overly specific edge case",
        ]
        answer_key = "A"

    quiz_text = "Question: " + question + "\n" + "\n".join([f"{l}) {o}" for l, o in zip(["A", "B", "C", "D"], options)])
    state["quiz"] = {"type": "mcq", "question": question, "choices": options, "answer_key": answer_key}
    state["last_question"] = question
    state["history"] = state.get("history", [])
    state["history"].append({"role": "assistant", "type": "quiz", "quiz_type": "mcq", "content": quiz_text, "choices": options})

    state["last_ai_message_type"] = "quiz"
    state["pending_user_action"] = "awaiting_answer"
    state["current_step"] = "end"
    return state


def quiz_node(state: LearningState) -> LearningState:
    """
    Generate a question for the current subtopic and store it.
    """
    logger.debug("Quiz node, state: %s", state)
    prompt = build_quiz_prompt(state)
    question = call_llm(prompt)

    state["last_question"] = question
    state["history"] = state.get("history", [])
    state["history"].append({"role": "assistant", "type": "quiz", "content": question})
    state["current_step"] = "evaluate"
    return state


def evaluate_node(state: LearningState) -> LearningState:
    """
    Evaluate the user's answer and update score, attempts, and weak topics.
    """
    logger.debug("Evaluate node, state: %s", state)
    answer = state.get("latest_user_message", "")

    quiz = state.get("quiz") or {}
    quiz_type = quiz.get("type", "mcq")
    question = quiz.get("question") or state.get("last_question", "")

    # MCQ-only evaluation
    state["attempts"] = state.get("attempts", 0) + 1
    user_choice = (answer or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", user_choice)
    picked = m.group(1) if m else (user_choice[:1] if user_choice else "")
    correct_key = (quiz.get("answer_key") or "").strip().upper()
    is_correct = picked == correct_key and correct_key in ["A", "B", "C", "D"]
    evaluation = ("✅ Correct.\n" if is_correct else "❌ Not quite.\n") + f"Result: {'CORRECT' if is_correct else 'INCORRECT'}"
    state["score"] = state.get("score", 0) + (1 if is_correct else 0)

    # mark weak topics if wrong multiple times
    plan = state.get("learning_plan") or []
    idx = state.get("current_topic_index", 0)
    current_subtopic = plan[idx] if 0 <= idx < len(plan) else state.get("topic", "")

    if not is_correct and state.get("attempts", 0) >= 2 and current_subtopic:
        weak = state.get("weak_topics", [])
        if current_subtopic not in weak:
            weak.append(current_subtopic)
            state["weak_topics"] = weak

    state["history"] = state.get("history", [])
    state["history"].append(
        {
            "role": "assistant",
            "type": "evaluation",
            "content": evaluation,
            "correct": is_correct,
        }
    )
    state["last_ai_message_type"] = "feedback"
    state["pending_user_action"] = "none"
    state["current_step"] = "decision"
    return state


def decision_node(state: LearningState) -> LearningState:
    """
    Decide what to do next based on correctness, attempts, and progress.
    """
    logger.debug("Decision node, state: %s", state)

    plan = state.get("learning_plan") or []
    idx = state.get("current_topic_index", 0)
    attempts = state.get("attempts", 0)
    score = state.get("score", 0)

    # Determine if last answer was correct from history
    last_eval = None
    for item in reversed(state.get("history", [])):
        if item.get("type") == "evaluation":
            last_eval = item
            break

    correct = bool(last_eval and last_eval.get("correct"))

    # Safety guard: if user is stuck too long on a topic, move on (POC should not infinite-loop)
    if not correct and attempts >= MAX_ATTEMPTS_PER_TOPIC:
        plan = state.get("learning_plan") or []
        idx = state.get("current_topic_index", 0)
        current_subtopic = plan[idx] if 0 <= idx < len(plan) else state.get("topic", "")
        if current_subtopic:
            weak = state.get("weak_topics", [])
            if current_subtopic not in weak:
                weak.append(current_subtopic)
                state["weak_topics"] = weak

        next_idx = idx + 1
        if next_idx < len(plan):
            state["current_topic_index"] = next_idx
            state["attempts"] = 0
            state["explanation_level"] = "normal"
            state["history"].append(
                {
                    "role": "assistant",
                    "type": "note",
                    "content": "Let's move on for now and come back later—I've marked this as a weak area.",
                }
            )
            state["current_step"] = "teach"
            return state

        state["current_step"] = "end"
        state["history"].append(
            {
                "role": "assistant",
                "type": "summary",
                "content": "We'll stop here for now. I've recorded weak topics so we can revisit them next time.",
            }
        )
        return state

    # Incorrect: go back to teach with simpler explanation
    if not correct:
        state["explanation_level"] = "simpler"
        state["current_step"] = "teach"
        return state

    # Correct but still early attempts: another quiz on same topic
    if correct and attempts < 2:
        state["current_step"] = "quiz_build"
        return state

    # Consistent success -> move to next topic
    next_idx = idx + 1
    if next_idx < len(plan):
        state["current_topic_index"] = next_idx
        state["attempts"] = 0
        state["explanation_level"] = "normal"
        state["current_step"] = "teach"
        return state

    # All topics complete -> end
    state["current_step"] = "end"
    state["history"].append(
        {
            "role": "assistant",
            "type": "summary",
            "content": f"You have completed the plan with score {score}/{len(plan) or 1}. Great job!",
        }
    )
    return state


def meta_action_node(state: LearningState) -> LearningState:
    """
    Handle meta/help actions without advancing the learning plan.
    """
    msg = (state.get("latest_user_message") or "").strip().lower()
    state["current_step"] = "meta"

    if any(k in msg for k in ["explain simpler", "simplify", "too hard"]):
        state["explanation_level"] = "simpler"
        state["pending_user_action"] = "none"
        return teach_node(state)

    if any(k in msg for k in ["learn more", "go deeper", "more detail", "more details", "explain more"]):
        state["explanation_level"] = "deeper"
        state["pending_user_action"] = "none"
        return teach_node(state)

    if any(k in msg for k in ["progress", "plan", "what's next", "whats next"]):
        plan = state.get("learning_plan") or []
        idx = state.get("current_topic_index", 0)
        pct = compute_progress(state)
        next_item = plan[idx] if 0 <= idx < len(plan) else "(none)"
        content = (
            f"Progress: {int(round(pct))}%\n"
            f"Current topic: {state.get('topic','')}\n"
            f"Next subtopic: {next_item}\n\n"
        )
        if plan:
            content += "Plan:\n- " + "\n- ".join(plan)
        else:
            content += 'No plan yet. Try: "Teach me <topic>".'

        state["history"] = state.get("history", [])
        state["history"].append({"role": "assistant", "type": "system", "content": content})
        state["last_ai_message_type"] = "system"
        state["current_step"] = "end"
        return state

    help_text = (
        "You can say:\n"
        '- "Teach me <topic>"\n'
        '- "Explain simpler"\n'
        '- "Learn more"\n'
        '- "Show my plan"\n'
        '- "Change topic to <new topic>"\n\n'
        "If I ask an MCQ, reply with A/B/C/D."
    )
    state["history"] = state.get("history", [])
    state["history"].append({"role": "assistant", "type": "system", "content": help_text})
    state["last_ai_message_type"] = "system"
    state["current_step"] = "end"
    return state


def switch_topic_node(state: LearningState) -> LearningState:
    """
    Prepare a topic switch by rewriting the message into a new \"Teach me ...\" request.
    SessionManager will preserve long-term memory snapshots.
    """
    new_topic = (state.get("new_topic") or "").strip()
    if not new_topic:
        new_topic = (state.get("latest_user_message") or "").strip()

    state["latest_user_message"] = f"Teach me {new_topic}"
    state["pending_user_action"] = "none"
    state["current_step"] = "intent"
    return state


def compute_progress(state: LearningState) -> float:
    """
    Progress percentage based on how many pre-decided quizzes are answered.
    """
    deck = state.get("deck") or []
    total = sum(1 for c in deck if (c.get("quiz") or {}).get("question"))
    if total <= 0:
        return 0.0
    answered = len(state.get("quiz_answered") or {})
    return max(0.0, min(100.0, (answered / total) * 100.0))


def deck_builder_node(state: LearningState) -> LearningState:
    """
    Build 3-5 learning cards and 3-5 MCQs upfront for the topic.
    """
    user_msg = state.get("latest_user_message", "")
    topic = _extract_topic_from_user_text(user_msg)

    state["topic"] = topic
    state["mode"] = "learning"
    state["score"] = 0
    state["deck"] = []
    state["card_index"] = 0
    state["quiz_index"] = 0
    state["weak_cards"] = []
    state["quiz_answered"] = {}
    state["pending_user_action"] = "awaiting_choice"
    state["current_step"] = "deck_build"

    # Decide number of cards (3-5). Keep deterministic for POC, but within required range.
    n = 4
    outline_raw = call_llm(_deck_outline_prompt(topic, n))
    subtopics = _parse_numbered_list(outline_raw)
    if len(subtopics) < 3:
        subtopics = [
            f"Basics of {topic}",
            f"Key concepts in {topic}",
            f"Practical applications of {topic}",
            f"Common mistakes in {topic}",
        ]
    subtopics = subtopics[:5]

    deck: List[DeckCard] = []
    for sub in subtopics:
        card_text = call_llm(_learning_card_prompt(topic, sub, "normal"))
        if card_text in (LLM_ERROR_RATE_LIMIT, LLM_ERROR_OTHER):
            state.setdefault("history", []).append(
                {"role": "assistant", "type": "system", "content": "I couldn't prepare the learning deck due to LLM limits. Please retry in a few seconds."}
            )
            state["current_step"] = "end"
            return state

        # Generate MCQ from card content (prevents generic placeholder options)
        quiz_raw = call_llm(_mcq_from_card_prompt(topic, card_text))
        parsed = _parse_mcq(quiz_raw) if quiz_raw not in (LLM_ERROR_RATE_LIMIT, LLM_ERROR_OTHER) else None
        if parsed:
            q, opts, key = parsed
            quiz: QuizState = {"type": "mcq", "question": q, "choices": opts, "answer_key": key}
        else:
            # fallback derived from subtopic, still non-generic
            quiz = {
                "type": "mcq",
                "question": f"In {sub}, which option is the best example of the main idea?",
                "choices": [
                    f"A correct example related to {sub}",
                    f"A common confusion about {sub}",
                    f"An example from a different topic (not {sub})",
                    f"An overly broad statement that doesn't apply to {sub}",
                ],
                "answer_key": "A",
            }

        deck.append(
            {
                "card_id": str(uuid.uuid4()),
                "title": sub,
                "content_structured": card_text,
                "status": "unseen",
                "quiz": quiz,
            }
        )

    state["deck"] = deck
    state["current_step"] = "render_card"
    return state


def _current_card(state: LearningState) -> Optional[DeckCard]:
    deck = state.get("deck") or []
    idx = int(state.get("card_index") or 0)
    if 0 <= idx < len(deck):
        return deck[idx]
    return None


def render_card_node(state: LearningState) -> LearningState:
    """
    Show current card. If interaction was learn_more, advance by one first.
    """
    deck = state.get("deck") or []
    if not deck:
        state.setdefault("history", []).append({"role": "assistant", "type": "system", "content": "No cards available. Try: Explain <topic>."})
        state["current_step"] = "end"
        return state

    if state.get("interaction_type") == "learn_more":
        cur = int(state.get("card_index") or 0)
        if cur >= len(deck) - 1:
            state["current_step"] = "quiz_prompt"
            return quiz_prompt_node(state)
        state["card_index"] = cur + 1

    card = _current_card(state)
    if not card:
        state.setdefault("history", []).append({"role": "assistant", "type": "system", "content": "No current card found."})
        state["current_step"] = "end"
        return state

    card["status"] = "seen"
    deck[int(state.get("card_index") or 0)] = card
    state["deck"] = deck

    state.setdefault("history", []).append({"role": "assistant", "type": "teaching", "content": card.get("content_structured", "")})
    state["last_ai_message_type"] = "teach"
    state["pending_user_action"] = "awaiting_choice"
    state["mode"] = "learning"
    state["current_step"] = "end"
    return state


def simplify_card_node(state: LearningState) -> LearningState:
    card = _current_card(state)
    if not card:
        state.setdefault("history", []).append({"role": "assistant", "type": "system", "content": "No current card to simplify."})
        state["current_step"] = "end"
        return state

    topic = state.get("topic") or "the topic"
    sub = card.get("title") or topic
    simplified = call_llm(_learning_card_prompt(topic, sub, "simpler", extra_instruction="Make it shorter and easier."))
    if simplified in (LLM_ERROR_RATE_LIMIT, LLM_ERROR_OTHER):
        state.setdefault("history", []).append({"role": "assistant", "type": "system", "content": "I'm rate-limited. Please retry simplifying in a few seconds."})
        state["current_step"] = "end"
        return state

    card["content_structured"] = simplified
    deck = state.get("deck") or []
    deck[int(state.get("card_index") or 0)] = card
    state["deck"] = deck

    state.setdefault("history", []).append({"role": "assistant", "type": "teaching", "content": simplified})
    state["last_ai_message_type"] = "teach"
    state["pending_user_action"] = "awaiting_choice"
    state["mode"] = "learning"
    state["current_step"] = "end"
    return state


def doubt_answer_node(state: LearningState) -> LearningState:
    card = _current_card(state)
    topic = state.get("topic") or "the topic"
    question = state.get("latest_user_message") or ""
    context = card.get("content_structured", "") if card else ""
    prompt = (
        "Answer the user's specific doubt based on the learning card context.\n"
        f"Topic: {topic}\n\n"
        f"Learning card context:\n{context}\n\n"
        f"User doubt: {question}\n\n"
        "Give a clear answer in simple language."
    )
    ans = call_llm(prompt)
    if ans in (LLM_ERROR_RATE_LIMIT, LLM_ERROR_OTHER):
        ans = "I'm rate-limited right now. Please retry your question in a few seconds."
        kind = "system"
    else:
        kind = "teaching"
    state.setdefault("history", []).append({"role": "assistant", "type": kind, "content": ans, "subtype": "doubt"})
    state["pending_user_action"] = "awaiting_choice"
    state["mode"] = "learning"
    state["current_step"] = "end"
    return state


def quiz_prompt_node(state: LearningState) -> LearningState:
    deck = state.get("deck") or []
    answered = state.get("quiz_answered") or {}
    if not deck:
        state.setdefault("history", []).append({"role": "assistant", "type": "system", "content": "No deck available. Start with: Explain <topic>."})
        state["current_step"] = "end"
        return state

    idx = _find_next_unanswered_index(deck, answered, start=int(state.get("card_index") or 0))
    if idx is None:
        state["current_step"] = "results"
        return results_node(state)

    state["quiz_index"] = idx

    card = deck[idx]
    quiz = card.get("quiz") or {}
    q = quiz.get("question", "")
    opts = quiz.get("choices") or []
    quiz_text = "Question: " + q + "\n" + "\n".join([f"{l}) {o}" for l, o in zip(["A", "B", "C", "D"], opts)])

    state.setdefault("history", []).append({"role": "assistant", "type": "quiz", "quiz_type": "mcq", "content": quiz_text, "choices": opts})
    state["mode"] = "quiz"
    state["pending_user_action"] = "awaiting_quiz_answer"
    state["current_step"] = "end"
    return state


def quiz_evaluate_node(state: LearningState) -> LearningState:
    deck = state.get("deck") or []
    qidx = int(state.get("quiz_index") or 0)
    if not deck or qidx >= len(deck):
        state["current_step"] = "results"
        return state

    card = deck[qidx]
    cid = card.get("card_id", "")
    quiz = card.get("quiz") or {}
    correct_key = (quiz.get("answer_key") or "").strip().upper()

    user_msg = (state.get("latest_user_message") or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", user_msg)
    picked = m.group(1) if m else (user_msg[:1] if user_msg else "")

    answered = state.get("quiz_answered") or {}
    rec: QuizAnswered = answered.get(cid, {"correct": False, "attempts": 0, "user_choice": ""})
    rec["attempts"] = int(rec.get("attempts") or 0) + 1
    rec["user_choice"] = picked
    is_correct = picked == correct_key and correct_key in ["A", "B", "C", "D"]
    rec["correct"] = is_correct
    answered[cid] = rec
    state["quiz_answered"] = answered

    if is_correct:
        state["score"] = int(state.get("score") or 0) + 1
        feedback = "✅ Correct."
    else:
        feedback = f"❌ Incorrect. Correct answer: {correct_key}."
        weak = state.get("weak_cards") or []
        if cid and cid not in weak:
            weak.append(cid)
        state["weak_cards"] = weak

    state.setdefault("history", []).append({"role": "assistant", "type": "evaluation", "content": feedback})
    state["mode"] = "learning"
    state["pending_user_action"] = "awaiting_choice"

    next_card_index = min(qidx + 1, max(len(deck) - 1, 0))
    state["card_index"] = next_card_index

    total_quizzes = sum(1 for c in deck if (c.get("quiz") or {}).get("question"))
    if len(answered) >= total_quizzes and total_quizzes > 0:
        state["current_step"] = "results"
        return results_node(state)

    upcoming_quiz_index = _find_next_unanswered_index(deck, answered, start=next_card_index)
    if upcoming_quiz_index is None:
        state["current_step"] = "results"
        return results_node(state)

    if upcoming_quiz_index == next_card_index:
        state["interaction_type"] = "learn_more"
        state["current_step"] = "render_card"
        return render_card_node(state)

    state["quiz_index"] = upcoming_quiz_index
    state["current_step"] = "quiz_prompt"
    return quiz_prompt_node(state)


def results_node(state: LearningState) -> LearningState:
    deck = state.get("deck") or []
    total = sum(1 for c in deck if (c.get("quiz") or {}).get("question"))
    score = int(state.get("score") or 0)
    pct = int(round(compute_progress(state)))

    weak_ids = state.get("weak_cards") or []
    weak_titles = [c.get("title") for c in deck if c.get("card_id") in weak_ids]
    summary = (
        f"RESULTS\n"
        f"Score: {score}/{total}\n"
        f"Progress: {pct}%\n"
        f"Weak cards: {', '.join([t for t in weak_titles if t]) or 'None'}\n"
    )
    state.setdefault("history", []).append({"role": "assistant", "type": "system", "content": summary})

    state["mode"] = "results"
    state["pending_user_action"] = "none"
    state["current_step"] = "end"
    return state


def switch_topic_node(state: LearningState) -> LearningState:
    new_topic = (state.get("new_topic") or "").strip()
    if not new_topic:
        new_topic = _extract_topic_from_user_text(state.get("latest_user_message") or "")
    state["latest_user_message"] = f"Explain {new_topic}"
    state["interaction_type"] = "topic_switch"
    state["current_step"] = "deck_build"
    return state
