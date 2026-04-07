import logging
from typing import Dict, Any

from .state import LearningState, initial_state
from .nodes import compute_progress
from .nodes import (
    router_node,
    deck_builder_node,
    render_card_node,
    simplify_card_node,
    doubt_answer_node,
    quiz_prompt_node,
    quiz_evaluate_node,
    results_node,
    switch_topic_node,
)

logger = logging.getLogger(__name__)

class SessionManager:
    """
    In-memory session and long-term memory manager.
    """

    def __init__(self):
        # short-term state per session
        self.session_state: Dict[str, LearningState] = {}
        # long-term memory POC
        self.long_term_memory: Dict[str, Dict[str, Any]] = {}

    def _run_turn(self, state: LearningState) -> LearningState:
        """
        Execute a single bounded turn based on `current_step`.
        This prevents request-time infinite loops and keeps one user message -> one response cycle.
        """
        step = state.get("current_step", "router")

        # Always route first (unless a node explicitly set a downstream step)
        if step == "router":
            state = router_node(state)
            step = state.get("current_step", "deck_build")

        if step == "switch_topic":
            state = switch_topic_node(state)
            step = state.get("current_step", "deck_build")

        if step == "deck_build":
            state = deck_builder_node(state)
            step = state.get("current_step", "render_card")
            if step == "render_card":
                state = render_card_node(state)
            return state

        if step == "render_card":
            return render_card_node(state)

        if step == "simplify_card":
            return simplify_card_node(state)

        if step == "doubt_answer":
            return doubt_answer_node(state)

        if step == "quiz_prompt":
            return quiz_prompt_node(state)

        if step == "quiz_evaluate":
            return quiz_evaluate_node(state)

        if step == "results":
            return results_node(state)

        return state

    def handle_message(self, session_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """
        Run one turn for a single user message, maintaining state in memory.
        """
        is_new_session = session_id not in self.session_state

        if session_id not in self.session_state:
            logger.info("Creating new session %s for user %s", session_id, user_id)
            self.session_state[session_id] = initial_state(user_id=user_id, latest_user_message=message)
        else:
            # update existing state with latest message
            st = self.session_state[session_id]

            # If this message looks like a topic switch, snapshot current progress into long-term memory.
            msg_lower = message.lower()
            is_switch = any(k in msg_lower for k in ["change topic", "switch topic", "new topic"]) or msg_lower.startswith("teach me ") or msg_lower.startswith("explain ")
            if is_switch:
                prev = {
                    "topic": st.get("topic"),
                    "progress": int(round(compute_progress(st))),
                    "score": st.get("score"),
                    "weak_cards": st.get("weak_cards"),
                }
                mem = self.long_term_memory.get(user_id, {})
                mem.setdefault("topic_history", [])
                mem["topic_history"].append(prev)
                self.long_term_memory[user_id] = mem

            st["latest_user_message"] = message
            st["history"] = st.get("history", [])
            st["history"].append({"role": "user", "content": message})
            st["current_step"] = "router"

        state = self.session_state[session_id]
        before_history_len = len(state.get("history", []))

        logger.info("Running turn for session %s, step=%s", session_id, state.get("current_step"))
        result: LearningState = self._run_turn(state)
        self.session_state[session_id] = result

        # update long-term memory POC
        mem = self.long_term_memory.get(user_id, {})
        mem.update({
            "topic": result.get("topic"),
            "history": result.get("history"),
            "score": result.get("score"),
            "weak_cards": result.get("weak_cards"),
        })
        self.long_term_memory[user_id] = mem

        # Build a topic progress list for UI (current + past topics)
        topic_progress = []
        # past topics
        for item in mem.get("topic_history", []) or []:
            topic_progress.append({"topic": item.get("topic") or "(unknown)", "progress": int(item.get("progress") or 0)})
        # current topic
        cur_topic = result.get("topic") or "(unknown)"
        cur_pct = int(round(compute_progress(result)))
        # replace if exists
        existing_idx = next((i for i, t in enumerate(topic_progress) if t["topic"] == cur_topic), None)
        if existing_idx is None:
            topic_progress.append({"topic": cur_topic, "progress": cur_pct})
        else:
            topic_progress[existing_idx] = {"topic": cur_topic, "progress": cur_pct}

        # Build reply from assistant messages produced in this turn only.
        # On a new session, user message is appended inside intent_node.
        history = result.get("history", [])
        start_idx = 0 if is_new_session else before_history_len
        turn_messages = history[start_idx:]
        assistant_messages = [str(item.get("content", "")) for item in turn_messages if item.get("role") == "assistant"]
        reply = "\n\n".join(msg for msg in assistant_messages if msg.strip())

        events = []
        for item in turn_messages:
            if item.get("role") != "assistant":
                continue
            events.append(
                {
                    "type": item.get("type", "message"),
                    "quiz_type": item.get("quiz_type"),
                    "content": item.get("content", ""),
                    "choices": item.get("choices"),
                }
            )

        debug = {
            "current_step": result.get("current_step"),
            "mode": result.get("mode"),
            "card_index": result.get("card_index"),
            "quiz_index": result.get("quiz_index"),
        }

        return {
            "reply": reply,
            "events": events,
            "topic_progress": topic_progress,
            "state": result,
            "debug": debug,
        }
