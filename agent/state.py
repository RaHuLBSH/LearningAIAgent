from typing import List, TypedDict, Literal, Dict, Any, Optional


StepLiteral = Literal[
    "router",
    "deck_build",
    "render_card",
    "simplify_card",
    "doubt_answer",
    "quiz_prompt",
    "quiz_evaluate",
    "results",
    "switch_topic",
    "end",
]

InteractionType = Literal[
    "topic_switch",
    "learn_more",
    "explain_simpler",
    "ask_me",
    "quiz_answer",
    "doubt",
    "results",
    "unknown",
]

ModeLiteral = Literal["learning", "quiz", "results"]
PendingUserAction = Literal["none", "awaiting_choice", "awaiting_quiz_answer"]
AiMessageType = Literal["teach", "quiz", "feedback", "system", "results"]
QuizType = Literal["mcq"]

CardStatus = Literal["unseen", "seen", "quizzed"]

class QuizState(TypedDict, total=False):
    type: QuizType
    question: str
    choices: List[str]  # for MCQ: ["A) ...", "B) ...", ...] or raw option text
    answer_key: Optional[str]  # e.g. "B" for MCQ


class DeckCard(TypedDict, total=False):
    card_id: str
    title: str
    content_structured: str  # learning card text (structured headings)
    status: CardStatus
    quiz: QuizState


class QuizAnswered(TypedDict, total=False):
    correct: bool
    attempts: int
    user_choice: str


class LearningState(TypedDict, total=False):
    """
    LangGraph state for a single learning session.
    """

    # core identifiers
    user_id: str
    topic: str

    # workflow control
    current_step: StepLiteral
    mode: ModeLiteral

    # deck state
    deck: List[DeckCard]
    card_index: int
    quiz_index: int
    weak_cards: List[str]  # list of card_ids
    quiz_answered: Dict[str, QuizAnswered]  # card_id -> answer record

    # turn classification / pausing
    interaction_type: InteractionType
    pending_user_action: PendingUserAction
    last_ai_message_type: AiMessageType

    # scoring / progress (derived from pre-decided quizzes)
    score: int

    # memories
    history: List[Dict[str, Any]]

    # runtime
    latest_user_message: str
    explanation_level: Literal["normal", "simpler", "deeper"]
    new_topic: str


def initial_state(user_id: str, latest_user_message: str) -> LearningState:
    """
    Helper to create an initial empty state for a new session.
    """
    return LearningState(
        user_id=user_id,
        topic="",
        current_step="router",
        mode="learning",
        deck=[],
        card_index=0,
        quiz_index=0,
        weak_cards=[],
        quiz_answered={},
        score=0,
        history=[],
        latest_user_message=latest_user_message,
        explanation_level="normal",
        interaction_type="unknown",
        pending_user_action="awaiting_choice",
        last_ai_message_type="system",
        new_topic="",
    )

