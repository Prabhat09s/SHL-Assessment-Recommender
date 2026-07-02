from typing import List
from .models import ChatMessage


def build_query(messages: List[ChatMessage], max_chars: int = 2000) -> str:
    """
    Builds a retrieval query from the conversation so far.

    We weight the most recent user turn heaviest (it usually carries the
    latest constraint - a refine, a new fact, etc.) while still including
    earlier user turns for context, since a shortlist-worthy query is
    often built up across several turns (see trace C9: JD -> backend vs
    frontend -> seniority, each turn adds a constraint).
    """
    user_turns = [m.content for m in messages if m.role == "user"]
    if not user_turns:
        return ""

    latest = user_turns[-1]
    earlier = " ".join(user_turns[:-1])

    # latest turn repeated for extra weight in the TF-IDF query
    query = f"{latest} {latest} {earlier}"
    return query[:max_chars]
