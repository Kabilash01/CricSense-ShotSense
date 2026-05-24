def classify_boundary(ball_speed, bounced):
    """
    Broadcast logic:
    - If bounced before boundary → FOUR
    - If no bounce → SIX
    """
    if bounced:
        return "FOUR"
    return "SIX"
