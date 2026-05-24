def intersects(ball_box, boundary_box):
    """
    ball_box: (x1, y1, x2, y2)
    boundary_box: (x1, y1, x2, y2)
    """
    bx1, by1, bx2, by2 = ball_box
    lx1, ly1, lx2, ly2 = boundary_box

    # AABB intersection check
    return not (
        bx2 < lx1 or
        bx1 > lx2 or
        by2 < ly1 or
        by1 > ly2
    )
def classify_boundary(speed_kmph, has_bounced):
    """
    Cricket logic:
    - Bounce before boundary → FOUR
    - Direct hit → SIX
    """
    if has_bounced:
        return "FOUR"
    else:
        return "SIX"
