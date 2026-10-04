def dollar_neutral(direction, gross=1.0):
    return direction * gross / 2, -direction * gross / 2


def hedge_ratio(direction, beta, gross=1.0, orientation="A_on_B"):
    """
    Return (weight_A, weight_B).

    A_on_B:
        spread = log(A) - alpha - beta * log(B)

        LONG spread  -> Long A, Short B
        SHORT spread -> Short A, Long B

        weights proportional to:
            (1, -beta)

    B_on_A:
        spread = log(B) - alpha - beta * log(A)

        LONG spread  -> Long B, Short A
        SHORT spread -> Short B, Long A

        Expressed as (A, B), weights proportional to:
            (-beta, 1)
    """

    b = abs(float(beta))
    d = 1.0 + b

    if orientation == "A_on_B":
        wa = direction * gross / d
        wb = -direction * gross * b / d

    elif orientation == "B_on_A":
        wa = -direction * gross * b / d
        wb = direction * gross / d

    else:
        raise ValueError(
            f"Unknown orientation: {orientation}. "
            "Expected 'A_on_B' or 'B_on_A'."
        )

    return wa, wb


def beta_neutral(direction, beta_a, beta_b, gross=1.0):
    a = abs(float(beta_a))
    b = abs(float(beta_b))

    if a + b == 0:
        return dollar_neutral(direction, gross)

    wa = gross * b / (a + b)
    wb = gross * a / (a + b)

    return direction * wa, -direction * wb