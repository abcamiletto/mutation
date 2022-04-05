from .state import pack, unpack


def model(t, X, l, g, a, B):
    """Calculates dX/dt and returns it"""
    S, I, R, W = unpack(X)

    dSdt = -l.T @ I * S
    dIdt = l * I * S - g * I + B.T @ W * I
    dRdt = g * I - a * R
    dWdt = a * R - B @ I * W

    dXdt = [dSdt, dIdt, dRdt, dWdt]
    return pack(dXdt)
