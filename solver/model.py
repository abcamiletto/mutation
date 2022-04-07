from .state import pack, unpack


def model(t, X, l, g, a, B, D):
    """Calculates dX/dt and returns it"""
    S, I, R, W = unpack(X)
    dI, dR, dW = D[:, 0, None], D[:, 1, None], D[:, 2, None]

    d = dI.T @ I + dR.T @ R + dW.T @ W
    dSdt = -l.T @ I * S + d
    dIdt = l * I * S - g * I + B.T @ W * I - dI * I
    dRdt = g * I - a * R - dR * R
    dWdt = a * R - B @ I * W - dW * W

    dXdt = [dSdt, dIdt, dRdt, dWdt]
    return pack(dXdt)
