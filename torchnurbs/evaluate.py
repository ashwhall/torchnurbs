def evaluate_1d(N, points):
    N_u = N[0]
    N_u = N_u[..., None]

    points = N_u * points
    points = points.sum(1)

    return points


def evaluate(N, points):
    if points.ndim == 2:
        return evaluate_1d(N, points)

    N_u, N_v = N
    N_u = N_u[..., None, None]
    points = N_u * points
    points = points.sum(1)

    N_v = N_v[None, ..., None]
    points = points[:, None]
    points = N_v * points
    points = points.sum(2)

    return points.reshape(-1, points.shape[-1])
