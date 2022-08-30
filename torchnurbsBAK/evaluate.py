def do_eval(N, selected, degree):
    # print_shape("selected", selected)
    N_u = N[0].reshape(-1)
    N_u = N_u[(..., ) + (None,) * (selected.ndim - 1)]
    degree_u = degree[0]
    # print_shape("N_u", N_u)
    selected = N_u * selected
    _, *dims = selected.shape
    selected = selected.reshape(-1, degree_u + 1, *dims)
    selected = selected.sum(1)

    if selected.ndim == 2:
        # It's a curve, not a surface
        return selected
    N_v = N[1].reshape(-1)
    N_v = N_v[None, :, None]
    degree_v = degree[1]
    selected = N_v * selected
    eval_u, eval_v_spanned, ndim = selected.shape
    selected = selected.reshape(eval_u, -1, degree_v+1, ndim)
    selected = selected.sum(2)

    _, _, ndim = selected.shape
    points = selected.reshape(-1, ndim)
    return points
