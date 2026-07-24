def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def sub(a, b):
    return [x - y for x, y in zip(a, b)]

def scale(v, c):
    return [c * x for x in v]

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    m = len(s_list)

    if m == 0:
        return [-x for x in gradient]

    rho = [0.0] * m
    alpha = [0.0] * m

    q = grad[:]
    for i in range(m - 1, -1, -1):
        rho[i] = 1.0 / dot(y_list[i], s_list[i])
        alpha[i] = rho[i] * dot(s_list[i], q)
        q = sub(q, scale(y_list[i], alpha[i]))

    # Initial Hessian scaling
    gamma = dot(s_list[-1], y_list[-1]) / dot(y_list[-1], y_list[-1])
    r = scale(q, gamma)

    # Forward loop
    for i in range(m):
        beta = rho[i] * dot(y_list[i], r)
        r = add(r, scale(s_list[i], alpha[i] - beta))

    return [-x for x in r]