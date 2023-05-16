import dolfin as df


def assign_mixed_function(p, V, compartments):
    """Create a function in a mixed function-space with sub-function being
    assigned from a dictionray of functions living in the subspaces."""
    P = df.Function(V)
    for j in compartments:
        if not j in p:
            raise KeyError(f"Missing key {j} in p; p.keys() = {p.keys()}")

    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Pint = [df.interpolate(p[j], subspaces[idx]) for idx, j in enumerate(compartments)]
    assigner = df.FunctionAssigner(V, subspaces)
    assigner.assign(P, Pint)
    return P


def print_progress(t, T, rank=0):
    if rank != 0:
        return
    progress = int(40 * t / T)
    print(
        f"[{'=' * progress}{' ' * (40 - progress)}] {round(progress * 100 / 40, 1)}%",
        end="\r",
        flush=True,
    )
