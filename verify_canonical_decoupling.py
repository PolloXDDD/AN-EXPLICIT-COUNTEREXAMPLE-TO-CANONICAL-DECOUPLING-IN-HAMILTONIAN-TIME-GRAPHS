#!/usr/bin/env python3
"""Exact, dependency-free verification of the seven-vertex counterexample.

Run with Python 3.10 or newer. All linear algebra is over F_2, represented
by integer XOR. This is a verifier, not the source of the mathematical proof.
"""

from itertools import permutations


N = 7
UNDIRECTED = {
    (1, 2), (1, 3), (1, 4), (1, 5), (2, 4),
    (2, 5), (3, 6), (3, 7), (4, 7), (5, 7),
}
E1 = (7, 2, 2)
E2 = (4, 3, 5)
H = (7, 5, 2, 4, 1, 3, 6)
A = (5, 7, 2, 4, 1, 3, 6)
B = (5, 7, 2, 1, 4, 3, 6)
C = (7, 5, 1, 2, 4, 3, 6)
WITNESS = (
    (6, 5, 1, 2, 4, 3, 7),
    (6, 5, 2, 1, 4, 3, 7),
    (6, 5, 1, 2, 4, 7, 3),
    (6, 5, 2, 1, 4, 7, 3),
    (7, 5, 1, 2, 4, 3, 6),
    (7, 5, 2, 1, 4, 3, 6),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def path_edges(pi):
    return {(pi[t - 1], pi[t], t) for t in range(1, N)}


def main():
    edges = [(a, b, t) for t in range(1, N)
             for a in range(1, N + 1) for b in range(1, N + 1)]
    index = {edge: j for j, edge in enumerate(edges)}
    allowed = {e for e in edges if tuple(sorted(e[:2])) in UNDIRECTED}
    forbidden = [E1, E2] + sorted(set(edges) - allowed - {E1, E2})
    level_of_edge = {e: j for j, e in enumerate(forbidden, 1)}
    require(len(forbidden) == len(set(forbidden)) == 174,
            "The complement must be enumerated exactly once.")

    def vector(pi):
        return sum(1 << index[e] for e in path_edges(pi))

    def entry(v, edge):
        return (v >> index[edge]) & 1

    def ell(v):
        return entry(v, (2, 4, 4)) ^ entry(v, (7, 5, 6))

    # Build the tensor directly from its six generators, without using the
    # factorization in the paper. Each row is a bit vector of all 294 columns.
    tensor = [0] * len(edges)
    for pi in WITNESS:
        require(sorted(pi) == list(range(1, N + 1)), "Invalid permutation.")
        v = vector(pi)
        for edge in path_edges(pi):
            tensor[index[edge]] ^= v
    require(all(tensor[index[e]] == 0 for e in forbidden),
            "A forbidden tensor row is nonzero.")
    diagonal = sum(((row >> j) & 1) << j for j, row in enumerate(tensor))
    require(diagonal == vector(H) ^ vector(A) ^ vector(B) ^ vector(C),
            "The four-term diagonal identity failed.")

    all_perms = list(permutations(range(1, N + 1)))
    vectors = {pi: vector(pi) for pi in all_perms}
    levels = {pi: max(level_of_edge.get(e, 0) for e in path_edges(pi))
              for pi in all_perms}
    require([levels[pi] for pi in (H, A, B, C)] == [0, 1, 2, 2],
            "The four distinguished permutations have wrong levels.")
    for pi in all_perms:
        if levels[pi] <= 1:
            require(ell(vectors[pi]) == 0, "The separator identity failed.")
    require((ell(vector(B)), ell(vector(C))) == (0, 1),
            "The separator must distinguish the two level-two generators.")

    # Construct a full canonical basis. At each level give priority to the
    # generators named in the proof, then complete in lexicographic order.
    preferred = {0: (H,), 1: (A,), 2: (B, C)}
    pivots = {}
    basis = []
    coefficients = {}
    prefix_ranks = []
    for level in range(len(forbidden) + 1):
        priority = preferred.get(level, ())
        candidates = list(priority) + [pi for pi in all_perms
                                       if levels[pi] == level and pi not in priority]
        for pi in candidates:
            require(levels[pi] == level, "Incorrect basis level.")
            value, used = vectors[pi], 0
            while value:
                pivot = value.bit_length() - 1
                if pivot not in pivots:
                    unit = 1 << len(basis)
                    pivots[pivot] = (value, used ^ unit)
                    basis.append((level, pi))
                    coefficients[pi] = unit
                    break
                row, representation = pivots[pivot]
                value ^= row
                used ^= representation
            else:
                coefficients[pi] = used
        prefix_ranks.append(len(basis))

    # Independently verify every recorded representation and all prefixes.
    for pi in all_perms:
        reconstructed = 0
        for j, (level, bp) in enumerate(basis):
            if (coefficients[pi] >> j) & 1:
                require(level <= levels[pi], "A prefix fails to span.")
                reconstructed ^= vectors[bp]
        require(reconstructed == vectors[pi], "Incorrect basis coordinates.")
    require(len(basis) == 211, "Unexpected full-space dimension.")
    require(prefix_ranks[:3] == [28, 30, 34], "Unexpected prefix dimensions.")

    value, expansion = diagonal, 0
    while value:
        row, representation = pivots[value.bit_length() - 1]
        value ^= row
        expansion ^= representation
    components = [0] * (len(forbidden) + 1)
    selected = []
    for j, (level, pi) in enumerate(basis):
        if (expansion >> j) & 1:
            components[level] ^= vectors[pi]
            selected.append((level, pi))
    require(selected == [(0, H), (1, A), (2, B), (2, C)],
            "The canonical expansion differs from the proof.")

    # Construct g_c explicitly; this checks the tensor decomposition as well.
    closed = tensor[:]
    for _, pi in selected:
        for edge in path_edges(pi):
            closed[index[edge]] ^= vector(pi)
    require(all(((row >> j) & 1) == 0 for j, row in enumerate(closed)),
            "P(g_c) must be zero.")
    for m, edge in enumerate(forbidden, 1):
        require(sum(entry(components[i], edge)
                    for i in range(m, len(components))) % 2 == 0,
                "Theorem 8 must remain valid.")
    higher = sum(entry(components[i], E1)
                 for i in range(2, len(components))) % 2
    require(higher == entry(components[1], E1) == 1,
            "Expected direct counterexample to canonical decoupling.")
    require(all(v == 0 for v in components[3:]), "Higher levels must vanish.")
    require(entry(components[2], E1) == 1,
            "The isolated component must obstruct supported lifting.")

    print("PASS: six valid permutations generate g in H_P^7.")
    print("PASS: all 174 forbidden rows vanish, each with 294 entries.")
    print("PASS: the separator identity holds for every permutation in G_1.")
    print("PASS: full canonical basis, rank 211; prefix ranks 28, 30, 34.")
    print("PASS: P(g) = T(h) + T(a) + T(b) + T(c), in levels 0, 1, 2, 2.")
    print("PASS: P(g_c) = 0 and Theorem 8 holds at every level.")
    print("COUNTEREXAMPLE: sum_{i>1} f^(i)(e_1) = 1, with e_1 = (7,2,2).")
    print("LIFTING OBSTRUCTION: f^(2)(e_1) = 1, although every level above 2 is zero.")


if __name__ == "__main__":
    main()
