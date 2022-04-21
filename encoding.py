#The first half in this file was written by Prof. Dr. Christian Mendl

"""
For the qubit simulator, we enumerate qubits as:
    primary (vertex) qubits in lexicographical order,
    followed by auxiliary (odd face) qubits.
Coordinate system:
    0------y
    |
    |
    x
Reference:
    Charles Derby, Joel Klassen, Johannes Bausch, Toby Cubitt
    Compact fermion to qubit mappings
    Phys. Rev. B 104, 035118 (2021)
"""

from scipy import sparse
from enum import Enum


def num_qubits_encoding(lattsize):
    """
    Total number of qubits (primary and auxiliary).
    """
    return lattsize[0]*lattsize[1] + ((lattsize[0] - 1)*(lattsize[1] - 1) + 1) // 2


def edge_to_odd_face(i, j):
    """
    Find the adjacent "odd" face of edge (i, j).
    A face is indexed via its vertex with smallest x- and y-coordinates.
    Note: one of the returned coordinates can be negative, in case the edge is
    on the boundary and not adjacent to an odd face inside the rectangular region.
    """
    # unpack vertex coordinates
    ix, iy = i
    jx, jy = j
    # ensure that i, j are nearest neighbors
    assert (ix == jx and abs(iy - jy) == 1) or (iy == jy and abs(ix - jx) == 1), "not a nearest neighbor edge"
    xmin = min(ix, jx)
    ymin = min(iy, jy)
    if (xmin + ymin) % 2 == 0:
        return (xmin, ymin)
    else:
        if ix == jx:
            return (xmin - 1, ymin)
        else:
            return (xmin, ymin - 1)


def _vertex_to_ordered_index(lattsize, i):
    """
    Determine the ordered qubit index of primary qubit `i`.
    """
    # unpack vertex coordinates
    ix, iy = i
    return ix*lattsize[1] + iy


def _odd_face_to_ordered_index(lattsize, f):
    """
    Determine the ordered qubit index of the auxiliary qubit at odd face `f`.
    A face is indexed via its vertex with smallest x- and y-coordinates.
    """
    fx, fy = f
    assert (fx + fy) % 2 == 0, "{} is not a valid odd face index".format(f)
    if fx < 0 or fy < 0 or fx == lattsize[0] - 1 or fy == lattsize[1] - 1:
        # face outside the rectangular region
        return -1
    else:
        # take offset by primary vertex qubits into account
        return lattsize[0]*lattsize[1] + (fx*(lattsize[1] - 1) + 1)//2 + (fy//2)


def _single_site_op(op, i, nsites):
    """
    Construct the sparse matrix representation of `op` acting on site `i`.
    """
    assert 0 <= i and i < nsites
    op = sparse.csr_matrix(op)
    return sparse.kron(sparse.eye(2**i), sparse.kron(op, sparse.eye(2**(nsites-i-1))))


def construct_vertex_operator(lattsize, j):
    """
    Construct the vertex operator V_j, given the index j = (jx, jy).
    """
    # Pauli-Z matrix
    Z = sparse.csr_matrix([[ 1,  0], [ 0, -1]], dtype=complex)
    # total number of qubits
    nqubits = num_qubits_encoding(lattsize)
    j_idx = _vertex_to_ordered_index(lattsize, j)
    return _single_site_op(Z, j_idx, nqubits)


def construct_edge_operator(lattsize, i, j):
    """
    Construct the edge operator E_{ij},
    given vertex indices i = (ix, iy) and j = (jx, jy).
    """
    # Pauli matrices
    X = sparse.csr_matrix([[ 0,   1], [ 1,   0]], dtype=complex)
    Y = sparse.csr_matrix([[ 0, -1j], [ 1j,  0]], dtype=complex)
    # total number of qubits
    nqubits = num_qubits_encoding(lattsize)
    # unpack vertex coordinates
    ix, iy = i
    jx, jy = j
    # ensure that i, j are nearest neighbors
    assert (ix == jx and abs(iy - jy) == 1) or (iy == jy and abs(ix - jx) == 1), "not a nearest neighbor edge"
    i_idx = _vertex_to_ordered_index(lattsize, i)
    j_idx = _vertex_to_ordered_index(lattsize, j)
    f_idx = _odd_face_to_ordered_index(lattsize, edge_to_odd_face(i, j))
    if ix == jx:    # horizontal edge
        if (ix % 2 == 0 and jy < iy) or (ix % 2 == 1 and iy < jy):
            # conforming horizontal orientation
            E =  _single_site_op(X, i_idx, nqubits).dot(_single_site_op(Y, j_idx, nqubits))
        else:
            E = -_single_site_op(X, j_idx, nqubits).dot(_single_site_op(Y, i_idx, nqubits))
        if f_idx != -1:
            E = E.dot(_single_site_op(Y, f_idx, nqubits))
    else:       # vertical edge
        assert iy == jy
        if iy % 2 == 0: # iy even
            if jx < ix:     # conforming up orientation
                E = -_single_site_op(X, i_idx, nqubits).dot(_single_site_op(Y, j_idx, nqubits))
            else:           # non-conforming down orientation
                E =  _single_site_op(X, j_idx, nqubits).dot(_single_site_op(Y, i_idx, nqubits))
        else:   # iy odd
            if ix < jx:     # conforming down orientation
                E =  _single_site_op(X, i_idx, nqubits).dot(_single_site_op(Y, j_idx, nqubits))
            else:           # non-conforming up orientation
                E = -_single_site_op(X, j_idx, nqubits).dot(_single_site_op(Y, i_idx, nqubits))
        if f_idx != -1:
            E = E.dot(_single_site_op(X, f_idx, nqubits))
    return E


def construct_path_operator(lattsize, i, j):
    """
    E_{ij} = -i \gamma_i \gamma_j between arbitary two sites.
    """
    nqubits = num_qubits_encoding(lattsize)
    # include factors of the imaginary unit
    E = 1j**(abs(i[0] - j[0]) + abs(i[1] - j[1]) - 1) * sparse.identity(2**nqubits, dtype=complex)
    while i != j:
        # select neighbor closest to target vertex `j`
        neighs = [(i[0] - 1, i[1]), (i[0] + 1, i[1]), (i[0], i[1] - 1), (i[0], i[1] + 1)]
        i_next = min(neighs, key=lambda k: (k[0] - j[0])**2 + (k[1] - j[1])**2)
        E = E.dot(construct_edge_operator(lattsize, i, i_next))
        i = i_next
    return E


class MajoranaType(Enum):
    """
    Majorana operator type: real or imaginary.
    """
    Re = 0
    Im = 1


def construct_majorana_operator(lattsize, i, mt: MajoranaType):
    """
    Construct the encoded single Majorana operator \gamma_i or \bar{\gamma_i}.
    For the conventions used here, only subspace dimension cases (I) or (III)
    in Phys. Rev. B 104, 035118 (2021) can occur.
    """
    X = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    nqubits = num_qubits_encoding(lattsize)
    # encoded Majorana operator on site (0, 0) is the X-gate acting on this site
    gamma = _single_site_op(X, _vertex_to_ordered_index(lattsize, (0, 0)), nqubits).dot(
                             1j*construct_path_operator(lattsize, (0, 0), i))
    if mt == MajoranaType.Im:
        gamma = gamma.dot(1j*construct_vertex_operator(lattsize, i))
    return gamma


def construct_logical_X(lattsize):
    """
    Construct Pauli-X gate acting on the logical qubit in case of one more odd face than even face.
    """
    assert (lattsize[0] - 1)*(lattsize[1] - 1) % 2 == 1, "require one more odd face than even face"
    nqubits = num_qubits_encoding(lattsize)
    # Pauli matrices
    X = sparse.csr_matrix([[ 0,   1], [ 1,   0]], dtype=complex)
    Z = sparse.csr_matrix([[ 1,   0], [ 0,  -1]], dtype=complex)
    op = sparse.identity(2**nqubits)
    for x in range(lattsize[0]):
        op = op.dot(_single_site_op(Z, _vertex_to_ordered_index(lattsize, (x, lattsize[1] - 1)), nqubits))
    for x in range(0, lattsize[0] - 1, 2):
        op = op.dot(_single_site_op(X, _odd_face_to_ordered_index(lattsize, (x, lattsize[1] - 2)), nqubits))
    return op


def construct_logical_Y(lattsize):
    """
    Construct Pauli-Y gate acting on the logical qubit in case of one more odd face than even face.
    """
    assert (lattsize[0] - 1)*(lattsize[1] - 1) % 2 == 1, "require one more odd face than even face"
    nqubits = num_qubits_encoding(lattsize)
    Y = sparse.csr_matrix([[ 0, -1j], [ 1j,  0]], dtype=complex)
    Z = sparse.csr_matrix([[ 1,   0], [ 0,  -1]], dtype=complex)
    op = sparse.identity(2**nqubits)
    for y in range(lattsize[1]):
        op = op.dot(_single_site_op(Z,_vertex_to_ordered_index(lattsize, (lattsize[0] - 1, y)), nqubits))
    for y in range(0, lattsize[1] - 1, 2):
        op = op.dot(_single_site_op(Y, _odd_face_to_ordered_index(lattsize, (lattsize[0] - 2, y)), nqubits))
    return op


def construct_logical_Z(lattsize):
    """
    Construct Pauli-Z gate acting on the logical qubit in case of one more odd face than even face.
    """
    return -1j*construct_logical_X(lattsize).dot(construct_logical_Y(lattsize))
