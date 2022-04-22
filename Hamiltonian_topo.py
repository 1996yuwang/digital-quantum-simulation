"""
def Hamiltonian from the paper:
<A quantized microwave quadrupole insulator with topologically protected corner states>
https://www.nature.com/articles/nature25777
The Hamiltonian consists of two parts:
1. coupling within unit cells: γ(ci+cj + cj+ci) for each pair of adjacent sites in the same unit cell, it becomes -γ(ci+cj + cj+ci) when it is a negative coupling (shown as red dashed lines in Figure 1.b in the paper).
2. coupling between unit cells: λ(ci+cj + cj+ci) for each pair of adjacent sites in the different unit cell,
it becomes -λ(ci+cj + cj+ci) when it is a negative coupling (shown as black dashed lines in Figure 1.b in the paper).
Contents in this file:
H_Quad_topo: Hamiltonian of the model with open boundary condition.
generate_Hamiltonian_with_occupation_number: A efficient function to extract a Hamiltonian with a fixed occupation number from a given Hamiltonian H.
calculate_part_hamiltonian: Use Lanczos Algorithm to calculate the 20 Smallest eigen-energy and corresponding eignestates.
H_Quad_topo_perio: Hamiltonian of the model with periodic boundary condition.
H_Quad_encoded: Encode the Hamiltonian before with functions in encoding.py.
H_Quad_encoded_perio: Encoded Hamiltonian of the model with periodic boundary condition.
"""


import numpy as np
from scipy import sparse
from gmpy2 import popcount
from encoding import construct_edge_operator, construct_vertex_operator
from fermi_sim import create_op, annihil_op



def H_Quad_topo(size_cell, gamma, lamda):
    """
    size_cell: size_cell * size_cell stands for number of cells. The total resonator is then 4*size_cell*size_cell
    gamma: coupling within  unit cells
    lamda: coupling between unit cells
    """

    L0 = 2* size_cell[0]
    L1 = 2* size_cell[1]
    nmodes = L0*L1
    H = sparse.csr_matrix((2**nmodes, 2**nmodes), dtype=float)

    """
    we arrange the resonators with 3 indices in physics:
    the indices of unit cells (i,j),
    and the indices k of resonators inside a cell: 1, 2, 3 ,4 as shown below:
    1, 2
    3, 4
    To sum up, one resonator has three indices: (i, j, k), which also correspinds to a indices in quantum computer.
    Example for 2*2 system:
    1, 2,     3, 4                        1, 2,   1, 2
    5, 6,     7, 8                        3, 4,   3, 4
                      <<<<<<<<>>>>>>>>
    9, 10,  11, 12                        1, 2,   1, 2
    13,14,  15, 16                        3, 4,   3, 4
    For example, (0,0,4) represnts 4th resonartor in the 1st unit cell, as (0,0) is cord of unit cell and 4 is number of resonator
    (0,0,4) >>> No.6
    """

    """
    A simpler method:
    (even, even): 1st resonator
    (even, odd) : 2nd resonator      1, 2
    (odd,  even): 3rd resonator      3, 4
    (odd,  odd) : 4th resonator
    will use it later.
    """


    cord = np.zeros([size_cell[0], size_cell[1], 5])
    for i in range(size_cell[0]):
        for j in range(size_cell[1]):
            cord[i,j,1] = 2*i*2*size_cell[1] + 2*j + 1
            cord[i,j,2] = 2*i*2*size_cell[1] + 2*j + 2
            cord[i,j,3] = (2*i+1)*2*size_cell[1] + 2*j + 1
            cord[i,j,4] = (2*i+1)*2*size_cell[1] + 2*j + 2


    #coupling terms inside unit cells
    for i in range(size_cell[0]):
        for j in range(size_cell[1]):
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,2])-1))))
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,1])-1))))

            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,4])-1))))
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,2])-1))))

            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,4])-1))))
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,3])-1))))

            H += -gamma* (create_op(nmodes, 1 << (int(cord[i,j,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,3])-1))))
            H += -gamma* (create_op(nmodes, 1 << (int(cord[i,j,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,1])-1))))



    #coupling between unit cells

    #horizontal direction
    for j in range(size_cell[1]-1):
        for i in range(size_cell[0]):
            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j,  2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j+1,1])-1))))
            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j+1,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,  2])-1))))

            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j,  4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j+1,3])-1))))
            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j+1,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,  4])-1))))


    #vertical direction
    for i in range(size_cell[0]-1):
        for j in range(size_cell[1]):
            H +=  lamda* (create_op(nmodes, 1 << (int(cord[i,  j,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i+1,j,1])-1))))
            H +=  lamda* (create_op(nmodes, 1 << (int(cord[i+1,j,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,  j,3])-1))))

            H += -lamda* (create_op(nmodes, 1 << (int(cord[i,  j,4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i+1,j,2])-1))))
            H += -lamda* (create_op(nmodes, 1 << (int(cord[i+1,j,2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,  j,4])-1))))


    return H






def generate_Hamiltonian_with_occupation_number(H, n):
    """
    A fast function to extract a Hamiltonian with a fixed occupation number from a given Hamiltonian H.
    """
    size = np.shape(H)
    H_occu = sparse.csr_matrix((size[0], size[0]), dtype=float)
    non_zero_list = H.nonzero()
    shape = np.shape(non_zero_list)
    i_group = []
    j_group = []


    for i in range(shape[1]):
        if popcount(int(non_zero_list[0][i])) == n:
            i_group.append([i])


    for i in i_group:
        if popcount(int(non_zero_list[1][i])) == n:
            j_group.append([i])


    non_zero_occu_list_0 = []
    non_zero_occu_list_1 = []

    for i in j_group:
        non_zero_occu_list_0.append(non_zero_list[0][i])
        non_zero_occu_list_1.append(non_zero_list[1][i])

    H_occu[non_zero_occu_list_0, non_zero_occu_list_1] = H[non_zero_occu_list_0, non_zero_occu_list_1]

    return H_occu



def calculate_part_hamiltonian(H, n):
    """
    Use Lanczos Algorithm to calculate the 20 Smallest eigen-energy and corresponding eignestates
    Input:
    H : A fULL Hamiltonian
    n : The fixed occupation number you want to calculate.
    """
    H_occu = generate_Hamiltonian_with_occupation_number(H, n)
    E, v = sparse.linalg.eigsh(H_occu , k = 20, which = 'SA')
    return E,v




def H_Quad_topo_perio(size_cell, gamma, lamda):

    L0 = 2*size_cell[0]
    L1 = 2*size_cell[1]
    nmodes = L0*L1
    H = sparse.csr_matrix((2**nmodes, 2**nmodes), dtype=float)


    cord = np.zeros([size_cell[0]+1, size_cell[1]+1, 5])
    for i in range(size_cell[0]):
        for j in range(size_cell[1]):
            cord[i,j,1] = 2*i*2*size_cell[1] + 2*j + 1
            cord[i,j,2] = 2*i*2*size_cell[1] + 2*j + 2
            cord[i,j,3] = (2*i+1)*2*size_cell[1] + 2*j + 1
            cord[i,j,4] = (2*i+1)*2*size_cell[1] + 2*j + 2



    for j in range(size_cell[1]):
        cord[size_cell[0],j,1] = cord[0,j,1]
        cord[size_cell[0],j,2] = cord[0,j,2]
        cord[size_cell[0],j,3] = cord[0,j,3]
        cord[size_cell[0],j,4] = cord[0,j,4]


    for i in range(size_cell[0]):
        cord[i,size_cell[1],1] =  cord[i,0,1]
        cord[i,size_cell[1],2] =  cord[i,0,2]
        cord[i,size_cell[1],3] =  cord[i,0,3]
        cord[i,size_cell[1],4] =  cord[i,0,4]


    #coupling terms inside unit cells
    for i in range(size_cell[0]):
        for j in range(size_cell[1]):
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,2])-1))))
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,1])-1))))

            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,4])-1))))
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,2])-1))))

            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,4])-1))))
            H +=  gamma* (create_op(nmodes, 1 << (int(cord[i,j,4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,3])-1))))

            H += -gamma* (create_op(nmodes, 1 << (int(cord[i,j,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,3])-1))))
            H += -gamma* (create_op(nmodes, 1 << (int(cord[i,j,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,1])-1))))



    #coupling between unit cells

    #horizontal direction
    for j in range(size_cell[1]):
        for i in range(size_cell[0]):

            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j,  2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j+1,1])-1))))
            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j+1,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,  2])-1))))

            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j,  4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j+1,3])-1))))
            H += lamda* (create_op(nmodes, 1 << (int(cord[i,j+1,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,j,  4])-1))))


    #vertical direction
    for i in range(size_cell[0]):
        for j in range(size_cell[1]):
            H +=  lamda* (create_op(nmodes, 1 << (int(cord[i,  j,3])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i+1,j,1])-1))))
            H +=  lamda* (create_op(nmodes, 1 << (int(cord[i+1,j,1])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,  j,3])-1))))

            H += -lamda* (create_op(nmodes, 1 << (int(cord[i,  j,4])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i+1,j,2])-1))))
            H += -lamda* (create_op(nmodes, 1 << (int(cord[i+1,j,2])-1)).dot(annihil_op(nmodes, 1 << (int(cord[i,  j,4])-1))))


    return H





def H_Quad_encoded(size_cell, gamma, lamda):

    """
    Encode the Hamiltonian before with functions in encoding.py
    ai(dager)aj + aj(dager)ai = -(i/2)(Eij*Vj + Vi*Eij)
    (even, even): 1st resonator
    (even, odd) : 2nd resonator     (0,0) (0,1)      1, 2
    (odd,  even): 3rd resonator     (1,0) (1,1)      3, 4
    (odd,  odd) : 4th resonator
    """

    lattsize = (2*size_cell[0], 2*size_cell[1])
    L0 = 2*size_cell[0]
    L1 = 2*size_cell[1]
    nqubits = num_qubits_encoding(lattsize)
    H = sparse.csr_matrix((2**nqubits, 2**nqubits), dtype=complex)

    for x in range(L0):
        for y in range(L1):

            #1st resonator
            if x%2 == 0 and y%2 == 0:
                    #horizontal (to right)
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical   (down )
                    H += -gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H += -gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))

            #2nd resonator
            if x%2 == 0 and y%2 == 1:
                if y != L1-1:
                    #if (x,y) not on the boundary
                    #horizontal
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))
                else:
                    #if (x,y) on the boundary, then only connection down
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))


            #3rd resonator
            if x%2 == 1 and y%2 == 0:
                if x != L0-1:
                    #if (x,y) not on the boundary
                    #horizontal
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))
                else:
                    #if (x,y) on the boundary, then only connection to right
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))


            #4th resonator
            if x%2 == 1 and y%2 == 1:
                if x != L0-1 and y != L1-1:
                    #if (x,y) not on the boundary
                    #horizontal
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical   (down )
                    H += -lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H += -lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))

                if x != L0-1 and y == L1-1:
                    #if (x,y) on the right boundary, then only connection down
                    H += -lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H += -lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))

                if x == L0-1 and y != L1-1:
                    #if (x,y) on the bottom boundary, then only connection to right
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))

    return H




def H_Quad_encoded_perio(size_cell, gamma, lamda):

    lattsize = (2*size_cell[0], 2*size_cell[1])
    L0 = 2*size_cell[0]
    L1 = 2*size_cell[1]
    nqubits = num_qubits_encoding(lattsize)
    H = sparse.csr_matrix((2**nqubits, 2**nqubits), dtype=complex)

    for x in range(L0):
        for y in range(L1):

            #1st resonator
            if x%2 == 0 and y%2 == 0:
                    #horizontal (to right)
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical   (down )
                    H += -gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H += -gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))

            #2nd resonator
            if x%2 == 0 and y%2 == 1:
                if y != L1-1:
                    #if (x,y) not on the boundary
                    #horizontal
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))
                if y == L1-1:
                    #if (x,y) on the boundary,
                    #vertical
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))
                    #horizontal
                    H +=  lamda*(-1j/2)*end_to_head_horz(lattsize,(x,y),(x,0)).dot(construct_vertex_operator(lattsize,(x,0)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(end_to_head_horz(lattsize,(x,y),(x,0)))




            #3rd resonator
            if x%2 == 1 and y%2 == 0:
                if x != L0-1:
                    #if (x,y) not on the boundary
                    #horizontal
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))
                if x == L0-1:
                    #if (x,y) on the boundary,
                    #horizontal
                    H +=  gamma*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  gamma*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical
                    H +=  lamda*(-1j/2)*end_to_head_vert(lattsize,(x,y),(0,y)).dot(construct_vertex_operator(lattsize,(0,y)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(end_to_head_vert(lattsize,(x,y),(0,y)))


            #4th resonator
            if x%2 == 1 and y%2 == 1:
                if x != L0-1 and y != L1-1:
                    #if (x,y) not on the boundary
                    #horizontal
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                    #vertical   (down )
                    H += -lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H += -lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))

                if x != L0-1 and y == L1-1:
                    #if (x,y) on the right boundary,
                    #vertical
                    H += -lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x+1,y)).dot(construct_vertex_operator(lattsize,(x+1,y)))
                    H += -lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x+1,y)))
                    #horizontal
                    H +=  lamda*(-1j/2)*end_to_head_horz(lattsize,(x,y),(x,0)).dot(construct_vertex_operator(lattsize,(x,0)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(end_to_head_horz(lattsize,(x,y),(x,0)))


                if x == L0-1 and y != L1-1:
                    #if (x,y) on the bottom boundary,
                    #horizontal
                    H +=  lamda*(-1j/2)*construct_edge_operator(lattsize,(x,y),(x,y+1)).dot(construct_vertex_operator(lattsize,(x,y+1)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(construct_edge_operator(lattsize,(x,y),(x,y+1)))
                     #vertical
                    H += -lamda*(-1j/2)*end_to_head_vert(lattsize,(x,y),(0,y)).dot(construct_vertex_operator(lattsize,(0,y)))
                    H += -lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(end_to_head_vert(lattsize,(x,y),(0,y)))

                if x == L0-1 and y == L1-1:
                    #corner
                    #horizontal
                    H +=  lamda*(-1j/2)*end_to_head_horz(lattsize,(x,y),(x,0)).dot(construct_vertex_operator(lattsize,(x,0)))
                    H +=  lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(end_to_head_horz(lattsize,(x,y),(x,0)))
                     #vertical
                    H += -lamda*(-1j/2)*end_to_head_vert(lattsize,(x,y),(0,y)).dot(construct_vertex_operator(lattsize,(0,y)))
                    H += -lamda*(-1j/2)*construct_vertex_operator(lattsize,(x,y)).dot(end_to_head_vert(lattsize,(x,y),(0,y)))

    return H



def end_to_head_horz(lattsize, i, j):
    """
    Eij = -i * γi * γj where i is the last site in a row and j is the first site in a row.
    γi, γj are all first Majorana op of the site.
    We will make use of this function to define a Hamiltonian with a periodic boundary condition.
    """
    #from i(i[0],i[1])(end) to j(j[0],0)(head)
    assert i[0] == j[0]
    E = construct_edge_operator(lattsize, (i[0],i[1]),((i[0],i[1]-1)))
    if i[1] >1:
        for count in range (abs(i[1]-1)):
            E = E.dot(construct_edge_operator(lattsize, (i[0],i[1]-1-count),((i[0],i[1]-2-count))))

    E = (1j)**(i[1]-1)*E

    return E


def end_to_head_vert(lattsize, i, j):
    """
    Eij = -i * γi * γj where i is the last site in a row and j is the first site in a column.
    γi, γj are all first Majorana op of the site.
    We will make use of this function to define a Hamiltonian with a periodic boundary condition.
    """
    #from i(i[0],i[1])(end) to j(0,j[1])(head)
    assert i[1] == j[1]
    E = construct_edge_operator(lattsize, (i[0],i[1]),((i[0]-1,i[1])))
    if i[0] >1:
        for count in range (abs(i[0]-1)):
            E = E.dot(construct_edge_operator(lattsize, (i[0]-1-count,i[1]),((i[0]-2-count,i[1]))))

    E = (1j)**(i[0]-1)*E

    return E
