from fermi_sim import *
from encoding import *
from encoding import _single_site_op, _vertex_to_ordered_index, _odd_face_to_ordered_index
from scipy import *
import scipy.sparse.linalg
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from stabilizer4_4 import *

def dynamics(H, v, dτ, nsteps, numberoprator):
    
    v /= np.linalg.norm(v)
    
    assert np.linalg.norm(R1.dot(v) - v) < 1e-10
    assert np.linalg.norm(R2.dot(v) - v) < 1e-10
    assert np.linalg.norm(R3.dot(v) - v) < 1e-10
    assert np.linalg.norm(R4.dot(v) - v) < 1e-10
    
    num_traj = np.zeros(nsteps)
    num_traj[0] = np.vdot(v,  numberoprator.dot(v)).real
    for n in range(1, nsteps):
        v = spla.expm_multiply(-(1j)*(dτ)*H, v)
        #v /= np.linalg.norm(v)
        num_traj[n] = np.vdot((v),  numberoprator.dot(v)).real
    
    
    assert np.linalg.norm(R1.dot(v) - v) < 1e-10
    assert np.linalg.norm(R2.dot(v) - v) < 1e-10
    assert np.linalg.norm(R3.dot(v) - v) < 1e-10
    assert np.linalg.norm(R4.dot(v) - v) < 1e-10
    
    plt.xlabel("τ")
    plt.ylabel(r"$\langle \psi(\tau) | n_{corner} | \psi(\tau) \rangle$")
    plt.plot(dτ*np.arange(nsteps), num_traj)
    plt.show()
    
    
    
def dynamics_compare(H, v1, v2, dτ, nsteps, numberoprator1, numberoprator2):
    
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    
    for v in [v1, v2]:
        assert np.linalg.norm(R1.dot(v) - v) < 1e-10
        assert np.linalg.norm(R2.dot(v) - v) < 1e-10
        assert np.linalg.norm(R3.dot(v) - v) < 1e-10
        assert np.linalg.norm(R4.dot(v) - v) < 1e-10
        
    num_traj1 = np.zeros(nsteps)
    num_traj1[0] = np.vdot(v1,  numberoprator1.dot(v1)).real
    for n in range(1, nsteps):
        v1 = spla.expm_multiply(-(1j)*(dτ)*H, v1)
        v1 /= np.linalg.norm(v1)
        num_traj1[n] = np.vdot((v1),  numberoprator1.dot(v1)).real
        
    num_traj2 = np.zeros(nsteps)
    num_traj2[0] = np.vdot(v2,  numberoprator2.dot(v2)).real
    for n in range(1, nsteps):
        v2 = spla.expm_multiply(-(1j)*(dτ)*H, v2)
        v2 /= np.linalg.norm(v2)
        num_traj2[n] = np.vdot((v2),  numberoprator2.dot(v2)).real
        
    for v in [v1, v2]:
        assert np.linalg.norm(R1.dot(v) - v) < 1e-10
        assert np.linalg.norm(R2.dot(v) - v) < 1e-10
        assert np.linalg.norm(R3.dot(v) - v) < 1e-10
        assert np.linalg.norm(R4.dot(v) - v) < 1e-10
        
    plt.plot(dτ*np.arange(nsteps), num_traj1, label="corner", color = 'red')
    plt.plot(dτ*np.arange(nsteps), num_traj2, label="bulk", color = 'blue')
    plt.xlabel("time")
    plt.ylabel("<n>")
    plt.legend()
    plt.show()
    
    
def imaginary_time_evolution_show1(ψ, H, εmin, dτ, nsteps):
    
    
    lattsize = (4,4)
    nqubits = num_qubits_encoding(lattsize)
 
    
    for R in [R1, R2, R3, R4]:
        assert np.linalg.norm(R.dot(ψ) - ψ) < 1e-10
    
    ϵ_traj = np.zeros(nsteps)
    ϵ_traj[0] = np.vdot(ψ, H.dot(ψ)).real
    
    for n in range(1, nsteps):
        if n == nsteps//2:
            print('50% has been finished')
        ψ = spla.expm_multiply(-dτ*H, ψ)
        ψ /= np.linalg.norm(ψ)
        ϵ_traj[n] = np.vdot(ψ, H.dot(ψ)).real
        
    for R in [R1, R2, R3, R4]:
        assert np.linalg.norm(R.dot(ψ) - ψ) < 1e-10
        
    print(ϵ_traj[nsteps-1])
    
    plt.plot(dτ*np.arange(nsteps), ϵ_traj)
    plt.plot([0, nsteps*dτ], [ϵmin, ϵmin], "--")
    plt.xlabel("τ")
    plt.ylabel(r"$\langle \psi(\tau) | H | \psi(\tau) \rangle$")
    plt.title("imaginary time evolution: energy expectation")
    plt.show()
    # difference to ground state energy
    plt.semilogy(dτ*np.arange(nsteps), ϵ_traj - ϵmin)
    plt.ylabel(r"$\langle \psi(\tau) | H | \psi(\tau) \rangle - \epsilon_{\min}$")
    plt.title("imaginary time evolution: difference to exact ground state energy")
    plt.show()
    print("distance from the real ground:", ϵ_traj[nsteps-1] - ϵmin)
    
    return ϵ_traj[nsteps-1], ψ




def adiabatic(H_initial, H_target, ϵ_tar, ψ_ini, dτ, nsteps):
     
    ϵ_traj = np.zeros(nsteps)
    ϵ_traj[0] = np.vdot(ψ_ini, H_target.dot(ψ_ini)).real
    ψ = ψ_ini    
    
    for R in [R1, R2, R3, R4]:
        assert np.linalg.norm(R.dot(ψ) - ψ) < 1e-10
    
    for n in range(1, nsteps):
        if n == nsteps//2:
            print('50% has been finished')
        s = n*(1/(nsteps-1))
        H = s*H_target + (1-s)*H_initial
        
        ψ = spla.expm_multiply(-(1j)*(dτ)*H, ψ)
        ψ /= np.linalg.norm(ψ)
        ϵ_traj[n] = np.vdot(ψ, H_target.dot(ψ)).real
        
    for R in [R1, R2, R3, R4]:
        assert np.linalg.norm(R.dot(ψ) - ψ) < 1e-10
        
    print(ϵ_traj[nsteps-1])
    
    plt.plot(dτ*np.arange(nsteps), ϵ_traj)
    
    plt.plot([0, nsteps*dτ], [ϵ_tar, ϵ_tar], "--")
    
    plt.xlabel("τ")
    plt.ylabel(r"$\langle \psi(\tau) | H_target | \psi(\tau) \rangle$")
    plt.title("adiabatic quantum computing: energy calculated by H_target")
    plt.show()
    
    plt.semilogy(dτ*np.arange(nsteps), ϵ_traj - ϵ_tar)
    plt.ylabel(r"$\langle \psi(\tau) | H_target | \psi(\tau) \rangle - \epsilon_{\min}$")
    plt.title("adiabatic quantum computing: difference to target state energy")
    plt.show()
    print("distance from the real ground:", ϵ_traj[nsteps-1] - ϵ_tar)
    #print("distance from the initial state:", ϵ_traj[nsteps-1] - εmin)
    
    
    return ϵ_traj[nsteps-1], ψ


def quench(H, v, dτ, nsteps, numeroprator):
    
    v /= np.linalg.norm(v)
    
    num_traj = np.zeros(nsteps)
    num_traj[0] = np.vdot(v,  numeroprator.dot(v)).real
    for n in range(1, nsteps):
        v = spla.expm_multiply(-(1j)*(dτ)*H, v)
        #v /= np.linalg.norm(v)
        num_traj[n] = np.vdot((v),  numeroprator.dot(v)).real
        
    
    
    plt.plot(dτ*np.arange(nsteps), num_traj)
    plt.show()
        
    
    
