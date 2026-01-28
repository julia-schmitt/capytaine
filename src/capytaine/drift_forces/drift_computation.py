import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from scipy.integrate import trapezoid 

import capytaine as cpt 
from capytaine.bem.problems_and_results import _default_parameters
from capytaine.post_pro.rao import rao

def compute_kochin_global(dataset, a):
    Hd = dataset['kochin_diffraction']
    Hr = dataset['kochin_radiation'] 
    X = rao(dataset) 
    sum_HdX = sum(Hr[:,i,:]*X[:,0,i] for i in range(6)) 
    return a*np.exp(1j*np.pi/2)*(Hd + sum_HdX)

def far_field(dataset, a):
    omega = dataset['omega']
    k = dataset['wavenumber']
    h = dataset['water_depth']
    theta = dataset['theta']
    beta = dataset['wave_direction']
    rho = dataset['rho']
    g = dataset['g']

    if h == np.inf:
        coef2 = 2*np.pi*rho*k**2
    else:
        k0 = omega**2/g 
        coef2 = 2*np.pi*rho*k*(k0*h)**2 / (h*((k*h)**2 - (k0*h)**2 + k0*h))
 
    H = compute_kochin_global(dataset, a)[:,0,:]
    H_beta = H[:,np.where(theta_range == wave_direction)[0][0]]
    coef1 = 2*np.pi*rho*a*omega
    
    F = np.array([
        -coef1*np.cos(beta)*np.imag(H_beta) - coef2*trapezoid(np.abs(H)**2*np.cos(theta), theta),
        -coef1*np.sin(beta)*np.imag(H_beta) - coef2*trapezoid(np.abs(H)**2*np.sin(theta), theta)
        ])

    return F

if __name__ == "__main__":
    g = _default_parameters["g"]
    rho = 1025 
    a = 1
    wave_direction = 0
    water_depth = 200
    radius = 1
    theta_range = np.linspace(0, 2*np.pi, 19)
    theta_range = np.sort(np.append(theta_range, wave_direction)) if not np.any(np.isclose(theta_range, wave_direction)) else theta_range

    mesh = cpt.mesh_sphere(radius=radius, resolution=(50, 50), center=(0,0,0)).immersed_part()
    lid_mesh = mesh.generate_lid()
    body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0,)),
            name=mesh.name,
            center_of_mass=(0,0,0), #ok (essayer autre)
            # lid_mesh=lid_mesh
            ).immersed_part()
    body.inertia_matrix = body.compute_rigid_body_inertia()
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()

    F_analytic=[0, 0, 0.07, 0.26, 0.49, 0.70, 0.83, 0.88, 0.84, 0.72, 0.66, 0.655, 0.652]
    ka_analytic = np.array([0.3, 0.5, 0.83, 0.92, 1.0, 1.05, 1.09, 1.16, 1.24, 1.39, 1.59, 1.88, 2.28])
    F_Delhommeau= [0.035, 0.8373, 0.6700, 0.6265]
    ka_Delhommeau= [0.77, 1.29, 1.53, 2.015]

    solver = cpt.BEMSolver()
    test_matrix = xr.Dataset(coords={
        'wavenumber': ka_analytic, 'water_depth': water_depth, 'wave_direction': wave_direction, 'theta': theta_range, 'radiating_dof': list(body.dofs.keys()), 'rho': rho
    })
    solver = cpt.BEMSolver()
    dataset = solver.fill_dataset(test_matrix, body, hydrostatics=True)

    plt.plot(ka_analytic, F_analytic, '-o', label = 'analytical')
    plt.plot(ka_Delhommeau, F_Delhommeau, '*', label = 'Delhommeau')
    plt.grid()
    plt.plot(ka_analytic, far_field(dataset, a)[0]/(g*rho*a**2*radius), '-x', label='far field formulation')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift/(rho*g*a²*r)')
    plt.show()