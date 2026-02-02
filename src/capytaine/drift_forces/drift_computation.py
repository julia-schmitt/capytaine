import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from scipy.integrate import trapezoid 

import capytaine as cpt 
from capytaine.bem.problems_and_results import _default_parameters
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.post_pro.rao import rao
from capytaine.tools.lists_of_points import _normalize_points, _normalize_free_surface_points


### Far field formulation ###
def compute_kochin_global(dataset, a, fixed=False):
    Hd = dataset['kochin_diffraction']
    Hr = dataset['kochin_radiation'] 
    X = rao(dataset) 
    if fixed:
        X = 0*X
    sum_HdX = sum(Hr[:,i,:]*X[:,0,i] for i in range(6)) 
    return a*np.exp(1j*np.pi/2)*(Hd + sum_HdX)

def far_field(dataset, a, fixed=False):
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
 
    H = compute_kochin_global(dataset, a, fixed)[:,0,:]
    H_beta = H[:,np.where(theta == beta)[0][0]]
    coef1 = 2*np.pi*rho*a*omega
    
    F = np.array([
        -coef1*np.cos(beta)*np.imag(H_beta) - coef2*trapezoid(np.abs(H)**2*np.cos(theta), theta),
        -coef1*np.sin(beta)*np.imag(H_beta) - coef2*trapezoid(np.abs(H)**2*np.sin(theta), theta)
        ])

    return F

def plot_far_field():
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    radius = 1
    theta_range = np.linspace(0, 2*np.pi, 19)
    theta_range = np.sort(np.append(theta_range, wave_direction)) if not np.any(np.isclose(theta_range, wave_direction)) else theta_range

    mesh = cpt.mesh_sphere(radius=radius, resolution=(50, 50), center=(0,0,0)).immersed_part()
    body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(0,0,0)),
            name=mesh.name,
            center_of_mass=(0,0,0),
            ).immersed_part()
    body.inertia_matrix = body.compute_rigid_body_inertia()
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()

    F_analytic=[0, 0, 0.07, 0.26, 0.49, 0.70, 0.83, 0.88, 0.84, 0.72, 0.66, 0.655, 0.652]
    ka_analytic = np.array([0.3, 0.5, 0.83, 0.92, 1.0, 1.05, 1.09, 1.16, 1.24, 1.39, 1.59, 1.88, 2.28])
    F_Delhommeau= [0.035, 0.8373, 0.6700, 0.6265]
    ka_Delhommeau= [0.77, 1.29, 1.53, 2.015]

    solver = cpt.BEMSolver()
    test_matrix = xr.Dataset(coords={
        'wavenumber': ka_analytic, 'wave_direction': wave_direction, 'theta': theta_range, 'radiating_dof': list(body.dofs.keys()), 
    })
    dataset = solver.fill_dataset(test_matrix, body, hydrostatics=True)

    plt.plot(ka_analytic, F_analytic, '-o', label = 'analytical')
    plt.plot(ka_Delhommeau, F_Delhommeau, '*', label = 'Delhommeau')
    plt.grid()
    plt.plot(ka_analytic, far_field(dataset, a)[0]/(g*rho*a**2*radius), '-x', label='far field formulation')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift/(rho*g*a²*r)')
    plt.show()


### Mesh utilities for near field formulation ###
def edges_water_line(mesh): 
    """Return array of shape (nb_edges_water_line,2) with the indices of the vertices"""
    epsilon = 1e-6
    indices_vertices = []
    edges_water_line = []
    for i in range(mesh.nb_vertices):
        if np.abs(mesh.vertices[i,-1]) <= epsilon:        
            indices_vertices.append(i)

    for k in range(mesh.nb_faces):
        list_indices = [index for index in mesh.faces[k,:] if index in indices_vertices]
        if len(list_indices) == 2:
            edges_water_line.append(list_indices)

    return np.array(edges_water_line)

def length_edges_water_line(mesh):
    """Return array of shape (nb_edges_water_line) with the length of the vertices"""
    size = np.shape(edges_water_line(mesh))[0]
    edges = edges_water_line(mesh)
    vertices_left = mesh.vertices[edges[:,0],:]
    vertices_right = mesh.vertices[edges[:,1],:]
    length_water_line = np.zeros(size)
    for k in range(size):
        length_water_line[k] = np.linalg.norm(vertices_left[k] - vertices_right[k],2)
    return length_water_line

def water_line_integral(mesh, data):
    """Return integral of given data along water line"""

    length = length_edges_water_line(mesh)
    return sum(length[k]*data[k] for k in range(np.shape(length)[0]))


### Near field formulation ###
def near_field(mesh, res, solver, pb):
    rho = pb.rho
    g = pb.g

    velocity = solver.compute_velocity(mesh, res)
    velocity_square = np.sum(np.abs(velocity)**2, axis=1) 
    normal_surface, _ = _normalize_free_surface_points(mesh)
    coef1 = (rho/4)*mesh.surface_integral(velocity_square*normal_surface[:,0]) #je recupere la composante x pour l'instant 

    edges = edges_water_line(mesh)
    vertices_left = mesh.vertices[edges[:,0],:]
    vertices_right = mesh.vertices[edges[:,1],:]
    vertices_middle = (vertices_left + vertices_right)/2
    normal_point, _ = _normalize_points(vertices_middle)

    eta = airy_waves_free_surface_elevation(vertices_middle, res) + solver.compute_free_surface_elevation(vertices_middle, res) 
    coef2 = (rho*g/4)*water_line_integral(mesh, np.abs(eta)**2*normal_point[:,0])  #je recupere la composante x pour l'instant

    return coef1 - coef2



if __name__ == "__main__":
    # g = _default_parameters["g"]
    # rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    # omega_range = [0.5]
    k = 1.25
    radius = 1
    theta_range = np.linspace(0, 2*np.pi, 19)

    mesh = cpt.mesh_sphere(radius=radius, resolution=(50, 50), center=(0,0,0)).immersed_part()
    body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(0,0,0)),
            name=mesh.name,
            center_of_mass=(0,0,0),
            ).immersed_part()
    body.inertia_matrix = body.compute_rigid_body_inertia()
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()

    solver = cpt.BEMSolver()
    problem = [cpt.DiffractionProblem(body=body, wavenumber=k, wave_direction=wave_direction)]
    res = solver.solve_all(problem)
    # test_matrix = xr.Dataset(coords={
    #     'wavenumber': k, 'wave_direction': wave_direction, 'theta': theta_range, 'radiating_dof': list(body.dofs.keys())
    # })
    # dataset = solver.fill_dataset(test_matrix, body, hydrostatics=True)

    near_field_value = near_field(mesh, res[0], solver, problem[0])
    print("Near field value:", near_field_value)
    far_field_value = 5159.32773366
    print("Far field value:", far_field_value)