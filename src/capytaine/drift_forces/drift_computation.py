import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from scipy.integrate import trapezoid 

import capytaine as cpt 
from capytaine.bem.problems_and_results import _default_parameters
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation, airy_waves_velocity
from capytaine.post_pro.rao import rao
from capytaine.meshes.geometry import compute_faces_normals
from capytaine.io.xarray import assemble_dataset
from capytaine.bem.problems_and_results import DiffractionProblem, DiffractionResult


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
    faces_water_line = []
    for i in range(mesh.nb_vertices):
        if np.abs(mesh.vertices[i,-1]) <= epsilon:        
            indices_vertices.append(i)

    for k in range(mesh.nb_faces):
        list_indices = [index for index in mesh.faces[k,:] if index in indices_vertices]
        if len(list_indices) == 2:
            edges_water_line.append(list_indices)
            faces_water_line.append(mesh.faces[k,:])

    return np.array(edges_water_line), np.array(faces_water_line)

def length_edges_water_line(mesh):
    """Return array of shape (nb_edges_water_line) with the length of the vertices"""
    size = np.shape(edges_water_line(mesh)[0])[0]
    edges = edges_water_line(mesh)[0]
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
def near_field(mesh, results, solver, problems, dataset, a):
    rho = dataset['rho']
    g = dataset['g']
    omega = dataset['omega'].data

    X = rao(dataset) 

    def d(x):
        size = np.shape(x)[0]
        d = np.empty([len(omega),size,3], dtype=complex)
        trans = X[:,0,:3]
        rot = X[:,0,3:]
        for line in range(size):
            d[:,line,:] =  trans + np.cross(rot, x[line,:])
        return d
    
    forces_first_order = dataset['excitation_force'][:,0,:] + compute_radiation_forces(X, dataset)[:,0,:]

    grad_phi = compute_total_velocity(solver, mesh, problems, results, X, omega)
    grad_phi_square = np.sum(np.abs(grad_phi)**2, axis=2) 
    time_der_grad_phi = compute_time_der_velocity(mesh, omega, d, grad_phi)
    coef1 = np.empty([len(omega)])
    for i in range(len(omega)):
        coef1[i] = (rho/4)*mesh.surface_integral((grad_phi_square[i,:] + time_der_grad_phi[i,:])*mesh.faces_normals[:,0]) #je recupere la composante x pour l'instant 

    edges, faces = edges_water_line(mesh)
    vertices_left = mesh.vertices[edges[:,0],:]
    vertices_right = mesh.vertices[edges[:,1],:]
    vertices_middle = ((vertices_left + vertices_right)/2) #[:,:-1]
    normal = compute_faces_normals(mesh.vertices, faces)
    n_x = normal[:,0] / np.sqrt(1-normal[:,-1]**2)

    eta =  compute_free_surface_elevation(solver, vertices_middle, problems, results, X, omega)
    d3 = d(vertices_middle)[:,:,-1]
    coef2 = np.empty([len(omega)])
    for i in range(len(omega)):
        coef2[i] = (rho*g/4)*water_line_integral(mesh, (np.abs(eta[i,:] - d3[i,:])**2*n_x))  #je recupere la composante x pour l'instant

    coef3 = np.empty([len(omega)])
    for i in range(len(omega)):
        coef3[i] = 0.5*np.real(np.cross(X[i,0,3:], np.conjugate(forces_first_order[i,0:3])))[0]

    return a**2*(coef1 - coef2 + coef3)

def compute_free_surface_elevation(solver, vertices_middle, problems, results, X, omega):
    len_omega = len(omega)
    size = np.shape(vertices_middle)[0]
    eta_incident = np.empty([len_omega,size], dtype=complex)
    eta_diffraction = np.empty([len_omega,size], dtype=complex)
    eta_radiation = np.zeros([len_omega,size], dtype=complex)

    idx_omega = 0
    for pb in problems:
        if isinstance(pb, DiffractionProblem):
            eta_incident[idx_omega,:] = airy_waves_free_surface_elevation(vertices_middle, pb)
            idx_omega += 1

    idx_omega_diff = 0 
    for res in results: 
        if isinstance(res, DiffractionResult):
            eta_diffraction[idx_omega_diff,:] = solver.compute_free_surface_elevation(vertices_middle, res)
            idx_omega_diff += 1 

    for idx_omg, omg in enumerate(omega):
        for idx_res, res in enumerate(results):
            if res.__class__ == cpt.bem.problems_and_results.RadiationResult and res.omega == omg:
                eta_radiation[idx_omg,:] += X[idx_omg,0,idx_res%7].item()*solver.compute_free_surface_elevation(vertices_middle, res)

    return eta_incident + eta_diffraction + eta_radiation


def compute_time_der_velocity(mesh, omega, d, grad_phi):
    time_der_grad_phi = np.empty([len(omega),mesh.nb_faces])
    dist = d(mesh.faces_centers)
    for i in range(len(omega)):
        time_der_grad_phi[i,:] = -0.5*np.real(1j * omega[i] * np.sum(dist[i,:,:]*np.conjugate(grad_phi[i,:,:]), axis=1))
    return time_der_grad_phi


def compute_total_velocity(solver, mesh, problems, results, X, omega):
    len_omega = len(omega)
    grad_phi_incident = np.empty([len_omega,mesh.nb_faces,3], dtype=complex)
    grad_phi_diffraction = np.empty([len_omega,mesh.nb_faces,3], dtype=complex)
    grad_phi_radiation = np.zeros([len_omega,mesh.nb_faces,3], dtype=complex)

    idx_omega = 0
    for pb in problems:
        if isinstance(pb, DiffractionProblem):
            grad_phi_incident[idx_omega,:,:] = airy_waves_velocity(mesh, pb)
            idx_omega += 1

    idx_omega_diff = 0 
    for res in results: 
        if isinstance(res, DiffractionResult):
            grad_phi_diffraction[idx_omega_diff,:,:] = solver._compute_potential_gradient(mesh, res)
            idx_omega_diff += 1 

    for idx_omg, omg in enumerate(omega):
        for idx_res, res in enumerate(results):
            if res.__class__ == cpt.bem.problems_and_results.RadiationResult and res.omega == omg:
                grad_phi_radiation[idx_omg,:,:] += X[idx_omg,0,idx_res%7].item()*solver._compute_potential_gradient(mesh, res)

    return grad_phi_incident + grad_phi_diffraction + grad_phi_radiation



def compute_radiation_forces(X, dataset):
    A = dataset["added_mass"]
    B = dataset["radiation_damping"]
    F = np.zeros(np.shape(A), dtype=complex)
    omega = dataset["omega"]
    for m in range(len(omega)):
        for i in range(6):
            F[m,0,i] = sum(omega[m]**2*A[m,i,k] + 1j*omega[m]*B[m,i,k]*X[m,0,k] for k in range(6))

    return F



def test_curvilinear_integration():
    radius = 1
    mesh = cpt.mesh_sphere(radius=radius, resolution=(50, 50), center=(0,0,0)).immersed_part()
    length = length_edges_water_line(mesh)
    integral = water_line_integral(mesh, np.ones(np.shape(length)[0]))
    assert np.isclose(integral, 2*np.pi*radius, rtol=1e-3), "Curvilinear integration along water line failed"


if __name__ == "__main__":
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    k = [0.8, 1.25]
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
    problems = [
        cpt.RadiationProblem(body=body, radiating_dof=dof, wavenumber=ki)
        for dof in body.dofs
        for ki in k 
    ]
    problems.extend([cpt.DiffractionProblem(body=body, wavenumber=ki, wave_direction=wave_direction) for ki in k])


    res = solver.solve_all(problems)
    test_matrix = xr.Dataset(coords={
        'wavenumber': k, 'wave_direction': wave_direction, 'theta': theta_range, 'radiating_dof': list(body.dofs.keys())
    })
    # dataset = solver.fill_dataset(test_matrix, body, hydrostatics=True)
    dataset = assemble_dataset(res)

    # far_field_value = far_field(dataset, a)[0].item()
    # print("Far field value:", far_field_value)
    # value 1424.1530400928964 12519.895535206133
    near_field_value = near_field(mesh, res, solver, problems, dataset, a)
    print("Near field value:", near_field_value)


    # print("error = ", np.abs(far_field_value - near_field_value)/far_field_value * 100)

    # test_curvilinear_integration()