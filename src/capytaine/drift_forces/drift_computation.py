import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from scipy.integrate import trapezoid 

import capytaine as cpt 
from capytaine.bem.problems_and_results import _default_parameters
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation, airy_waves_velocity
from capytaine.post_pro.rao import rao
from capytaine.meshes.geometry import compute_faces_normals
from capytaine.io.xarray import assemble_dataset, kochin_data_array
from capytaine.bem.problems_and_results import DiffractionProblem, DiffractionResult



### Far field formulation ###
def compute_kochin_global(X, dataset, a):
    Hd = dataset['kochin_diffraction']
    Hr = dataset['kochin_radiation'] 
    sum_HdX = sum(Hr[:,i,:]*X[:,0,i] for i in range(6)) 
    return a*np.exp(1j*np.pi/2)*(Hd + sum_HdX)

def far_field(X, dataset, a):
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
 
    H = compute_kochin_global(X, dataset, a)[:,0,:]
    H_beta = H[:,np.where(theta == beta)[0][0]]
    coef1 = 2*np.pi*rho*a*omega
    
    F = np.array([
        -coef1*np.cos(beta)*np.imag(H_beta) - coef2*trapezoid(np.abs(H)**2*np.cos(theta), theta),
        -coef1*np.sin(beta)*np.imag(H_beta) - coef2*trapezoid(np.abs(H)**2*np.sin(theta), theta)
        ])

    return F



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
def near_field(X, mesh, body, results, solver, dataset, a):
    rho = dataset['rho']
    g = dataset['g']
    omega = dataset['omega'].data

    def d(x):
        size = np.shape(x)[0]
        d = np.empty([len(omega),size,3], dtype=complex)
        trans = X[:,0,:3]
        rot = X[:,0,3:]
        for line in range(size):
            d[:,line,:] = trans + np.cross(rot, x[line,:])
        return d
    
    forces_first_order = dataset['excitation_force'][:,0,:] + compute_radiation_forces(X, dataset)[:,0,:]

    grad_phi = compute_total_velocity(solver, mesh, results, X, omega)
    grad_phi_square = np.sum(np.abs(grad_phi)**2, axis=2) 
    time_der_grad_phi = compute_time_der_velocity(mesh, omega, d, grad_phi)
    pressure_field = grad_phi_square/4 + time_der_grad_phi/2
    coef1 = np.empty([len(omega),3])
    for i in range(len(omega)):
        for dim in range(3):
            coef1[i,dim] = rho*mesh.surface_integral(pressure_field[i,:]*mesh.faces_normals[:,dim])  

    edges, faces = edges_water_line(mesh)
    vertices_left = mesh.vertices[edges[:,0],:]
    vertices_right = mesh.vertices[edges[:,1],:]
    vertices_middle = ((vertices_left + vertices_right)/2) #[:,:-1]
    normal = compute_faces_normals(mesh.vertices, faces)
    # n = normal[:,0] / np.sqrt(1-normal[:,-1]**2)

    eta =  compute_free_surface_elevation(solver, vertices_middle, results, X, omega)
    d3 = d(vertices_middle)[:,:,-1]
    coef2 = np.zeros([len(omega),3])
    for i in range(len(omega)):
        for dim in range(2):
            coef2[i,dim] = (rho*g/4)*water_line_integral(mesh, (np.abs(eta[i,:] - d3[i,:])**2*normal[:,dim])) 

    coef3 = np.empty([len(omega),3])
    for i in range(len(omega)):
        coef3[i,:] = 0.5*np.real(np.cross(X[i,0,3:], np.conjugate(forces_first_order[i,0:3])))

    coef4 = np.zeros([len(omega),3])
    S = body.hydrostatic_stiffness
    for i in range(len(omega)):
        coef4[i,-1] = 1/4*(S[2,2]*(np.abs(X[i,:,3])**2 + np.abs(X[i,:,4])**2) \
                        - S[2,3]* np.real(X[i,:,4]*np.conjugate(X[i,:,5])) \
                        + S[2,4]*np.real(X[i,:,3]*np.conjugate(X[i,:,5])))

    return a**2*(coef1 - coef2 + coef3 + coef4), pressure_field

def compute_free_surface_elevation(solver, vertices_middle, results, X, omega):
    len_omega = len(omega)
    size = np.shape(vertices_middle)[0]
    eta_incident = np.empty([len_omega,size], dtype=complex)
    eta_diffraction = np.empty([len_omega,size], dtype=complex)
    eta_radiation = np.zeros([len_omega,size], dtype=complex)

    idx_omega = 0
    for res in results:
        if isinstance(res, DiffractionResult):
            eta_incident[idx_omega,:] = airy_waves_free_surface_elevation(vertices_middle, res)
            eta_diffraction[idx_omega,:] = solver.compute_free_surface_elevation(vertices_middle, res) 
            idx_omega += 1

    for idx_omg, omg in enumerate(omega):
        for idx_res, res in enumerate(results):
            if res.__class__ == cpt.bem.problems_and_results.RadiationResult and res.omega == omg:
                eta_radiation[idx_omg,:] += X[idx_omg,0,idx_res%7].item()*solver.compute_free_surface_elevation(vertices_middle, res)

    return eta_incident + eta_diffraction + eta_radiation


def compute_time_der_velocity(mesh, omega, d, grad_phi):
    time_der_grad_phi = np.empty([len(omega),mesh.nb_faces])
    dist = d(mesh.faces_centers)
    for i in range(len(omega)):
        time_der_grad_phi[i,:] = np.real(np.sum(dist[i,:,:]*np.conjugate(-1j * omega[i] * grad_phi[i,:,:]), axis=1))

    return time_der_grad_phi


def compute_total_velocity(solver, mesh, results, X, omega):
    len_omega = len(omega)
    grad_phi_incident = np.empty([len_omega,mesh.nb_faces,3], dtype=complex)
    grad_phi_diffraction = np.empty([len_omega,mesh.nb_faces,3], dtype=complex)
    grad_phi_radiation = np.zeros([len_omega,mesh.nb_faces,3], dtype=complex)

    idx_omega = 0
    for res in results:
        if isinstance(res, DiffractionResult):
            grad_phi_incident[idx_omega,:,:] = airy_waves_velocity(mesh, res)
            grad_phi_diffraction[idx_omega,:,:] = solver._compute_potential_gradient(mesh, res)
            idx_omega += 1

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




### EXAMPLE ### 
def hemisphere_example():
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    k = np.array([0.3, 0.5, 0.83, 0.92, 1.0, 1.05, 1.09, 1.16, 1.24, 1.39, 1.59, 1.88, 2.28])
    radius = 1
    theta_range = np.linspace(0, 2*np.pi, 19)

    resolution = (50,50)
    mesh = cpt.mesh_sphere(radius=radius, resolution=resolution, center=(0,0,0)).immersed_part()
    lid_mesh = mesh.generate_lid()
    body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(0,0,0)),
            name=mesh.name,
            center_of_mass=(0,0,0),
            lid_mesh=lid_mesh,
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
    dataset = assemble_dataset(res)
    kochin = kochin_data_array(res, theta_range)
    dataset.update(kochin)

    X = rao(dataset)

    F, pressure_field = near_field(X, mesh, body, res, solver, dataset, a)
    # mesh.show(backend="pyvista", color_field=pressure_field[5,:], cmap=plt.get_cmap("hot"))

    F_analytic=[0, 0, 0.07, 0.26, 0.49, 0.70, 0.83, 0.88, 0.84, 0.72, 0.66, 0.655, 0.652]
    F_Delhommeau= [0.035, 0.8373, 0.6700, 0.6265]
    ka_Delhommeau= [0.77, 1.29, 1.53, 2.015]
    factor = rho*g*a**2*radius

    # plt.plot(k, F_analytic, '-o', label = 'analytical')
    # plt.plot(ka_Delhommeau, F_Delhommeau, '*', label = 'Delhommeau')
    # plt.grid()
    # plt.plot(k, far_field(X, dataset, a)[0]/factor, '-x', label='far field formulation')
    plt.plot(k, F[:,-1]/factor, '-v', label='near field formulation')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift/(rho*g*a²*r)')
    plt.title(f"resolution = {resolution}, with lid")
    plt.show()


def cylinder_example():
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    k = np.linspace(0.1,3,20)
    theta_range = np.linspace(0, 2*np.pi, 19)
    radius = 1

    mesh = cpt.mesh_vertical_cylinder(length=20, radius=radius, resolution=(10,50,50)).immersed_part()
    lid_mesh = mesh.generate_lid()
    body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(0,0,0)),
            name=mesh.name,
            center_of_mass=(0,0,0),
            lid_mesh=lid_mesh,
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
    dataset = assemble_dataset(res)
    kochin = kochin_data_array(res, theta_range)
    dataset.update(kochin)

    X = 0*rao(dataset)

    F, pressure_field = near_field(X, mesh, body, res, solver, dataset, a)
    # mesh.show(backend="pyvista", color_field=pressure_field[5,:], cmap=plt.get_cmap("hot"))
    factor = rho*g*radius*a**2

    plt.grid()
    plt.plot(k, far_field(X, dataset, a)[0]/factor, '-x', label='far field formulation')
    plt.plot(k, F[:,0]/factor, '-v', label='near field formulation')
    plt.legend()
    plt.xlabel('w')
    plt.ylabel('F_drift')
    plt.show()


def barge_example():
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    water_depth = 50
    omega = np.array([0.3, 0.5, 0.8, 1., 1.2])
    theta_range = np.linspace(0, 2*np.pi, 19)

    resolution = (100,40,20)
    L = 150
    B = 50
    mesh = cpt.mesh_parallelepiped(size=(L,B,20), resolution=resolution).immersed_part()
    # lid_mesh = mesh.generate_lid()
    lid_mesh = cpt.mesh_rectangle(size=(L,B), resolution=resolution[:2])
    lid_mesh.show()
    body = cpt.FloatingBody(
            mesh=mesh,
            dofs=cpt.rigid_body_dofs(rotation_center=(0,0,0)),
            name=mesh.name,
            center_of_mass=(0,0,0),
            lid_mesh=lid_mesh,
            ).immersed_part()
    body.inertia_matrix = body.compute_rigid_body_inertia()
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()

    solver = cpt.BEMSolver()
    problems = [
        cpt.RadiationProblem(body=body, radiating_dof=dof, omega=omg, water_depth=water_depth)
        for dof in body.dofs
        for omg in omega
    ]
    problems.extend([cpt.DiffractionProblem(body=body, omega=omg, wave_direction=wave_direction, water_depth=water_depth) for omg in omega])

    res = solver.solve_all(problems)
    dataset = assemble_dataset(res)
    kochin = kochin_data_array(res, theta_range)
    dataset.update(kochin)

    X = rao(dataset)

    F, _ = near_field(X, mesh, body, res, solver, dataset, a)

    x = dataset['omega']*np.sqrt(L/g)
    factor = 0.5*rho*g*a**2*B**2/L 

    plt.grid()
    plt.plot(x, far_field(X, dataset, a)[0]/factor, '-x', label='far field formulation')
    plt.plot(x, F[:,-1]/factor, '-v', label='near field formulation')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift/(rho*g*a²*r)')
    plt.title(f"resolution = {resolution}, with lid")
    plt.show()



if __name__ == "__main__":
    hemisphere_example()
