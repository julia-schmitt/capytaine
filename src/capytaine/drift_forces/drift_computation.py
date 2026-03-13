import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from scipy.integrate import trapezoid 
from collections import Counter
import pyvista as pv



import capytaine as cpt 
from capytaine.bem.problems_and_results import _default_parameters
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation, airy_waves_velocity
from capytaine.post_pro.rao import rao
from capytaine.meshes.geometry import compute_faces_normals
from capytaine.io.xarray import assemble_dataset, kochin_data_array
from capytaine.bem.problems_and_results import DiffractionResult, RadiationResult



### Far field formulation ###
    
def far_field(X, dataset):
    omega = dataset['omega']
    k = dataset['wavenumber']
    h = dataset['water_depth']
    beta = dataset['wave_direction']
    rho = dataset['rho']
    g = dataset['g']
    H_diff = dataset['kochin_diffraction']
    H_rad = dataset['kochin_radiation'] 

    H_rad_tot = sum(H_rad.sel(radiating_dof=d)*X.sel(radiating_dof=d) for d in X.radiating_dof) 
    H_tot = np.exp(1j*np.pi/2)*(H_diff + H_rad_tot)
    H_beta = H_tot.sel(theta=beta)

    coef1 = 2*np.pi*rho*omega
    if h == np.inf:
        coef2 = 2*np.pi*rho*k**2
    else:
        k0 = omega**2/g 
        coef2 = 2*np.pi*rho*k*(k0*h)**2 / (h*((k*h)**2 - (k0*h)**2 + k0*h))

    dims = [d for d in X.dims if d != 'radiating_dof']
    coords = {c: X.coords[c] for c in X.coords if c != 'radiating_dof'}

    base = np.abs(H_tot)**2
    
    Fx = xr.DataArray(
        data=-coef1*np.cos(beta)*np.imag(H_beta) - coef2*(base * np.cos(H_tot.theta)).integrate("theta"),
        coords=coords,
        dims=dims,
        name='drift_force_surge'
        )
    
    Fy = xr.DataArray(
        data=-coef1*np.sin(beta)*np.imag(H_beta) - coef2*(base * np.sin(H_tot.theta)).integrate("theta"),
        coords=coords,
        dims=dims,
        name='drift_force_sway'
        )
    
    return xr.Dataset({Fx.name: Fx, Fy.name: Fy})


#documenter la taille, dataarray renvoyer 2 (puis 3) pu un dataset avec les variables horizontales puis la rotation veticale 
#ouvrir une PR avec far field + test et doc dans un fichier drift force dans post pro 
#mettre le a**2 autre part 



### Mesh utilities for near field formulation ###
#PR a part + test des cas limites 
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
        set_indices = {index for index in mesh.faces[k,:] if index in indices_vertices}
        if len(set_indices) == 2:
            edges_water_line.append(list(set_indices))
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


def test_curvilinear_integration():
    radius = 0.8
    mesh_sphere = cpt.mesh_sphere(radius=radius, resolution=(50, 50), center=(0,0,0)).immersed_part()
    l = 2
    L = 3
    h = 4
    mesh_rectangle = cpt.mesh_parallelepiped(size=(l,L,h), resolution=(10*l,10*L,10*h)).immersed_part()
    length_sphere = length_edges_water_line(mesh_sphere)
    integral_sphere = water_line_integral(mesh_sphere, np.ones(np.shape(length_sphere)[0]))
    length_rectangle = length_edges_water_line(mesh_rectangle)
    integral_rectangle = water_line_integral(mesh_rectangle, np.ones(np.shape(length_rectangle)[0]))
    assert np.isclose(integral_sphere, 2*np.pi*radius, rtol=1e-3)
    assert np.isclose(integral_rectangle, 2*(l+L))



### Near field formulation ###
    #on suppose qu'on a une fonction qui a toutes les radiations à cette freq et une diffraction et on itère pour les freq 
    # les 6 + 1 + 1 pb sont cohérents 
    # a terme 2 au lieu de 1 + 1 
    # virer le dataset 
    #type de retour 3 diff (x,y,z) en translation 
    # on suppose qu'on a un seul objet rigide 

def near_field(X, results, solver, dataset):
    # tous les res ont le meme body et le meme mesh ? 
    mesh = results[0].body.mesh 
    S = results[0].body.hydrostatic_stiffness 
    # tous les res ont le meme rho et g ?  
    rho = results[0].rho 
    g = results[0].g

    omega = X.coords['omega']
    len_freq = len(omega)
    len_dir = len(X.coords['wave_direction'])

    pressure_field = compute_pressure_field(mesh, solver, results, X).values
    coef1 = surface_integral_term(rho, mesh, len_freq, len_dir, pressure_field)

    free_surf_elev_field = free_surface_elevation_field(mesh, solver, results, X)
    coef2 = water_line_integral_term(rho, g, mesh, len_freq, len_dir, free_surf_elev_field)

    hydro = compute_hydro_forces(S, len_freq, X)
    forces_first_order = dataset['excitation_force'][:,0,:] + compute_radiation_forces(X, dataset, omega, results)[:,0,:] + hydro 
    coef3 = forces_first_order_term(X, len_freq, forces_first_order)
    coef4 = forces_hydrostatics_term(len_freq, X, S)

    forces = coef1 + coef2 + coef3 + coef4

    dims = [d for d in X.dims if d != 'radiating_dof']
    coords = {c: X.coords[c] for c in X.coords if c != 'radiating_dof'}

    # print(forces)
    # print(np.shape(forces))

    # F = xr.DataArray(
    #     data=forces,
    #     coords=coords,
    #     dims=dims,
    #     name='drift_force'
    #     )

    return forces

def d(x, X):
    trans = X.sel(radiating_dof=['Surge', 'Sway', 'Heave']).values
    rot = X.sel(radiating_dof=['Roll', 'Pitch', 'Yaw']).values
    res = trans[:, :, None, :] + np.cross(rot[:, :, None, :], x[None, None, :, :])
    return res
    

def compute_pressure_field(mesh, solver, results, X):
    grad_phi = total_potential_gradient(solver, mesh, results, X)
    omega = grad_phi.coords['omega']
    time_der_grad_phi = time_derivative_potential_gradient(mesh, d, grad_phi, omega, X)
    grad_phi_square = np.sum(np.abs(grad_phi)**2, axis=3) 
    return grad_phi_square/4 + time_der_grad_phi/2

def free_surface_elevation_field(mesh, solver, results, X):
    edges, faces = edges_water_line(mesh)
    vertices_middle = ((mesh.vertices[edges[:,0],:] + mesh.vertices[edges[:,1],:])/2) 
    
    eta = total_free_surface_elevation(solver, vertices_middle, results, X)
    d3 = d(vertices_middle,X)[:,:,:,-1]

    return np.abs(eta - d3)**2

def surface_integral_term(rho, mesh, len_freq, len_dir, pressure_field):
    res = np.empty([len_freq, len_dir, 3])
    for beta in range(len_dir):
        for i in range(len_freq):
            for dim in range(3):
                res[i,beta,dim] = rho*mesh.surface_integral(pressure_field[i,beta,:]*mesh.faces_normals[:,dim]) 
    return res  

def water_line_integral_term(rho, g, mesh, len_freq, len_dir, free_surface_elevation_field):
    edges, faces = edges_water_line(mesh)
    normal = compute_faces_normals(mesh.vertices, faces)
    normal[:,-1] = 0
    for k in range(np.shape(normal)[0]):
        normal[k,:] /= np.sqrt(normal[k,0]**2 + normal[k,1]**2)
    res = np.empty([len_freq,len_dir,3])
    for beta in range(len_dir):
        for i in range(len_freq):
            for dim in range(2):
                res[i,beta,dim] = (rho*g/4)*water_line_integral(mesh, free_surface_elevation_field[i,beta,:]*normal[:,dim])
    return -res

def forces_first_order_term(X, len_freq, forces_first_order):
    # for res in results:
    #     freq = getattr(res, main_freq_type)

    #     if isinstance(res, DiffractionResult):
    #         phi_grad['incident'].loc[{main_freq_type: freq}] = airy_waves_velocity(mesh, res)
    #         phi_grad['diffraction'].loc[{main_freq_type: freq}] = solver._compute_potential_gradient(mesh, res)

    #     if isinstance(res, RadiationResult):
    #         X_rad = X.sel({main_freq_type: freq, 'radiating_dof': res.radiating_dof}).values
    #         phi_grad['radiation'].loc[{main_freq_type: freq}] += X_rad * solver._compute_potential_gradient(mesh, res)

    res = np.empty([len_freq,3])
    for i in range(len_freq):
        res[i,:] = 0.5*np.real(np.cross(X[i,0,3:], np.conjugate(forces_first_order[i,0:3])))
    return res

def forces_hydrostatics_term(len_freq, X, S):
    res = np.zeros([len_freq,3])
    for i in range(len_freq):
        res[i,-1] = 1/4*(S[2,2]*(np.abs(X[i,:,3])**2 + np.abs(X[i,:,4])**2) \
                        - 2*S[2,3]* np.real(X[i,:,4]*np.conjugate(X[i,:,5])) \
                        + 2*S[2,4]*np.real(X[i,:,3]*np.conjugate(X[i,:,5])))
    return res

def compute_hydro_forces(S, len_freq, X):
    hydro = np.zeros([len_freq,6], dtype=complex)
    for i in range(len_freq):
        hydro[i,2] = -(S[2,2]*X[i,:,2] + S[2,3]*X[i,:,3] + S[2,4]*X[i,:,4])

    return hydro 

def total_free_surface_elevation(solver, vertices_middle, results, X):
    main_freq_type = Counter((res.provided_freq_type for res in results)).most_common(1)[0][0] #passer en arg 
    len_freq = len(X.coords['omega'])
    len_dir = len(X.coords['wave_direction'])
    size = np.shape(vertices_middle)[0]
    eta = xr.Dataset(coords={main_freq_type: X.coords[main_freq_type], 'wave_direction': X.coords['wave_direction'], 'vertices_water_line': np.arange(size)})

    empty_data = np.zeros((len_freq, len_dir, size), dtype=complex)
    eta['incident'] = ([main_freq_type, 'wave_direction', 'vertex'], empty_data)
    eta['diffraction'] = ([main_freq_type, 'wave_direction', 'vertex'], empty_data.copy())
    eta['radiation'] = ([main_freq_type, 'wave_direction', 'vertex'], empty_data.copy())

    for res in results:
        freq = getattr(res, main_freq_type)

        if isinstance(res, DiffractionResult):
            eta['incident'].loc[{main_freq_type: freq}] = airy_waves_free_surface_elevation(vertices_middle, res)
            eta['diffraction'].loc[{main_freq_type: freq}] = solver.compute_free_surface_elevation(vertices_middle, res) 

        elif isinstance(res, RadiationResult):
            X_rad = X.sel({main_freq_type: freq, 'radiating_dof': res.radiating_dof}).values
            eta['radiation'].loc[{main_freq_type: freq}] += X_rad * solver.compute_free_surface_elevation(vertices_middle, res) 

    return eta['incident'] + eta['diffraction'] + eta['radiation']


def total_potential_gradient(solver, mesh, results, X):
    main_freq_type = Counter((res.provided_freq_type for res in results)).most_common(1)[0][0] #passer en arg 
    len_freq = len(X.coords['omega'])
    len_dir = len(X.coords['wave_direction'])
    size = mesh.nb_faces
    phi_grad = xr.Dataset(coords={main_freq_type: X.coords[main_freq_type], 'wave_direction': X.coords['wave_direction'], 'face': np.arange(size), 'space': np.arange(3)})
    empty_data = np.zeros((len_freq, len_dir, size, 3), dtype=complex)
    phi_grad['incident'] = ([main_freq_type, 'wave_direction', 'face', 'space'], empty_data)
    phi_grad['diffraction'] = ([main_freq_type, 'wave_direction', 'face', 'space'], empty_data.copy())
    phi_grad['radiation'] = ([main_freq_type, 'wave_direction', 'face', 'space'], empty_data.copy())

    for res in results:
        freq = getattr(res, main_freq_type)

        if isinstance(res, DiffractionResult):
            phi_grad['incident'].loc[{main_freq_type: freq}] = airy_waves_velocity(mesh, res)
            phi_grad['diffraction'].loc[{main_freq_type: freq}] = solver._compute_potential_gradient(mesh, res)

        if isinstance(res, RadiationResult):
            X_rad = X.sel({main_freq_type: freq, 'radiating_dof': res.radiating_dof}).values
            phi_grad['radiation'].loc[{main_freq_type: freq}] += X_rad * solver._compute_potential_gradient(mesh, res)

    return phi_grad['incident'] + phi_grad['diffraction'] + phi_grad['radiation']


def time_derivative_potential_gradient(mesh, d, grad_phi, omega, X):
    dist = d(mesh.faces_centers,X)
    time_der_grad_phi = np.real(np.sum(dist*np.conjugate(-1j * omega * grad_phi), axis=3))
    return time_der_grad_phi


def compute_radiation_forces(X, dataset, omega, results):
    A = dataset["added_mass"]
    B = dataset["radiation_damping"]
    F = np.zeros(np.shape(A), dtype=complex)
    for m in range(len(omega)):
        for i in range(6):
            F[m,0,i] = sum((omega[m]**2*A[m,i,k] + 1j*omega[m]*B[m,i,k])*X[m,0,k] for k in range(6))

    # force = xr.DataArray(data=np.zeros(list(X.sizes.values()), dtype=complex), coords=X.coords, dims=X.dims)

    # for res in results:
    #     if isinstance(res, RadiationResult):
    #         print(res.added_mass)
    #         print(res.added_mass['Surge'])
    #         print(X.radiating_dof)
    #         force.loc[{'omega':res.omega, 'radiating_dof':res.radiating_dof}] = sum((res.omega**2*res.added_mass[dof.values] + 1j*res.omega*res.radiation_damping[dof.values])*X.sel(radiating_dof=dof) for dof in X.radiating_dof)
            

    return F







### EXAMPLE ### 
def hemisphere_example():
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
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

    F_analytic=[0, 0, 0.07, 0.26, 0.49, 0.70, 0.83, 0.88, 0.84, 0.72, 0.66, 0.655, 0.652]
    k = np.array([0.3, 0.5, 0.83, 0.92, 1.0, 1.05, 1.09, 1.16, 1.24, 1.39, 1.59, 1.88, 2.28])
    # k = np.array([0.8, 0.2])

    problems = [
        cpt.RadiationProblem(body=body, radiating_dof=dof, wavenumber=ki)
        for dof in body.dofs
        for ki in k
    ]
    problems.extend([cpt.DiffractionProblem(body=body, wavenumber=ki, wave_direction=wave_direction) for ki in k])

    res = solver.solve_all(problems)

    # for r in res:
    #     if isinstance(r, DiffractionResult):
    #         print(len(r.records))
    #         print(r.records[0]['diffraction_force'])

    dataset = assemble_dataset(res)
    kochin = kochin_data_array(res, theta_range)
    dataset.update(kochin)
    X = rao(dataset)
    F_near = near_field(X, res, solver, dataset)
    factor = rho*g*a**2*radius

    plt.plot(k, F_analytic, '-o', label = 'analytical')
    plt.plot(k, F_near[:,0]/factor, '-v', label='near field formulation')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift/(rho*g*a²*r)')
    plt.show()





def barge_example():
    g = _default_parameters["g"]
    rho = _default_parameters["rho"]
    a = 1
    wave_direction = 0
    water_depth = 50
    theta_range = np.linspace(0, 2*np.pi, 19)
    V = 73750
    X_analytic= np.array([0.126, 0.467, 0.737, 1.006, 1.059, 1.130, 1.147, 1.165, 1.181, 1.285, 1.479, 1.549, 1.710, 1.907])
    omega = X_analytic / np.sqrt((V**(1/3)/g))
    Y_analytic= [0, 0, 0, 0.031, 0.087, 0.142, 0.198, 0.254, 0.356, 0.689, 1.042, 1.172, 1.228, 1.293]

    resolution = (50,20,10)
    L = 150
    B = 50
    mesh = cpt.mesh_parallelepiped(size=(L,B,20), resolution=resolution).immersed_part()
    # mesh.show()
    lid_mesh = mesh.generate_lid()
    # lid_mesh.show()
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

    X = 0*rao(dataset)

    F, _ = near_field(X, mesh, body, res, solver, dataset, a)

    x = dataset['omega']*np.sqrt(L/g)
    factor = 0.5*rho*g*a**2*V**(1/3)

    plt.grid()
    plt.plot(x, far_field(X, dataset, a)[0]/factor, '-x', label='far field formulation', color='green')
    plt.plot(x, F[:,-1]/factor, '-v', label='near field formulation', color='blue')
    plt.plot(X_analytic, Y_analytic, '-o', label = 'analytical', color = 'black')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift/(rho*g*a²*r)')
    plt.title(f"resolution = {resolution}, with lid")
    plt.show()


def barge_DNV_example():
    # Coef d'inertie à mettre quelque part ? 
    a = 1
    wave_direction = 0
    theta_range = np.linspace(0, 2*np.pi, 19)

    resolution = (10,10,9)
    mesh = cpt.mesh_parallelepiped(size=(90,90,80), resolution=resolution).immersed_part()
    lid_mesh = mesh.generate_lid()
    # lid_mesh.show()
    F_Delhommeau = [456776.57, 400045.81, 404889.49, 464429.14, 344675.27, 233204.32, 617723.82, 41779.49, 1647.74, 1293.86, 641.40]
    T= np.array([6.46, 8.44, 9.24, 10.32, 12.23, 14.20, 16.23, 18.14, 20.07, 22.20, 26.13])
    # T=np.array([6,8])
    # omega = 2*np.pi/T
    omega = [0.8]

    F_DNV = [515895.9857, 339709.948, 249803.3144,	538079.5905, 177800.5972, 121125.1363]
    T_DNV =[10.071552, 12.157682, 14.133594, 16.344162, 18.18097, 19.832452]

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
        cpt.RadiationProblem(body=body, radiating_dof=dof, omega=omg)
        for dof in body.dofs
        for omg in omega
    ]
    problems.extend([cpt.DiffractionProblem(body=body, omega=omg, wave_direction=wave_direction) for omg in omega])

    res = solver.solve_all(problems)
    dataset = assemble_dataset(res)
    kochin = kochin_data_array(res, theta_range)
    dataset.update(kochin)

    X = rao(dataset)

    # F_far = far_field(X, dataset)['drift_force_surge']
    # pressure_field = compute_pressure_field(mesh, solver, res, X)[0,:].data
    # plotter = pv.Plotter()
    # plotter = mesh.show(backend="pyvista", color_field=pressure_field, cmap=plt.get_cmap("RdBu"), plotter=plotter)
    # # plotter.view_xy(negative=True)
    # plotter.view_xy(negative=True)
    # plotter.show()
    F_near = near_field(X, res, solver, dataset)
    factor = a**2

    # plt.grid()
    # plt.plot(T, F_far/factor, '-x', label='far field formulation', color='green')
    # plt.plot(T, F_near[:,0]/factor, '-v', label='near field formulation', color='blue')
    # plt.plot(T, F_Delhommeau, '-o', label = 'Delhommeau', color = 'black')
    # plt.plot(T_DNV, F_DNV, '-^', label='experiments', color = 'red')
    # plt.legend()
    # plt.xlabel('T (period)')
    # plt.ylabel('Fx/a²')
    # plt.title(f"Caisson DNV, resolution = {resolution}")
    # plt.show()


if __name__ == "__main__":
    hemisphere_example()
    # barge_DNV_example()
