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


def test_curvilinear_integration():
    radius = 1
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
def near_field(X, mesh, body, results, solver, dataset):
    # mesh via body via res 
    #on suppose qu'on a une fonction qui a toutes les radiations à cette freq et une diffraction et on itère pour les freq 
    # les 6 + 1 + 1 pb sont cohérents 
    # a terme 2 au lieu de 1 + 1 
    #type de retour 3 diff (x,y,z) en translation 
    # on suppose qu'on a un seul objet rigide 
    # faire des fonctions indép pour chaque terme et aussi quantifier 
    rho = dataset['rho']
    g = dataset['g']
    omega = dataset['omega'].data

    def d(x):
        size = np.shape(x)[0]
        d = np.empty([len(omega),size,3], dtype=complex)
        trans = X[:,0,:3]
        rot = X[:,0,3:]
        for line in range(size):
            for w in range(len(omega)):
                d[w,line,:] = trans[w,:] + np.cross(rot[w,:], x[line,:])
        return d
    
    S = body.hydrostatic_stiffness #body depuis res 
    hydro = np.zeros([len(omega),6], dtype=complex)
    for i in range(len(omega)):
        hydro[i,2] = -(S[2,2]*X[i,:,2] + S[2,3]*X[i,:,3] + S[2,4]*X[i,:,4])
    forces_first_order = dataset['excitation_force'][:,0,:] + compute_radiation_forces(X, dataset)[:,0,:] + hydro 

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
    normal[:,-1] = 0
    for k in range(np.shape(normal)[0]):
        normal[k,:] /= np.sqrt(normal[k,0]**2 + normal[k,1]**2)

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
    for i in range(len(omega)):
        coef4[i,-1] = 1/4*(S[2,2]*(np.abs(X[i,:,3])**2 + np.abs(X[i,:,4])**2) \
                        - S[2,3]* np.real(X[i,:,4]*np.conjugate(X[i,:,5])) \
                        + S[2,4]*np.real(X[i,:,3]*np.conjugate(X[i,:,5])))

    return (coef1 - coef2 + coef3 + coef4), pressure_field



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

    dof_to_idx = {"Surge":0,"Sway":1,"Heave":2,"Roll":3,"Pitch":4,"Yaw":5}

    for idx_omg, omg in enumerate(omega):
        for res in results:
            if isinstance(res, RadiationResult) and res.omega == omg: 
                dof_idx = dof_to_idx[res.radiating_dof]
                eta_radiation[idx_omg,:] += X[idx_omg, 0, dof_idx].item()*solver.compute_free_surface_elevation(vertices_middle, res)
                #datarray .sel(radiating_dofs='Surge') iterer dessus 

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

    dof_to_idx = {"Surge":0,"Sway":1,"Heave":2,"Roll":3,"Pitch":4,"Yaw":5}

    for idx_omg, omg in enumerate(omega):
        for res in results:
            if isinstance(res, RadiationResult) and res.omega == omg:
                dof_idx = dof_to_idx[res.radiating_dof]
                grad_phi_radiation[idx_omg, :, :] += X[idx_omg, 0, dof_idx].item() * solver._compute_potential_gradient(mesh, res)

    return grad_phi_incident + grad_phi_diffraction + grad_phi_radiation



def compute_radiation_forces(X, dataset):
    A = dataset["added_mass"]
    B = dataset["radiation_damping"]
    F = np.zeros(np.shape(A), dtype=complex)
    omega = dataset["omega"]
    for m in range(len(omega)):
        for i in range(6):
            F[m,0,i] = sum((omega[m]**2*A[m,i,k] + 1j*omega[m]*B[m,i,k])*X[m,0,k] for k in range(6))

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

    kx = np.array([0.3, 0.5, 0.83, 0.92, 1.0, 1.05, 1.09, 1.16, 1.24, 1.39, 1.59, 1.88, 2.28])
    kx = kx/radius
    kz = [0.09582684505613773, 0.18238014348096274, 0.29675431763881227, 0.43585787086713385, 0.602782181908937, 0.7357032134903257, 0.8902627170773499, 0.9737249905177946, 1.0139103199469692, 1.0602781710230764, 1.1221019724578862, 1.16537868063007, 1.2457495753275052, 1.3848531285558265, 1.6228749055832958, 1.987635192545221]

    def solve(k):
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

        F_near, pressure_field = near_field(X, mesh, body, res, solver, dataset)
        # mesh.show(backend="pyvista", color_field=pressure_field[5,:], cmap=plt.get_cmap("hot"))

        return F_near, dataset, X

    Fx = [0, 0, 0.07, 0.26, 0.49, 0.70, 0.83, 0.88, 0.84, 0.72, 0.66, 0.655, 0.652]
    Fz = [0.13281240267679684, 0.2625000022351741, 0.4109374933876102, 0.5624999962747099, 0.6921875064261254, 0.7484374643303475, 0.6953124883584685, 0.5218749927356845, 0.3593750083819027, 0.1531250087544318, -0.11718752281740163, -0.2984374881722037, -0.39687491077930254, -0.4312499802559626, -0.41093743378296876, -0.36562491264194763]
    factor = rho*g*a**2*radius

    Fx_near, dataset, X = solve(kx)
    Fx_far = a**2*far_field(X, dataset)['drift_force_surge']
    # Fz_near, _, _= solve(kz)

    plt.subplot(121)
    # plt.plot(kx*radius, Fx, '-o', label = 'analytical', color = 'black')
    plt.grid()
    plt.plot(kx*radius, Fx_far/factor, '-x', label='far field formulation', color='green')
    plt.plot(kx*radius, Fx_near[:,0]/factor, '-v', label='near field formulation', color='blue')
    plt.legend()
    plt.xlabel('k*r')
    plt.ylabel('F_drift_x/(rho*g*a²*r)')
    plt.title(f"resolution = {resolution}, with lid")

    # plt.subplot(122)
    # plt.grid()
    # plt.plot(kz, Fz, '-o', label = 'analytical', color = 'black')
    # plt.plot(kz, Fz_near[:,-1]/factor, '-v', label='near field formulation', color='blue')
    # plt.legend()
    # plt.xlabel('k*r')
    # plt.ylabel('F_drift_z/(rho*g*a²*r)')
    # plt.title(f"resolution = {resolution}, with lid")
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

    X = rao(dataset)

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

    resolution = (10,10,8)
    mesh = cpt.mesh_parallelepiped(size=(90,90,80), resolution=resolution).immersed_part()
    # mesh.show()
    lid_mesh = mesh.generate_lid()
    # lid_mesh.show()
    F_Delhommeau = [456776.57, 400045.81, 404889.49, 464429.14, 344675.27, 233204.32, 617723.82, 41779.49, 1647.74, 1293.86, 641.40]
    T= np.array([6.46, 8.44, 9.24, 10.32, 12.23, 14.20, 16.23, 18.14, 20.07, 22.20, 26.13])
    omega = np.flip(2*np.pi/T)

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

    F_far = far_field(X, dataset)['drift_force_surge']
    F_near, _ = near_field(X, body.mesh, body, res, solver, dataset)

    factor = a**2

    plt.grid()
    plt.plot(T, np.flip(F_far)/factor, '-x', label='far field formulation', color='green')
    plt.plot(T, np.flip(F_near[:,0])/factor, '-v', label='near field formulation', color='blue')
    plt.plot(T, F_Delhommeau, '-o', label = 'Delhommeau', color = 'black')
    plt.plot(T_DNV, F_DNV, '-^', label='experiments', color = 'red')
    plt.legend()
    plt.xlabel('T (period)')
    plt.ylabel('Fx/a²')
    plt.title(f"Caisson DNV, resolution = {resolution}")
    plt.show()


if __name__ == "__main__":
    # hemisphere_example()
    barge_DNV_example()