from cold_spray import *

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

init_obj = 'meshes/Stair_Substrate.obj'
init_obj = MeshHandler(init_obj, device=device)
init_mesh = init_obj.mesh
number_of_vertices = len(init_mesh.verts_packed())
init_length_x, init_length_y = init_obj.mesh_dimensions()

stand_off_dist = 30 # Stand-off distance in mm
Nozzle_angle = -15 # Nozzle angle in degrees
Nozzle_speed =  0 #Nozzle speed (mm/s)
Nozzle_radius = 4.5 #Rayon de la buse (mm)
nozzle = Nozzle(Nozzle_radius, Nozzle_speed, Nozzle_angle, stand_off_dist, device=device)

n_p = 1
dt = 1e-2 # time step
tf = 0.3 # Final time (s)
n_t = int(tf / dt) + 1 # number of time intervals

mesh_modification = MeshModification(nozzle, init_mesh, device=device)
mesh_modification.create_delta_matrix()

for pass_idx in range(n_p):
    for m in range(n_t):
        nozzle.position = torch.stack([(3 + nozzle.stand_off_dist * torch.sin(nozzle.angle_beta)),
                                        torch.tensor(0., device = device), 
                                        nozzle.stand_off_dist * torch.cos(nozzle.angle_beta)])
        nozzle.normal = torch.reshape(torch.stack((-torch.sin(nozzle.angle_beta),
                                                    torch.tensor(0.0, device=device), 
                                                    -torch.cos(nozzle.angle_beta))),[1,3])
        
        nozzle_ray_origins, nozzle_ray_directions = nozzle.generate_rays(nozzle.position, nozzle.normal)

        ray_traced_cells_indices, ray_traced_vertices = mesh_modification.moller_trumbore_multi_ray_trace(nozzle_ray_origins, nozzle_ray_directions, Multi_hit=False)
        
        mesh_modification.vertex_modification(ray_traced_vertices, ray_traced_cells_indices, mesh_modification.mesh_properties(), dt)

final_mesh_path = f'meshes/Mesh_simulation_Shadow_Effect.obj'
mesh_modification.export_obj(final_mesh_path)

