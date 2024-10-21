from cold_spray import *

# Set the appropiate deivce for the optimization computations.
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# Load the target mesh. In this case a 5 pass mesh simulation done at 30° from the z-axis.
target_obj = f'meshes/Mesh_simulation_30_degrees.obj'
target_obj = MeshHandler(target_obj, device=device)
target_mesh = target_obj.mesh

# Load the initial mesh (before applying any spray). In this case a plane substrate.
# init_obj = 'meshes/init_mesh.obj'
init_obj = 'meshes/init_mesh.obj'
init_obj = MeshHandler(init_obj, device=device)
init_mesh = init_obj.mesh

# Get the number of vertices in the initial mesh, which will be used in loss calculation.
number_of_vertices = len(init_mesh.verts_packed())

# Get the dimensions of the initial mesh, used to determine nozzle motion and time steps.
init_length_x, init_length_y = init_obj.mesh_dimensions()

# === Nozzle Parameters ===
# Define the nozzle parameters including stand-off distance, angle, speed, and radius.
stand_off_dist = 30  # Stand-off distance (in mm), distance between nozzle and target.
Nozzle_angle = 0     # Nozzle angle (in degrees), angle of the nozzle with respect to the z-axis of the reference frame.
Nozzle_speed = 100   # Nozzle speed (in mm/s), speed at which the nozzle moves across the target.
Nozzle_radius = 6.5  # Nozzle radius (in mm), radius of the nozzle.

# Initialize the Nozzle object with the defined parameters.
nozzle = Nozzle(Nozzle_radius, Nozzle_speed, Nozzle_angle, stand_off_dist, device=device)

# === Simulation Parameters ===
# Simulation parameters include number of passes (n_p), time step (dt), final time (tf), and number of time intervals (n_t).
n_p = 5           # Total number of passes over the target mesh.
dt = 1e-2         # Time step for the simulation (in seconds).
tf = (init_length_y + 2 * nozzle.radius) / nozzle.speed # Final time when the nozzle finishes moving across the mesh.
n_t = int(tf / dt) + 1  # Total number of time intervals based on final time and time step.

# === Optimization Parameters ===
# Number of iterations for the optimization process.
n_iter = 100

# Set up the mesh optimization with the target mesh, initial mesh, and nozzle angle as optimization parameter.
optimization = MeshOptimization(target_mesh, init_mesh, nozzle.angle_beta)
optimization.initialize_optimizer(lr=1e-2)
losses = []
nozzle_angles = []
# === Optimization Loop ===
# Perform the optimization process for 'n_iter' iterations.
for _ in range(n_iter):
    # Reset the gradients before the backward pass.
    optimization.optimizer.zero_grad()
    
    # Initialize the mesh modification process, which will apply the nozzle's effects on the mesh.
    mesh_modification = MeshModification(nozzle, init_mesh, device=device)
    # Create a delta matrix used in the mesh modification (see documentation).
    mesh_modification.create_delta_matrix()

    # Simulate the nozzle passes over the mesh.
    for pass_idx in range(n_p):
        for m in range(n_t):
            # Update the nozzle position at each time step dt.
            nozzle.position = torch.stack([2 + (nozzle.stand_off_dist * torch.sin(nozzle.angle_beta)),
                                            torch.tensor((-1)**pass_idx * ((-nozzle.radius + pass_idx%2 * init_length_y.item()) + nozzle.speed * (m * dt)), device = device), 
                                            nozzle.stand_off_dist * torch.cos(nozzle.angle_beta)])
            
            # Update the normal vector for the nozzle based on its angle.
            nozzle.normal = torch.reshape(torch.stack((-torch.sin(nozzle.angle_beta),
                                                        torch.tensor(0.0, device=device), 
                                                        -torch.cos(nozzle.angle_beta))),[1,3])
            
            # Generate rays from the nozzle's position and normal, which will interact with the mesh.
            nozzle_ray_origins, nozzle_ray_directions = nozzle.generate_rays(nozzle.position, nozzle.normal)
            
            # Perform ray tracing to find intersections between the rays and the mesh.
            ray_traced_cells_indices, ray_traced_vertices = mesh_modification.moller_trumbore_multi_ray_trace(nozzle_ray_origins, nozzle_ray_directions, Multi_hit=False)
            
            # Modify the mesh vertices based on the intersections.
            mesh_modification.vertex_modification(ray_traced_vertices, ray_traced_cells_indices, mesh_modification.mesh_properties(), dt)
    
    # After the simulation is done, update the source mesh in the optimization object with the modified mesh.
    optimization.source_mesh = mesh_modification.mesh
    
    # Compute and store the loss
    loss = optimization.loss_computation(int(10 * number_of_vertices))
    losses.append(float(loss.detach().cpu()))
    nozzle_angles.append(180 / math.pi * float(nozzle.angle_beta.detach().cpu()))

    # Perform backpropagation to compute the gradients of the loss with respect to the optimization parameters.
    optimization.loss.backward()
    
    # Update the optimization parameters using the optimizer.
    optimization.optimizer.step()

    print(f'Iteration {_}, Nozzle angle = {nozzle.angle_beta * 180 / math.pi}, grad = {nozzle.angle_beta.grad}, loss value = {loss}')

# Plot and export the losses and optimization parameters.
optimization.plot(losses,'Plots/Nozze_angles_optimization_loss.png',label = 'Loss')
optimization.plot(nozzle_angles,'Plots/Nozze_angles_optimization.png',label = 'Angle Values')

# Export the final modified mesh to an .obj file (optional).
final_mesh_path = f'meshes/Final_mesh.obj'
mesh_modification.export_obj(final_mesh_path)