import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import math
import matplotlib.pyplot as plt

class MeshHandler:
    '''
    Class to handle 3D mesh loading and obtaining its dimensions.
    '''
    def __init__(self, obj_file, device):
        '''
        Initializes the MeshHandler by loading a 3D mesh from an OBJ file and storing its vertices and faces.
        
        Parameters:
        - obj_file: str
            Path to the OBJ file containing the 3D mesh.
        - device: torch.device
            Device (CPU or GPU) on which the mesh data will be stored.
        
        Upon initialization:
        - The vertices and faces of the mesh are extracted from the OBJ file.
        - 'self.faces_idx' is a tensor containing the vertex indices for each face, with shape (N_c, 3), where N_c is the number of faces. 
          Each face is described by the indices of the vertices that form it.
        - 'self.verts' is a tensor containing the 3D coordinates of the vertices, with shape (N_v, 3), where N_v is the number of vertices.
        - 'self.mesh' is a `Meshes` object representing the 3D mesh using the extracted vertices and faces.
        '''
        self.obj_file = obj_file
        self.device = device
        verts, faces, _ = load_obj(self.obj_file)
        self.faces_idx = faces.verts_idx.to(self.device)
        self.verts = verts.to(self.device)
        self.mesh = Meshes(verts=[self.verts], faces=[self.faces_idx])
    
    def mesh_dimensions(self):
        '''
        Computes the dimensions of the mesh along the x and y axes.

        Returns:
        - init_length_x: float
            The length of the mesh along the x-axis.
        - init_length_y: float
            The length of the mesh along the y-axis.
        '''
        verts = self.mesh.verts_packed()
        init_length_x = verts[-1,0] - verts[0,0] # x-distance
        init_length_y = verts[-1,1] - verts[0,1] # y-distance
        return  init_length_x, init_length_y
    
class Nozzle:
    '''
    Class to establish the main nozzle properties and generate the nozzle surface grid.
    It also defines two key functions for modeling Cold Spray deposition: deposition efficiency (DE) and deposition profile.
    '''
    def __init__(self, radius, speed, angle_beta, stand_off_dist, device):
        '''
        Initializes the Nozzle object by defining and storing its main properties.

        Parameters:
        - radius: float
            The radius of the nozzle.
        - speed: float
            The speed of the nozzle.
        - angle_beta: float
            The angle of orientation along a chosen axis (in degrees).
        - stand_off_dist: float
            The distance from the nozzle to the substrate surface.
        - device: torch.device
            The device (CPU or GPU) on which tensors for nozzle calculations will be stored.

        Notes:
        - The angle 'angle_beta' is converted from degrees to radians and stored as a torch tensor.
        - The nozzle's position and normal are initialized to `None` and should be manually assigned during simulations.
        '''  
        self.stand_off_dist = stand_off_dist
        self.angle_beta = torch.tensor((math.pi / 180) * angle_beta, dtype=torch.float32, device=device)
        self.radius = radius
        self.speed = speed
        self.device = device
        self.position = None
        self.normal = None 

    def generate_rays(self, origin, normal, coefficient = 3e-2):
        '''
        Computes the grid on the surface of the nozzle where each vertex is an origin for a ray.

        Parameters:
        - origin: torch.tensor 
            The center from which the grid is computed.
        - normal: torch.tensor
            The normal vector pointing out of the surface of the nozzle.
        - coefficient: float, optional (default=3e-2)
            Controls the resolution of the grid. Larger values create coarser grids. Should be a value between 0 and 1.

        Returns:
        - ray_origins: torch.tensor
            A tensor containing the coordinates of the ray origins.
        - ray_normals: torch.tensor
            A tensor containing the normal vectors at each ray origin.

        Notes:
        - The grid is computed as a square but filtered to only include points within the nozzle's radius (circular section).
        - Two orthogonal vectors on the nozzle's surface are computed based on the provided normal vector for the ray grid.
        '''
        # Step size in the x direction of nozzle grid
        dx_nozzle = coefficient * self.radius 
        # Step size in the y direction of nozzle grid
        dy_nozzle = coefficient * self.radius 

        # Defines the range of x and y values for the grid points
        x_range = torch.arange(- self.radius, self.radius + dx_nozzle, dx_nozzle, device=self.device)
        y_range = torch.arange(- self.radius, self.radius + dy_nozzle, dy_nozzle, device=self.device)
        x_points, y_points = torch.meshgrid(x_range, y_range, indexing='ij')

        x_nozzle_points = torch.flatten(x_points)
        y_nozzle_points = torch.flatten(y_points)

        # Maks the points in the square grid that lie inside the radius of the nozzle. 
        mask = x_nozzle_points**2 + y_nozzle_points**2 <= self.radius**2
        x_nozzle_points = x_nozzle_points[mask]
        y_nozzle_points = y_nozzle_points[mask]
        number_of_rays = len(x_nozzle_points)

        # Computes two orthogonal vectors on the surface of the nozzle based on the normal vector provided.
        v_parallel_1 = torch.tensor([-normal.squeeze()[2], 0, normal.squeeze()[0]], device=self.device)
        v_parallel_1 = v_parallel_1 / torch.linalg.norm(v_parallel_1)
        v_parallel_2 = torch.linalg.cross(v_parallel_1,normal.squeeze())

        # Computes the ray origins based on the grid points and the given origin. 
        ray_origins = origin * torch.ones((number_of_rays + 1,3),device=self.device)
        rays_radial_distance = torch.sqrt(x_nozzle_points**2 + y_nozzle_points**2).unsqueeze(1)
        rays_angles = torch.atan2(x_nozzle_points, y_nozzle_points).unsqueeze(1)
        v_parallel_component = v_parallel_1 * torch.cos(rays_angles) + v_parallel_2 * torch.sin(rays_angles)
        ray_origins[1:] = origin + rays_radial_distance * v_parallel_component

        # Assigns to each ray origin the provided normal.
        ray_directions = normal.tile((number_of_rays + 1,1))    

        return ray_origins, ray_directions
    
    def profile(self, r, coeff = [1.8826, 2.0316, 7.8639]):
        '''
        Computes the profile of the mass flux of particles, modeled as a super-Gaussian function.

        Parameters:
        - r: torch.tensor
            Radial distances from the nozzle's center.

        Returns:
        - torch.tensor
            The mass flux profile as a function of radial distance.

        Notes:
        - The profile is computed using optimal values of parameters s, n, and A.
        - The function can be modified to make the parameters adjustable for optimization processes.
        '''
        s, n, A = coeff
        # s, n, A = [1.8826, 2.0316, 7.8639] #Optimal Values
        return A * torch.exp( - ( (r / (2 * s) )**2 )**n) * (torch.abs(r) <= self.radius)
        
    def DE(self, phi, coeff = [8.9680, 0.6083]):
        '''
        Computes the deposition efficiency (DE), modeled as a sigmoid function.

        Parameters:
        - phi: torch.tensor
            Spray angles.

        Returns:
        - torch.tensor
            Deposition efficiency as a function of the spray angle.

        Notes:
        - The function uses optimal values for the sigmoid parameters a and b.
        - The parameters can be made adjustable for optimization processes.
        '''
        a,b = coeff
        # a, b = [8.9680, 0.6083] # Optimal Values
        return torch.special.expit(a * (phi - b))
    
class MeshModification:
    '''
    Class for modifying a 3D mesh based on nozzle properties and ray-tracing techniques. 
    The mesh is used to model material deposition in a cold spray simulation.
    '''
    def __init__(self, nozzle: Nozzle, mesh, device):
        '''
        Initializes the MeshModification object with the given nozzle, mesh, and computation device.

        Parameters:
        - nozzle: Nozzle
            A Nozzle object representing the properties of the nozzle used in the cold spray process.
        - mesh: Meshes
            A 3D mesh object that will be modified during the simulation the cold spray process.
        - device: torch.device
            The device (CPU or GPU) on which the mesh and calculations will be performed.
        
        Attributes:
        - delta_matrix: torch.tensor
            Matrix used in the modification of the mesh vertices. Initialized as None and created later by `create_delta_matrix()`.
        '''
        self.nozzle = nozzle
        self.device = device
        self.mesh = mesh
        self.delta_matrix = None

    def mesh_properties(self):
        '''
        Computes and returns key properties of the mesh, such as the face centers, normals, and areas.

        Returns:
        - faces_centers: torch.tensor
            The center point of each face in the mesh.
        - faces_normals: torch.tensor
            The normalized normal vectors for each face in the mesh.
        - faces_areas: torch.tensor
            The area of each face in the mesh.
        '''
        # Extract from the mesh the faces and the vertices.
        faces = self.mesh.faces_list()[0]  
        verts = self.mesh.verts_packed()
        # Extracts the first, second and third vertices of each mesh triangular face.
        v1,v2,v3 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]  
        # Computes the centroid of all the triangular faces.
        faces_centers = (v1 + v2 + v3) / 3.0
        # Extract from the mesh the normal unit vector to each traingular face.
        faces_normals = torch.nn.functional.normalize(self.mesh.faces_normals_packed())
        # Extract from the mesh the areas to each traingular face.
        faces_areas = self.mesh.faces_areas_packed()

        return faces_centers, faces_normals, faces_areas
    
    def create_delta_matrix(self):
        '''
        Creates the delta matrix (See documentation), which defines the relationship between mesh vertices and mesh faces.
        The matrix has size N_v x N_c, where N_c is the number of mesh faces and N_v the number of mesh vertices,
        if this matrix has is defined by its entries a_ij, we have that 

            a_ij = 1, if the i-th vertex belongs to the j-th face
            a_ij = 0, otherwise

        The delta matrix is used for modifying the mesh during the simulation.

        Notes:
        -'self.mesh.faces_packed()' is a tensor containing the 3 indices of each triangular faces, for every face in the mesh.
        Each face is composed of three vertices.
        '''
        # Extract the dimension for the delta matrix 
        N_c = len(self.mesh.faces_packed())
        N_v = len(self.mesh.verts_packed())
        delta_matrix = torch.zeros((N_v,N_c),device=self.device)

        # Row indices correspond to the mesh vertices indices. 
        delta_matrix_row_indices = self.mesh.faces_packed().flatten()
        # Column indices correspond to the mesh faces indices. It repeats every three values because each face is composed of 3 vertices.
        delta_matrix_col_indices = torch.arange(N_c, device=self.device).repeat_interleave(3) 
        # Change the value for the selected roe and column indices.
        delta_matrix[delta_matrix_row_indices, delta_matrix_col_indices] = 1
        self.delta_matrix = delta_matrix

    def moller_trumbore_multi_ray_trace(self, ray_origins, ray_directions, Multi_hit = True):
        '''
        Implements the Möller-Trumbore algorithm to perform ray-tracing for multiple rays and mesh faces.
        The method computes intersections between rays and triangular mesh faces.

        Parameters:
        - ray_origins: torch.tensor
            A tensor of shape (num_rays, 3) containing the starting points of the rays.
        - ray_directions: torch.tensor
            A tensor of shape (num_rays, 3) containing the direction vectors of the rays.
        - Multi_hit: bool, optional (default=True)
            If True, returns all intersections. If False, returns only the closest intersection to the starting point
            for each ray.

        Returns:
        - intersection_faces_idx: torch.tensor
            The indices of the faces that the rays intersect.
        - vertices_in_intersection_face: torch.tensor
            The vertices of the faces that are intersected by the rays.
        '''
        # Extract vertices and faces of the mesh
        vertices = self.mesh.verts_packed() # Shape: (num_vertices, 3)
        faces = self.mesh.faces_packed() # Shape: (num_faces, 3)

        # Extracts the first, second and third vertices respectively of all the faces.
        vertex_1 = vertices[faces[:, 0]]  # Shape: (num_faces, 3)
        vertex_2 = vertices[faces[:, 1]]  # Shape: (num_faces, 3)
        vertex_3 = vertices[faces[:, 2]]  # Shape: (num_faces, 3)

        # Computes two of the edges of the triangles having a commun vertex.
        edge_1 = vertex_2 - vertex_1       # Shape: (num_faces, 3)
        edge_2 = vertex_3 - vertex_1       # Shape: (num_faces, 3)

        # Computes de normal vector of all the faces.
        triangles_normals = torch.cross(edge_1, edge_2, dim=1)  # Shape: (num_faces, 3)

        # Expand ray_origins and ray_directions for broadcasting operations.
        #   ray_origins: originally (num_rays, 3), expand to (num_rays, 1, 3) to pair each ray origin with every face.
        #   ray_directions: originally (num_rays, 3), expand to (num_rays, 1, 3) similarly for ray directions.
        # This allows us to perform parallel computation of ray intersections with all triangles in a single operation.
        ray_origins_exp = ray_origins[:, None, :]  # Shape: (num_rays, 1, 3)
        ray_directions_exp = ray_directions[:, None, :]  # Shape: (num_rays, 1, 3)

        # Compute the determinant of the system for the Möller-Trumbore intersection algorithm. To ensure wich rays
        # are parallel to the triangular faces. We compute dot products of triangle normals (shape: (num_faces, 3)) 
        # with ray directions.
        dets = torch.matmul(triangles_normals[None, :, :], -ray_directions_exp.transpose(1, 2)).squeeze(-1)  # Shape: (num_rays, num_faces, 1)

        # Calculate the 'u' parameter of the intersection (barycentric coordinate), it satisfies 0<=u<=1.
        # 's' is the vector from the ray origin to the first vertex of each triangle.
        s = ray_origins_exp - vertex_1[None, :, :]  # Shape: (num_rays, num_faces, 3)
        s_cross_e2 = torch.cross(s, edge_2[None, :, :], dim=2)  # Shape: (num_rays, num_faces, 3)
        u = torch.sum(s_cross_e2 * (-ray_directions_exp), dim=2) / dets  # Shape: (num_rays, num_faces)

        # Mask out invalid 'u' values (outside the range 0-1, which are not in the triangle).
        u = u.masked_fill((u < 0) | (u > 1), float('nan'))

        # Calculate v parameter of the intersection (barycentric coordinate), it satisfies 0<=v<=1 and 0<= u+v <= 1.
        s_cross_e1 = torch.cross(edge_1[None, :, :], s, dim=2)  # Shape: (num_rays, num_faces, 3)
        v = torch.sum(s_cross_e1 * (-ray_directions_exp), dim=2) / dets  # Shape: (num_rays, num_faces)

        # Mask out invalid 'v' values and ensure that the sum of `u + v` is less than 1 (which is a requirement for valid intersections).
        v = v.masked_fill((torch.isnan(u)) | (v < 0) | (u + v > 1), float('nan'))

        # Calculate the 't' parameter, which represents the distance from the ray origin to the intersection point.
        t = torch.sum(s * triangles_normals[None, :, :], dim=2) / dets  # Shape: (num_rays, num_faces)
        # Mask out invalid `t` values, ensuring we only keep valid intersections.
        t = t.masked_fill((torch.isnan(u)) | (torch.isnan(v)), float('inf'))

        if Multi_hit == False:
            # If we only want the closest intersection, we find the smallest `t` for each ray.
            t_min = torch.min(t,dim=1,keepdim=True)[0]

            # We apply a mask to filter out invalid or non-intersecting rays (NaN or infinity in `t_min`).
            valid_mask = ~(torch.isnan(t_min) | torch.isinf(t_min))

            # Calculate the intersection points for the valid rays.
            intersection_points = ray_origins + t_min[valid_mask][:, None, None] * ray_directions  # Shape: (num_rays, 3)
        
            # Get the indices of the faces intersected by the rays.
            intersection_faces_idx = torch.min(t,dim=1,keepdim=True)[1][valid_mask]
            
            # Remove duplicate face indices by taking unique values.
            intersection_faces_idx = torch.unique(intersection_faces_idx)

            # Extract the vertices of the intersected faces.
            vertices_in_intersection_face = torch.unique(self.mesh.faces_packed()[intersection_faces_idx])

            return intersection_faces_idx, vertices_in_intersection_face

        # If 'Multi_hit == True', we find all valid intersections instead of just the closest.
        # Create a mask to filter out invalid intersections (NaN or infinity in `t` values).
        valid_mask = ~(torch.isnan(t) | torch.isinf(t))

        # Calculate the intersection points for all valid intersections.
        intersection_points = ray_origins_exp + t[:, :, None] * ray_directions_exp  # Shape: (num_rays, num_faces, 3)

        # Filter out invalid intersection points using the mask.
        intersection_points = intersection_points[valid_mask]  # Flattened valid points

        # Get the indices of the faces intersected by the rays.
        intersection_faces_idx = torch.where(valid_mask)[1]  #[Indices of ray intersecions, Indices of face intersections]
        
        # Remove duplicate face indices by taking unique values.
        intersection_faces_idx = torch.unique(intersection_faces_idx)
        
        # Extract the vertices of the intersected faces.
        vertices_in_intersection_face = torch.unique(self.mesh.faces_packed()[intersection_faces_idx])
        
        return intersection_faces_idx, vertices_in_intersection_face

    def vertex_modification(self, selected_verts, selected_faces_idx, mesh_properties, dt):
        '''
        Modifies the vertices of the mesh based on the intersection points obtained by the Möller-Trumbore algorithm 
        and the cold spray deposition model.

        Parameters:
        - selected_verts: torch.tensor
            Indices of the vertices selected for modification.
        - selected_faces_idx: torch.tensor
            Indices of the faces selected for modification.
        - mesh_properties: tuple
            A tuple containing the mesh properties (faces_centers, faces_normals, faces_areas), computed with mesh_properties().
        - dt: float
            Time step for the simulation.
        '''
        faces_centers, faces_normals, faces_areas = mesh_properties

        delta_matrix_rt = self.delta_matrix[selected_verts][:,selected_faces_idx]
        number_of_intersection_cells = len(selected_faces_idx)
        number_of_intersection_vertices = len(selected_verts)

        # ----MESH MODIFICATION----
        Nozzle_to_Centers = faces_centers[selected_faces_idx] - self.nozzle.position
        distance_centers_to_nozzle_axis = torch.linalg.norm(torch.linalg.cross(-self.nozzle.normal, Nozzle_to_Centers), dim = 1) / torch.linalg.norm(self.nozzle.normal)
        spray_angle = math.pi / 2 - torch.atan2(torch.linalg.norm(torch.linalg.cross(-self.nozzle.normal, faces_normals[selected_faces_idx]), dim = 1),
                                                                (faces_normals[selected_faces_idx] * (-self.nozzle.normal)).sum(axis = 1))
        
        dz_center = self.nozzle.DE(spray_angle) * self.nozzle.profile(distance_centers_to_nozzle_axis) * dt

        Weigthed_areas = torch.reshape(faces_areas[selected_faces_idx] * torch.abs(torch.sin(spray_angle)),(1,number_of_intersection_cells))
        Weigthed_areas_with_delta = delta_matrix_rt @ torch.transpose(Weigthed_areas,0,1)
        M_matrix = delta_matrix_rt * Weigthed_areas / Weigthed_areas_with_delta
        dz_vertices = torch.zeros((1,len(self.mesh.verts_packed())),device=self.device)
        dz_vertices[0,selected_verts] = torch.reshape(M_matrix @ torch.reshape(dz_center,(number_of_intersection_cells,1)),(1,number_of_intersection_vertices))

        self.mesh = self.mesh.offset_verts(- dz_vertices.T * self.nozzle.normal / torch.linalg.norm(self.nozzle.normal))
  
    def export_obj(self, file_path):
        '''
        Exports the modified mesh to an OBJ file.

        Parameters:
        - file_path: str
            The path to the file where the OBJ should be saved.
        '''
        final_verts, final_faces = self.mesh.get_mesh_verts_faces(0)
        save_obj(f'{file_path}',final_verts,final_faces)

class MeshOptimization:
    '''
    A class that handles the optimization process for aligning two 3D meshes, using the Chamfer distance as the loss function.
    The source mesh is transformed to align with the target mesh, and optimization parameters are adjusted accordingly.
    '''
    def __init__(self, target_mesh, source_mesh, optimization_parameters):
        '''
        Initializes the MeshOptimization class with the target mesh, source mesh, and optimization parameters.

        Parameters:
        - target_mesh: Meshes
            The target 3D mesh that the source mesh will be aligned to.
        - source_mesh: Meshes
            The source 3D mesh that will undergo transformation to match the target mesh.
        - optimization_parameters: torch.tensor
            A tensor of parameters (such as transformation parameters) that will be optimized during the process.

        Notes: 
        - loss: torch.tensor
            The loss value computed during the optimization process, typically the Chamfer distance between meshes. To be 
            computed after one simulation step has passed with loss_computation().
        - optimizer: torch.optim.Optimizer
            The optimizer used to adjust the optimization parameters during training. The user has to manually set it up with 
            initialize_optimizer() with the appropiate optimization parameters.    
        '''
        self.target_mesh = target_mesh
        self.source_mesh = source_mesh
        self.parameters = optimization_parameters
        self.loss = None
        self.optimizer = None

    def initialize_optimizer(self, lr=1e-2):
        '''
        Initializes the optimizer for updating the optimization parameters.

        Parameters:
        - lr: float, optional (default=1e-2)
            The learning rate for the optimizer. Controls the step size of gradient descent algorithm during the parameter update.
        
        Notes:
        - The optimizer used here is Adam, a commonly used optimization algorithm in deep learning.
        - The 'requires_grad_()' method is used to ensure that the optimization parameters are differentiable and can be updated by the optimizer.
        '''
        self.optimizer = torch.optim.Adam([self.parameters.requires_grad_()], lr=lr) 

    def loss_computation(self, number_of_sampling_points, chamfer_weigth = 1):
        '''
        Computes the loss function between the target and source meshes using the Chamfer distance.

        Parameters:
        - number_of_sampling_points: int
            The number of points sampled from each mesh to compute the Chamfer distance. A higher number provides a more accurate distance measure.
        - chamfer_weigth: float, optional (default=1)
            The weighting factor for the Chamfer distance in the loss function. Can be adjusted to scale the contribution of the Chamfer loss 
            when other loss contributions are present.

        Returns:
        - loss: torch.tensor
            The computed loss, which is the weighted Chamfer distance between the sampled points of the target and source meshes.
        
        Notes:
        - The Chamfer distance measures the similarity between two point clouds by computing the nearest neighbor distances between points.
        '''
        sample_trg = sample_points_from_meshes(self.target_mesh, number_of_sampling_points)
        sample_src = sample_points_from_meshes(self.source_mesh, number_of_sampling_points)
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        self.loss = loss_chamfer * chamfer_weigth 
        return self.loss
    
    def plot(self, values, file_path,label, domain=None):
        '''
        Plots the given values (e.g., loss or other metrics) and saves the plot to a file.

        Parameters:
        - values: list or array
            The list or array of values to plot (e.g., loss over iterations).
        - file_path: str
            The file path where the plot image will be saved.
        - label: str
            The label for the plot (usually describing the values being plotted).
        - domain: list or array, optional (default=None)
            The domain (x-axis values) for the plot. If not provided, the values will be plotted against their index.
        
        Notes:
        - The function uses matplotlib to create and save the plot.
        - If the `domain` is provided, it plots 'values' against the 'domain'. Otherwise, it uses the index of 'values' as the x-axis.
        '''
        if domain is not None: 
            fig = plt.figure()
            ax = fig.add_subplot()    
            ax.plot(domain, values, color="red", label = label)        
            ax.legend(loc="upper left")
            plt.savefig(file_path)
        else:
            fig = plt.figure()
            ax = fig.add_subplot()    
            ax.plot(values, color="red", label = label)        
            ax.legend(loc="upper left")
            plt.savefig(file_path)


    


