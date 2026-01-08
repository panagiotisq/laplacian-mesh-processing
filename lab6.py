import scipy.linalg
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, Scene3D_, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D, Label3D
)

from matplotlib import colormaps as cm
import numpy as np
import scipy
from scipy import sparse
import time

WIDTH = 800
HEIGHT = 600

class Lab6(Scene3D_):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab6", output=True, n_sliders=2)
        self.reset_mesh()
        self.reset_sliders()
        self.printHelp()

    def reset_mesh(self):
        # Choose mesh
        # self.mesh = Mesh3D.create_bunny(color=Color.GRAY)
        self.mesh = Mesh3D("C:/Users/panag/Desktop/hmte/8o/3dcompgeo/labs/Lab6/resources/bunny_low.obj", color=Color.GRAY)
        #self.mesh = Mesh3D("C:/Users/panag/Desktop/hmte/8o/3dcompgeo/labs/Lab6/resources/dragon_low_low.obj", color=Color.GRAY)
        #self.mesh = Mesh3D("C:/Users/panag/Desktop/hmte/8o/3dcompgeo/labs/Lab6/resources/dolphin.obj", color=Color.GRAY)


        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        vertices = self.mesh.vertices
        vertices -= np.mean(vertices, axis=0)
        distanceSq = (vertices ** 2).sum(axis=-1)
        max_dist = np.sqrt(np.max(distanceSq))
        self.mesh.vertices = vertices / max_dist
        self.removeShape("mesh")
        self.addShape(self.mesh, "mesh")

        self.wireframe = LineSet3D.create_from_mesh(self.mesh)
        self.removeShape("wireframe")
        self.addShape(self.wireframe, "wireframe")
        self.show_wireframe = True

        self.eigenvectors = None

    def reset_sliders(self):
        self.set_slider_value(0, 0)
        self.set_slider_value(1, 0.1)

        
    @world_space
    def on_mouse_press(self, x, y, z, button, modifiers):
        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
            if np.isinf(z):
                return
            
            self.selected_vertex = find_closest_vertex(self.mesh, (x, y, z))

            vc = self.mesh.vertex_colors
            vc[self.selected_vertex] = (1, 0, 0)
            self.mesh.vertex_colors = vc

            self.updateShape("mesh", True)

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.R:
            self.reset_mesh()

        if symbol == Key.W:
            if self.show_wireframe:
                self.removeShape("wireframe")
                self.show_wireframe = False
            else:
                self.addShape(self.wireframe, "wireframe")
                self.show_wireframe = True
                
        if symbol == Key.A and hasattr(self, "selected_vertex"):
            adj = find_adjacent_vertices(self.mesh, self.selected_vertex)
            colors = self.mesh.vertex_colors
            for idx in adj:
                colors[idx] = (0, 0, 1)
            self.mesh.vertex_colors = colors
            self.updateShape("mesh")

        if symbol == Key.D and not modifiers & Key.MOD_CTRL and hasattr(self, "selected_vertex"):
            d = delta_coordinates_single(self.mesh, self.selected_vertex)
            self.print(d)

        if symbol == Key.D and modifiers & Key.MOD_CTRL:
            start = time.time()

            num_vertices = len(self.mesh.vertices)
            delta = []

            for idx in range(num_vertices):
                delta.append(delta_coordinates_single(self.mesh, idx))

            self.print(f"Took {(time.time() - start):.3f} seconds.")

            self.display_delta_coords(np.array(delta))

        if symbol == Key.L:
            start = time.time()
            d_coords = delta_coordinates(self.mesh)
            self.print(f"Took {(time.time() - start):.3f} seconds.")

            self.display_delta_coords(d_coords)

        if symbol == Key.S:
            start = time.time()
            d_coords = delta_coordinates_sparse(self.mesh)
            self.print(f"Took {(time.time() - start):.3f} seconds.")

            self.display_delta_coords(d_coords)

        if symbol == Key.E:
            _, vecs = eigendecomposition_full(self.mesh)
            self.eigenvectors = vecs
            self.display_eigenvector(vecs[:, self.eigenvector_idx])

        if symbol == Key.B:
            _, vecs = eigendecomposition_some(self.mesh, self.percent, "SM")  # keep the smallest self.percent eigenvectors
            vertices = self.mesh.vertices
            new_vertices = vecs @ vecs.T @ vertices  # reconstruct the mesh vertices
            self.mesh.vertices = new_vertices
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            # self.updateShape("wireframe")

        if symbol == Key.C:
            _, vecs1 = eigendecomposition_some(self.mesh, 0.01, "SM")  # keep the smallest 1% eigenvectors
            _, vecs2 = eigendecomposition_some(self.mesh, self.percent, "LM")  # keep the largest self.percent eigenvalues
            vecs = np.hstack((vecs1, vecs2))  # concatenate the eigenvectors (stack vecs1 on top of vecs2)
            #vecs = vecs2
            vertices = self.mesh.vertices
            new_vertices = vecs @ vecs.T @ vertices  # reconstruct the mesh vertices
            self.mesh.vertices = new_vertices
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            # self.updateShape("wireframe")

        if symbol == Key.Q:
            self.smooth_with_delta_cord(5)

        if symbol == Key.T:
            self.smooth_with_taubin(5)


        if symbol == Key.SLASH:
            self.printHelp()

    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            self.eigenvector_idx = int(value * (len(self.mesh.vertices) - 1))
            if self.eigenvectors is not None:
                self.display_eigenvector(self.eigenvectors[:, self.eigenvector_idx])

        if slider_id == 1:
            self.percent = value

    
    def printHelp(self):
        self.print("\
SHIFT+M1: Select vertex\n\
R: Reset mesh\n\
W: Toggle wireframe\n\
A: Adjacent vertices\n\
D: Delta coordinates single\n\
CTRL+D: Delta coordinates loop\n\
L: Delta coordinates laplacian\n\
S: Delta coordinates sparse\n\
E: Eigendecomposition\n\
B: Reconstruct from first {slider2}% eigenvetors\n\
C: Reconstruct from last {slider2}% eigenvetors\n\
Q: Smooth with delta coords\n\
T: Taubin Smoothing\n\
?: Show this list\n\n")

    def display_delta_coords(self, delta: np.ndarray):
        norm = np.sqrt((delta * delta).sum(-1))

        # linear interpolation
        norm = (norm - norm.min()) / (norm.max() - norm.min()) if norm.max() - norm.min() != 0 else np.zeros_like(norm)
        
        colormap = cm.get_cmap("plasma")
        colors = colormap(norm)
        self.mesh.vertex_colors = colors[:,:3]
        self.updateShape("mesh")

    def display_eigenvector(self, vec: np.ndarray):
        # linear interpolation
        vec = (vec - vec.min()) / (vec.max() - vec.min()) if vec.max() - vec.min() != 0 else np.zeros_like(vec)
        
        colormap = cm.get_cmap("plasma")
        colors = colormap(vec)
        self.mesh.vertex_colors = colors[:,:3]
        self.updateShape("mesh")

    def smooth_with_delta_cord(self, iter):
        
        for i in range(iter):
            vertices = self.mesh.vertices
            for vertex in vertices:
                idx = find_closest_vertex(self.mesh, vertex)
                vertex += 0.8*(-delta_coordinates_single(self.mesh, idx))
            self.mesh.vertices = vertices

        self.updateShape("mesh")

    def smooth_with_taubin(self, iter):
        
        for i in range(iter):
            vertices = self.mesh.vertices
            for vertex in vertices:
                idx = find_closest_vertex(self.mesh, vertex)
                vertex += 0.8*(-delta_coordinates_single(self.mesh, idx))
            self.mesh.vertices = vertices
            for vertex in vertices:
                idx = find_closest_vertex(self.mesh, vertex)
                vertex += 0.6*(delta_coordinates_single(self.mesh, idx))
            self.mesh.vertices = vertices
        self.updateShape("mesh")


def find_closest_vertex(mesh: Mesh3D, query: tuple) -> int:

    difference = (mesh.vertices - query)
    dist = (difference * difference).sum(axis=1)

    closest_vertex_index = np.argmin(dist)

    return closest_vertex_index

def find_adjacent_vertices(mesh: Mesh3D, idx: int) -> np.ndarray:
    
    vertices = mesh.vertices
    triangles = mesh.triangles

    adj = []

    # for t in triangles:
    #     if idx in t:
    #         # Save other indices in adj
    #         # idx_position = list(t).index(idx)
    #         idx_position = np.where(t == idx)[0]
    #         for i in range(3):
    #             if i == idx_position:
    #                 continue

    #             if t[i] not in adj:
    #                 adj.append(t[i])

    
    # for t in triangles:
    #     if idx in t:
    #         adj.extend(t[t != idx])


    # return np.unique(adj)


    adjacent_triangles = triangles[np.any(triangles == idx, axis=1)]
    adj.extend(adjacent_triangles[adjacent_triangles != idx])
    return np.unique(adj)


def delta_coordinates_single(mesh: Mesh3D, idx: int) -> np.ndarray:
    
    N_i = find_adjacent_vertices(mesh, idx)
    d_i = len(N_i)
    v_i = mesh.vertices[idx]
    v_j = mesh.vertices[N_i]

    return v_i - 1 / d_i * v_j.sum(axis=0)

def adjacency(mesh: Mesh3D) -> np.ndarray:

    num_vertices = len(mesh.vertices)
    triangles = mesh.triangles

    A = np.zeros((num_vertices, num_vertices), dtype=np.uint8)

    # for tri in triangles:
    #     v1, v2, v3 = tri
    #     A[v1, v2] = 1
    #     A[v2, v1] = 1
    #     A[v2, v3] = 1
    #     A[v3, v2] = 1
    #     A[v1, v3] = 1
    #     A[v3, v1] = 1

    # return A

    # === DIFFERENT SOLUTION ===
    # Extract all edges from triangles
    i = triangles[:, [0, 1, 2]].flatten()
    j = triangles[:, [1, 2, 0]].flatten()

    # Add edges in both directions
    A[i, j] = 1
    A[j, i] = 1

    return A

def adjacency_sparse(mesh: Mesh3D) -> sparse.csr_array:
    
    num_vertices = len(mesh.vertices)
    triangles = mesh.triangles

    A = sparse.lil_array((num_vertices, num_vertices), dtype=np.uint8)

    # for tri in triangles:
    #     v1, v2, v3 = tri
    #     A[v1, v2] = 1
    #     A[v2, v1] = 1
    #     A[v2, v3] = 1
    #     A[v3, v2] = 1
    #     A[v1, v3] = 1
    #     A[v3, v1] = 1

    # return A.tocsr()

    # === DIFFERENT SOLUTION ===
    # Extract all edges from triangles
    i = triangles[:, [0, 1, 2]].flatten()
    j = triangles[:, [1, 2, 0]].flatten()

    # Add edges in both directions
    A[i, j] = 1
    A[j, i] = 1

    return A.tocsr()

def degree(A: np.ndarray) -> np.ndarray:

    D = np.zeros_like(A, dtype=np.uint8)
    np.fill_diagonal(D, A.sum(axis=0))  

    return D

def degree_sparse(A: sparse.csr_array) -> sparse.csr_array:
    
    D = sparse.dia_array((A.sum(axis=1), 0), shape=A.shape, dtype=np.uint8)
    return D.tocsr()

def diagonal_inverse(mat: np.ndarray) -> np.ndarray:

    d = np.diag(mat)
    return np.diag(1 / d)

def diagonal_inverse_sparse(mat: sparse.csr_array) -> sparse.csr_array:
    
    d = mat.diagonal()
    return sparse.dia_array((1 / d, 0), shape=mat.shape, dtype=np.float64).tocsr()

def random_walk_laplacian(mesh: Mesh3D) -> np.ndarray:
    
    A = adjacency(mesh)
    D = degree(A)
    D_inv = diagonal_inverse(D)
    I = np.eye(*A.shape)

    L_RW = I - D_inv @ A

    return L_RW

def random_walk_laplacian_sparse(mesh: Mesh3D) -> sparse.csr_array:

    A = adjacency_sparse(mesh)
    D = degree_sparse(A)
    D_inv = diagonal_inverse_sparse(D)
    I = sparse.csr_array(sparse.eye(A.shape[0]))

    L_RW = I - D_inv @ A

    return L_RW

def delta_coordinates(mesh: Mesh3D) -> np.ndarray:
    
    X = mesh.vertices
    L = random_walk_laplacian(mesh)

    return L @ X

def delta_coordinates_sparse(mesh: Mesh3D) -> np.ndarray:

    X = mesh.vertices
    L = random_walk_laplacian_sparse(mesh)
    return L @ X

def graph_laplacian_sparse(mesh: Mesh3D) -> sparse.csr_array:
    
    A = adjacency_sparse(mesh).astype(np.int8)  # in order to change the dtype
    D = degree_sparse(A)

    return D - A

def eigendecomposition_full(mesh: Mesh3D) -> tuple[np.ndarray, np.ndarray]:

    L = graph_laplacian_sparse(mesh)
    return scipy.linalg.eigh(L._asfptype().toarray())

def eigendecomposition_some(mesh: Mesh3D, keep_percentage=0.1, which="SM") -> tuple[np.ndarray, np.ndarray]:
    
    L = graph_laplacian_sparse(mesh)
    k = int(L.shape[0] * keep_percentage)
    k = 1 if k == 0 else k

    return sparse.linalg.eigsh(L._asfptype().toarray(), k=k, which=which)

        
if __name__ == "__main__":
    app = Lab6()
    app.mainLoop()
