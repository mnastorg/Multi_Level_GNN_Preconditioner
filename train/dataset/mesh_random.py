### PACKAGES ###
import os 
import numpy as np
from math import *
import meshio 
import gmsh 
import subprocess
from fenics import *
################

class MyMesh : 

    def __init__(self, mesh_dict):
        
        self.folder_to_save = mesh_dict["folder_to_save"]
        self.name = mesh_dict["name"]

        self.radius = mesh_dict["radius"]
        self.nb_boundary_points = mesh_dict["nb_boundary_points"]
        self.hsize = mesh_dict["hsize"]

        self.dirichlet_tag = mesh_dict["dirichlet_tag"]

        self.seed = mesh_dict["seed"]

    def generate(self):

        gmsh.initialize()
        gmsh.model.add(self.name)

        np.random.seed(self.seed)
        
        alpha = np.linspace(0, 2*pi, self.nb_boundary_points)

        x_alpha = self.radius*np.cos(alpha)
        y_alpha = self.radius*np.sin(alpha)

        L = []
        for i in range(self.nb_boundary_points - 1):
            t = (1-0.75)*np.random.random()+0.75
            x_u = t*x_alpha[i]
            x_v = t*y_alpha[i]
            L.append(i+1)
            gmsh.model.geo.addPoint(x_u, x_v, 0, self.hsize, i+1)
        L.append(1)

        diri = []
        diri.append(gmsh.model.geo.addSpline(L, 1))
        
        curveloop = gmsh.model.geo.addCurveLoop([1], 1)

        surface = gmsh.model.geo.addPlaneSurface([1], 1)

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, diri, self.dirichlet_tag)
        gmsh.model.setPhysicalName(1, self.dirichlet_tag, "dirichlet")

        gmsh.model.addPhysicalGroup(2, [surface], 606)
        gmsh.model.setPhysicalName(2, 606, 'surface')
        
        # if we want rectangles : 
        # gmsh.model.mesh.setRecombine(2, 1)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

        # saving
        if not os.path.exists(self.folder_to_save) :
            os.mkdir(self.folder_to_save)
        
        path_msh = os.path.join(self.folder_to_save, self.name + ".msh")
        gmsh.write(path_msh)

        # use meshio to read and save to XDMF format
        path_xdmf = os.path.join(self.folder_to_save, self.name + ".xdmf")
        meshio_mesh = meshio.read(path_msh)
        meshio.write(path_xdmf, meshio_mesh)

        path_xml = os.path.join(self.folder_to_save, self.name + ".xml")
        subprocess.check_output('dolfin-convert ' + path_msh + " " + path_xml, shell = True)

        mesh = Mesh(path_xml)
        cd = MeshFunction('size_t', mesh , os.path.join(self.folder_to_save, self.name + "_physical_region.xml"))
        fd = MeshFunction('size_t', mesh, os.path.join(self.folder_to_save, self.name + "_facet_region.xml"))
        path_hdf5 = os.path.join(self.folder_to_save, self.name + "_fenics" + ".h5")
        hdf5 = HDF5File(mesh.mpi_comm(), path_hdf5, "w")
        hdf5.write(mesh, "/mesh")
        hdf5.write(cd, "/physical")
        hdf5.write(fd, "/facet")

        os.remove(path_xml)
        os.remove(os.path.join(self.folder_to_save, self.name + "_physical_region.xml"))
        os.remove(os.path.join(self.folder_to_save, self.name + "_facet_region.xml"))

        self.nb_nodes = meshio_mesh.points.shape[0]
        self.nb_cells = meshio_mesh.cells_dict['triangle'].shape[0]



         