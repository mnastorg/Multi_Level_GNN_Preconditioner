##########################################################################
################################ PACKAGES ################################
##########################################################################

import os
import subprocess
import shutil
import argparse

from fenics import *
import numpy as np
from math import *
import meshio
import gmsh

##########################################################################
##########################################################################
##########################################################################

class MyMesh : 

    def __init__(self, mesh_dict):
        
        self.folder_to_save = mesh_dict["folder_to_save"]
        self.name = mesh_dict["name"]

        self.hsize = mesh_dict["hsize"]

        self.radius = mesh_dict["radius"]
        
        self.dirichlet_tag = mesh_dict["dirichlet_tag"]

    def generate(self):

        gmsh.initialize()
        gmsh.model.add(self.name)

        # CAR BOTTOM PART 
        gmsh.model.geo.addPoint(-1.0*self.radius, -0.55*self.radius, 0, self.hsize, 1)
        gmsh.model.geo.addPoint(-0.8*self.radius, -0.55*self.radius, 0, self.hsize, 2)
        gmsh.model.geo.addPoint(-0.8*self.radius, -0.1*self.radius, 0, self.hsize, 3)
        gmsh.model.geo.addPoint(-0.55*self.radius, -0.138*self.radius, 0, self.hsize, 4)
        gmsh.model.geo.addPoint(-0.55*self.radius, -0.3*self.radius, 0, self.hsize, 5)
        gmsh.model.geo.addPoint(-0.65*self.radius, -0.3*self.radius, 0, self.hsize, 6)
        gmsh.model.geo.addPoint(-0.65*self.radius, -0.45*self.radius, 0, self.hsize, 7)
        gmsh.model.geo.addPoint(-0.35*self.radius, -0.45*self.radius, 0, self.hsize, 8)
        gmsh.model.geo.addPoint(-0.35*self.radius, -0.3*self.radius, 0, self.hsize, 9)
        gmsh.model.geo.addPoint(-0.45*self.radius, -0.3*self.radius, 0, self.hsize, 10)
        gmsh.model.geo.addPoint(-0.45*self.radius, -0.154*self.radius, 0, self.hsize, 11)
            
        gmsh.model.geo.addPoint(-0.15*self.radius, -0.2*self.radius, 0, self.hsize, 12)

        gmsh.model.geo.addPoint(-0.1*self.radius, -0.25*self.radius, 0, self.hsize, 13)
        gmsh.model.geo.addPoint(0.0*self.radius, -0.32*self.radius, 0, self.hsize, 14)
        gmsh.model.geo.addPoint(0.1*self.radius, -0.32*self.radius, 0, self.hsize, 15)
        gmsh.model.geo.addPoint(0.2*self.radius, -0.3*self.radius, 0, self.hsize, 16)
        gmsh.model.geo.addPoint(0.25*self.radius, -0.25*self.radius, 0, self.hsize, 17)
        gmsh.model.geo.addPoint(0.3*self.radius, -0.2*self.radius, 0, self.hsize, 18)

        gmsh.model.geo.addPoint(0.5*self.radius, -0.1*self.radius, 0, self.hsize, 19)
        
        gmsh.model.geo.addPoint(0.6*self.radius, -0.1*self.radius, 0, self.hsize, 20)
        gmsh.model.geo.addPoint(0.6*self.radius, -0.3*self.radius, 0, self.hsize, 21)
        gmsh.model.geo.addPoint(0.5*self.radius, -0.3*self.radius, 0, self.hsize, 22)
        gmsh.model.geo.addPoint(0.5*self.radius, -0.45*self.radius, 0, self.hsize, 23)
        gmsh.model.geo.addPoint(0.8*self.radius, -0.45*self.radius, 0, self.hsize, 24)
        gmsh.model.geo.addPoint(0.8*self.radius, -0.3*self.radius, 0, self.hsize, 25)
        gmsh.model.geo.addPoint(0.7*self.radius, -0.3*self.radius, 0, self.hsize, 26)
        gmsh.model.geo.addPoint(0.7*self.radius, -0.1*self.radius, 0, self.hsize, 27)
        gmsh.model.geo.addPoint(0.85*self.radius, -0.1*self.radius, 0, self.hsize, 28)
        gmsh.model.geo.addPoint(0.85*self.radius, -0.25*self.radius, 0, self.hsize, 29)
        gmsh.model.geo.addPoint(1.0*self.radius, -0.25*self.radius, 0, self.hsize, 30)


        # CAR TOP PART 
        gmsh.model.geo.addPoint(-1.0*self.radius, 0.55*self.radius, 0, self.hsize, 60)
        gmsh.model.geo.addPoint(-0.8*self.radius, 0.55*self.radius, 0, self.hsize, 59)
        gmsh.model.geo.addPoint(-0.8*self.radius, 0.1*self.radius, 0, self.hsize, 58)
        gmsh.model.geo.addPoint(-0.55*self.radius, 0.138*self.radius, 0, self.hsize, 57)
        gmsh.model.geo.addPoint(-0.55*self.radius, 0.3*self.radius, 0, self.hsize, 56)
        gmsh.model.geo.addPoint(-0.65*self.radius, 0.3*self.radius, 0, self.hsize, 55)
        gmsh.model.geo.addPoint(-0.65*self.radius, 0.45*self.radius, 0, self.hsize, 54)
        gmsh.model.geo.addPoint(-0.35*self.radius, 0.45*self.radius, 0, self.hsize, 53)
        gmsh.model.geo.addPoint(-0.35*self.radius, 0.3*self.radius, 0, self.hsize, 52)
        gmsh.model.geo.addPoint(-0.45*self.radius, 0.3*self.radius, 0, self.hsize, 51)
        gmsh.model.geo.addPoint(-0.45*self.radius, 0.154*self.radius, 0, self.hsize, 50)
            
        gmsh.model.geo.addPoint(-0.15*self.radius, 0.2*self.radius, 0, self.hsize, 49)

        gmsh.model.geo.addPoint(-0.1*self.radius, 0.25*self.radius, 0, self.hsize, 48)
        gmsh.model.geo.addPoint(0.0*self.radius, 0.32*self.radius, 0, self.hsize, 47)
        gmsh.model.geo.addPoint(0.1*self.radius, 0.32*self.radius, 0, self.hsize, 46)
        gmsh.model.geo.addPoint(0.2*self.radius, 0.3*self.radius, 0, self.hsize, 45)
        gmsh.model.geo.addPoint(0.25*self.radius, 0.25*self.radius, 0, self.hsize, 44)
        gmsh.model.geo.addPoint(0.3*self.radius, 0.2*self.radius, 0, self.hsize, 43)

        gmsh.model.geo.addPoint(0.5*self.radius, 0.1*self.radius, 0, self.hsize, 42)
        
        gmsh.model.geo.addPoint(0.6*self.radius, 0.1*self.radius, 0, self.hsize, 41)
        gmsh.model.geo.addPoint(0.6*self.radius, 0.3*self.radius, 0, self.hsize, 40)
        gmsh.model.geo.addPoint(0.5*self.radius, 0.3*self.radius, 0, self.hsize, 39)
        gmsh.model.geo.addPoint(0.5*self.radius, 0.45*self.radius, 0, self.hsize, 38)
        gmsh.model.geo.addPoint(0.8*self.radius, 0.45*self.radius, 0, self.hsize, 37)
        gmsh.model.geo.addPoint(0.8*self.radius, 0.3*self.radius, 0, self.hsize, 36)
        gmsh.model.geo.addPoint(0.7*self.radius, 0.3*self.radius, 0, self.hsize, 35)
        gmsh.model.geo.addPoint(0.7*self.radius, 0.1*self.radius, 0, self.hsize, 34)
        gmsh.model.geo.addPoint(0.85*self.radius, 0.1*self.radius, 0, self.hsize, 33)
        gmsh.model.geo.addPoint(0.85*self.radius, 0.25*self.radius, 0, self.hsize, 32)
        gmsh.model.geo.addPoint(1.0*self.radius, 0.25*self.radius, 0, self.hsize, 31)

        # LINES BOTTOM 
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 7, 6)
        gmsh.model.geo.addLine(7, 8, 7)
        gmsh.model.geo.addLine(8, 9, 8)
        gmsh.model.geo.addLine(9, 10, 9)
        gmsh.model.geo.addLine(10, 11, 10)
        gmsh.model.geo.addLine(11, 12, 11)

        gmsh.model.geo.addSpline([12,13,14,15,16,17,18,19], 12)

        gmsh.model.geo.addLine(19, 20, 13)
        gmsh.model.geo.addLine(20, 21, 14)
        gmsh.model.geo.addLine(21, 22, 15)
        gmsh.model.geo.addLine(22, 23, 16)
        gmsh.model.geo.addLine(23, 24, 17)
        gmsh.model.geo.addLine(24, 25, 18)
        gmsh.model.geo.addLine(25, 26, 19)
        gmsh.model.geo.addLine(26, 27, 20)
        gmsh.model.geo.addLine(27, 28, 21)
        gmsh.model.geo.addLine(28, 29, 22)
        gmsh.model.geo.addLine(29, 30, 23)

        gmsh.model.geo.addLine(30, 31, 24)

        # LINES TOP 
        gmsh.model.geo.addLine(31, 32, 25)
        gmsh.model.geo.addLine(32, 33, 26)
        gmsh.model.geo.addLine(33, 34, 27)
        gmsh.model.geo.addLine(34, 35, 28)
        gmsh.model.geo.addLine(35, 36, 29)
        gmsh.model.geo.addLine(36, 37, 30)
        gmsh.model.geo.addLine(37, 38, 31)
        gmsh.model.geo.addLine(38, 39, 32)
        gmsh.model.geo.addLine(39, 40, 33)
        gmsh.model.geo.addLine(40, 41, 34)
        gmsh.model.geo.addLine(41, 42, 35)

        gmsh.model.geo.addSpline([42,43,44,45,46,47,48,49], 36)

        gmsh.model.geo.addLine(49, 50, 37)
        gmsh.model.geo.addLine(50, 51, 38)
        gmsh.model.geo.addLine(51, 52, 39)
        gmsh.model.geo.addLine(52, 53, 40)
        gmsh.model.geo.addLine(53, 54, 41)
        gmsh.model.geo.addLine(54, 55, 42)
        gmsh.model.geo.addLine(55, 56, 43)
        gmsh.model.geo.addLine(56, 57, 44)
        gmsh.model.geo.addLine(57, 58, 45)
        gmsh.model.geo.addLine(58, 59, 46)
        gmsh.model.geo.addLine(59, 60, 47)

        gmsh.model.geo.addLine(60, 1, 48)
        
        L = np.arange(1,49)
        gmsh.model.geo.addCurveLoop(L, 1)

        ## COCKPIT 
        c_x = -0.1*self.radius 
        c_y = 0.0*self.radius
        R = 0.1*self.radius
        gmsh.model.geo.addPoint(c_x, c_y, 0, self.hsize, 61)
        gmsh.model.geo.addPoint(c_x, c_y + R, 0, self.hsize, 62)
        gmsh.model.geo.addPoint(c_x, c_y - R, 0, self.hsize, 63)
        gmsh.model.geo.addCircleArc(63, 61, 62, 49)
        gmsh.model.geo.addLine(62, 63, 50)

        gmsh.model.geo.addCurveLoop([49, 50], 2)

        ## BOTTOM STRIE 1
        x_A = -0.95*self.radius
        y_A = -0.35*self.radius
        tx = 0.1*self.radius
        ty = 0.05*self.radius
        gmsh.model.geo.addPoint(x_A, y_A, 0, self.hsize, 64)
        gmsh.model.geo.addPoint(x_A + tx, y_A - ty, 0, self.hsize, 65)
        gmsh.model.geo.addPoint(x_A + tx, y_A, 0, self.hsize, 66)
        gmsh.model.geo.addPoint(x_A, y_A + ty, 0, self.hsize, 67)
        gmsh.model.geo.addLine(64, 65, 51)
        gmsh.model.geo.addLine(65, 66, 52)
        gmsh.model.geo.addLine(66, 67, 53)
        gmsh.model.geo.addLine(67, 64, 54)

        gmsh.model.geo.addCurveLoop([51, 52, 53, 54], 3)

        ## BOTTOM STRIE 2
        x_A = -0.95*self.radius
        y_A = -0.15*self.radius
        tx = 0.1*self.radius
        ty = 0.05*self.radius
        gmsh.model.geo.addPoint(x_A, y_A, 0, self.hsize, 68)
        gmsh.model.geo.addPoint(x_A + tx, y_A - ty, 0, self.hsize, 69)
        gmsh.model.geo.addPoint(x_A + tx, y_A, 0, self.hsize, 70)
        gmsh.model.geo.addPoint(x_A, y_A + ty, 0, self.hsize, 71)
        gmsh.model.geo.addLine(68, 69, 55)
        gmsh.model.geo.addLine(69, 70, 56)
        gmsh.model.geo.addLine(70, 71, 57)
        gmsh.model.geo.addLine(71, 68, 58)

        gmsh.model.geo.addCurveLoop([55, 56, 57, 58], 4)

        ## TOP STRIE 1
        x_A = -0.95*self.radius
        y_A = 0.15*self.radius
        tx = 0.1*self.radius
        ty = 0.05*self.radius
        gmsh.model.geo.addPoint(x_A, y_A, 0, self.hsize, 72)
        gmsh.model.geo.addPoint(x_A + tx, y_A + ty, 0, self.hsize, 73)
        gmsh.model.geo.addPoint(x_A + tx, y_A, 0, self.hsize, 74)
        gmsh.model.geo.addPoint(x_A, y_A - ty, 0, self.hsize, 75)
        gmsh.model.geo.addLine(72, 73, 59)
        gmsh.model.geo.addLine(73, 74, 60)
        gmsh.model.geo.addLine(74, 75, 61)
        gmsh.model.geo.addLine(75, 72, 62)

        gmsh.model.geo.addCurveLoop([59, 60, 61, 62], 5)

        ## TOP STRIE 2
        x_A = -0.95*self.radius
        y_A = 0.35*self.radius
        tx = 0.1*self.radius
        ty = 0.05*self.radius
        gmsh.model.geo.addPoint(x_A, y_A, 0, self.hsize, 76)
        gmsh.model.geo.addPoint(x_A + tx, y_A + ty, 0, self.hsize, 77)
        gmsh.model.geo.addPoint(x_A + tx, y_A, 0, self.hsize, 78)
        gmsh.model.geo.addPoint(x_A, y_A - ty, 0, self.hsize, 79)
        gmsh.model.geo.addLine(76, 77, 63)
        gmsh.model.geo.addLine(77, 78, 64)
        gmsh.model.geo.addLine(78, 79, 65)
        gmsh.model.geo.addLine(79, 76, 66)

        gmsh.model.geo.addCurveLoop([63, 64, 65, 66], 6)

        ## BACK WING 
        gmsh.model.geo.addPoint(0.9*self.radius, -0.1*self.radius, 0, self.hsize, 80)
        gmsh.model.geo.addPoint(0.95*self.radius, -0.1*self.radius, 0, self.hsize, 81)
        gmsh.model.geo.addPoint(0.95*self.radius, 0.1*self.radius, 0, self.hsize, 82)
        gmsh.model.geo.addPoint(0.9*self.radius, 0.1*self.radius, 0, self.hsize, 83)
        gmsh.model.geo.addLine(80, 81, 67)
        gmsh.model.geo.addLine(81, 82, 68)
        gmsh.model.geo.addLine(82, 83, 69)
        gmsh.model.geo.addLine(83, 80, 70)

        gmsh.model.geo.addCurveLoop([67, 68, 69, 70], 7)

        # FULL DOMAIN
        gmsh.model.geo.addPlaneSurface([1, 2, 3, 4, 5, 6, 7], 1)

        gmsh.model.geo.synchronize()
        
        L2 = np.arange(49,71)

        # PHYSICAL GROUPS FOR BOUNDARY CONDITIONS
        gmsh.model.addPhysicalGroup(1, np.arange(1,71), self.dirichlet_tag)
        gmsh.model.setPhysicalName(1, self.dirichlet_tag, "dirichlet")

        gmsh.model.addPhysicalGroup(2, [1], 606)
        gmsh.model.setPhysicalName(2, 606, 'Surface')
        
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