import math
from typing import List

from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape

# Core
from Core.Topology import Topology as coreTopology
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Shell import Shell as coreShell
from Core.Cluster import Cluster as coreCluster
from Core.Cell import Cell as coreCell
from Core.CellComplex import CellComplex as coreCellComplex
from Core.Aperture import Aperture as coreAperture
from Core.Context import Context as coreContext

from Core.Dictionary import Dictionary as coreDictionary
from Core.Utilities.TopologicUtilities import VertexUtility, EdgeUtility, FaceUtility

from Core.TopologyConstants import TopologyTypes

# Wrapper
from Wrapper.Vertex import Vertex
from Wrapper.Vector import Vector
from Wrapper.Cluster import Cluster
from Wrapper.Topology import Topology

class Shell(Topology):
    @staticmethod
    def ByFaces(faces: list, tolerance: float = 0.0001) -> coreShell:
        """
        Creates a shell from the input list of faces.

        Parameters
        ----------
        faces : list
            The input list of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreShell
            The created Shell.

        """
        if not isinstance(faces, list):
            return None
        faceList = [x for x in faces if isinstance(x, coreFace)]
        if len(faceList) < 1:
            return None
        shell = coreShell.by_faces(faceList, tolerance, True)
        if not shell:
            result = faceList[0]
            remainder = faceList[1:]
            cluster = coreCluster.by_topologies(remainder, False)
            result = result.merge(cluster, False)
            if result.get_topology_type() > TopologyTypes.SHELL:
                returnShells = result.shells(None)
                return returnShells
            else:
                return None
        else:
            return shell

    @staticmethod
    def ByFacesCluster(cluster: coreCluster, tolerance: float = 0.0001) -> coreShell:
        """
        Creates a shell from the input cluster of faces.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        coreShell
            The created shell.

        """
        if not isinstance(cluster, coreCluster):
            return None
        faces = cluster.faces()
        return Shell.ByFaces(faces, tolerance=tolerance)

    @staticmethod
    def ByWires(wires: list, triangulate: bool = True, tolerance: float = 0.0001) -> coreShell:
        """
        Creates a shell by lofting through the input wires
        Parameters
        ----------
        wires : list
            The input list of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        coreShell
            The creates shell.
        """
        from Wrapper.Edge import Edge
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        if not isinstance(wires, list):
            return None
        wireList = [x for x in wires if isinstance(x, coreWire)]
        faces = []
        for i in range(len(wireList)-1):
            wire1 = wireList[i]
            wire2 = wireList[i+1]

            if wire1.get_type().value < TopologyTypes.EDGE.value or wire2.get_type().value < TopologyTypes.EDGE.value:
                return None
            if wire1.get_type().value == TopologyTypes.EDGE.value:
                w1_edges = [wire1]
            else:
                w1_edges = []
                w1_edges = wire1.edges()
            if wire2.get_type().value == TopologyTypes.EDGE.value:
                w2_edges = [wire2]
            else:
                w2_edges = []
                w2_edges = wire2.edges()
            if len(w1_edges) != len(w2_edges):
                return None
            if triangulate == True:
                for j in range (len(w1_edges)):
                    e1 = w1_edges[j]
                    e2 = w2_edges[j]
                    e3 = None
                    e4 = None
                    try:
                        e3 = Edge.ByVertices([e1.start_vertex(), e2.start_vertex()])
                    except:
                        e4 = Edge.ByVertices([e1.end_vertex(), e2.end_vertex()])
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e4])))
                    try:
                        e4 = Edge.ByVertices([e1.end_vertex(), e2.end_vertex()])
                    except:
                        e3 = Edge.ByVertices([e1.start_vertex(), e2.start_vertex()])
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3])))
                    if e3 and e4:
                        e5 = Edge.ByVertices([e1.start_vertex(), e2.end_vertex()])
                        faces.append(Face.ByWire(Wire.ByEdges([e1, e5, e4]))) # The resulting wire consists of a single edge. Explanation: the edges making up the wire are coincident!
                        faces.append(Face.ByWire(Wire.ByEdges([e2, e5, e3])))
            else:
                e1 = w1_edges[0]
                e2 = w1_edges[1]
                e3 = w1_edges[2]
                e4 = w1_edges[3]

                faces.append(Face.ByWire(Wire.ByEdges([e1, e2, e3, e4])))

                # for j in range (len(w1_edges)):
                #     e1 = w1_edges[j]
                #     e2 = w2_edges[j]
                #     e3 = None
                #     e4 = None
                #     try:
                #         e3 = Edge.ByVertices([e1.start_vertex(), e2.start_vertex()])
                #     except:
                #         try:
                #             e4 = Edge.ByVertices([e1.end_vertex(), e2.end_vertex()])
                #         except:
                #             pass
                #     try:
                #         e4 = Edge.ByVertices([e1.end_vertex(), e2.end_vertex()])
                #     except:
                #         try:
                #             e3 = Edge.ByVertices([e1.start_vertex(), e2.start_vertex()])
                #         except:
                #             pass
                #     if e3 and e4:
                #         try:
                #             faces.append(Face.ByWire(coreWire.by_edges([e1, e4, e2, e3])))
                #         except:
                #             faces.append(Face.ByWire(coreWire.by_edges([e1, e3, e2, e4])))
                #     elif e3:
                #             faces.append(Face.ByWire(coreWire.by_edges([e1, e3, e2])))
                #     elif e4:
                #             faces.append(Face.ByWire(coreWire.by_edges([e1, e4, e2])))
        return Shell.ByFaces(faces, tolerance)

    @staticmethod
    def ByWiresCluster(cluster: coreCluster, triangulate: bool = True, tolerance: float = 0.0001) -> coreShell:
        """
        Creates a shell by lofting through the input cluster of wires

        Parameters
        ----------
        wires : coreCluster
            The input cluster of wires.
        triangulate : bool , optional
            If set to True, the faces will be triangulated. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreShell
            The creates shell.

        """
        from Wrapper.Cluster import Cluster
        if not cluster:
            return None
        if not isinstance(cluster, coreCluster):
            return None
        wires = Cluster.Wires(cluster)
        return Shell.ByWires(wires, triangulate=triangulate, tolerance=tolerance)

    @staticmethod
    def Circle(origin: coreVertex = None, radius: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreShell:
        """
        Creates a circle.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The  radius of the circle. The default is 0.5.
        sides : int , optional
            The number of sides of the circle. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreShell
            The created circle.
        """
        return Shell.Pie(origin=origin, radiusA=radius, radiusB=0, sides=sides, rings=1, fromAngle=fromAngle, toAngle=toAngle, direction=direction, placement=placement, tolerance=tolerance)

    @staticmethod
    def Delaunay(vertices: list, face: coreFace = None) -> coreShell:
        """
        Returns a delaunay partitioning of the input vertices. The vertices must be coplanar. See https://en.wikipedia.org/wiki/Delaunay_triangulation.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : coreFace , optional
            The input face. If specified, the delaunay triangulation is clipped to the face.

        Returns
        -------
        shell
            A shell representing the delaunay triangulation of the input vertices.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        from random import sample
        import sys
        import subprocess

        try:
            from scipy.spatial import Delaunay
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'scipy', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                from scipy.spatial import Delaunay
            except:
                print("Shell.Delaunay - ERROR: Could not import scipy. Returning None.")
                return None
        
        if not isinstance(vertices, list):
            return None
        vertices = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertices) < 2:
            return None

        if not isinstance(face, coreFace):
            face_vertices = sample(vertices,3)
            tempFace = Face.ByWire(Wire.ByVertices(face_vertices))
            # Flatten the input face
            flatFace = Face.Flatten(tempFace)
        else:
            flatFace = Face.Flatten(face)
            faceVertices = Face.Vertices(face)
            vertices += faceVertices
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        # Create a Vertex at the world's origin (0,0,0)
        world_origin = Vertex.ByCoordinates(0,0,0)

        # Create a cluster of the input vertices
        verticesCluster = Cluster.ByTopologies(vertices)

        # Flatten the cluster using the same transformations
        verticesCluster = Topology.Translate(verticesCluster, -xTran, -yTran, -zTran)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        flatVertices = Cluster.Vertices(verticesCluster)
        tempFlatVertices = []
        points = []
        for flatVertex in flatVertices:
            tempFlatVertices.append(Vertex.ByCoordinates(flatVertex.X(), flatVertex.Y(), 0))
            points.append([flatVertex.X(), flatVertex.Y()])
        flatVertices = tempFlatVertices
        delaunay = Delaunay(points)
        simplices = delaunay.simplices

        faces = []
        for simplex in simplices:
            tempTriangleVertices = []
            tempTriangleVertices.append(flatVertices[simplex[0]])
            tempTriangleVertices.append(flatVertices[simplex[1]])
            tempTriangleVertices.append(flatVertices[simplex[2]])
            faces.append(Face.ByWire(Wire.ByVertices(tempTriangleVertices)))

        shell = Shell.ByFaces(faces)
        if isinstance(face, coreFace):
            edges = Shell.Edges(shell)
            edgesCluster = Cluster.ByTopologies(edges)
            shell = Topology.Boolean(flatFace,edgesCluster, operation="slice")
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=1, z=0, degree=theta)
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=0, z=1, degree=phi)
        shell = Topology.Translate(shell, xTran, yTran, zTran)
        return shell

    @staticmethod
    def Edges(shell: coreShell) -> list:
        """
        Returns the edges of the input shell.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(shell, coreShell):
            return None

        list_edges = shell.edges()
        # edges = []
        # iterator = TopTools_ListIteratorOfListOfShape(list_edges)
        # while iterator.More():
        #     edge = iterator.Value()
        #     edges.append(edge)
        #     iterator.Next()

        return list_edges

    @staticmethod
    def ExternalBoundary(shell: coreShell) -> coreWire:
        """
        Returns the external boundary (closed wire) of the input shell.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        coreWire
            The external boundary (closed wire) of the input shell.

        """
        if not isinstance(shell, coreShell):
            return None
        edges = []
        edges: List[coreEdge] = shell.edges() # edges should be a list of topologies but it is a list of shapes
        obEdges = []
        for anEdge in edges:
            faces = anEdge.faces(shell)
            if len(faces) == 1:
                obEdges.append(anEdge)
        returnTopology = None
        try:
            returnTopology = coreWire.by_edges(obEdges)
        except:
            returnTopology: coreCluster = coreCluster.by_topologies(obEdges)
            returnTopology = returnTopology.self_merge()
        return returnTopology

    @staticmethod
    def Faces(shell: coreShell) -> list:
        """
        Returns the faces of the input shell.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(shell, coreShell):
            return None
        faces = []
        _ = shell.Faces(None, faces)
        return faces
    
    @staticmethod
    def IsInside(shell: coreShell, vertex: coreVertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input shell. Returns False otherwise. Inside is defined as being inside one of the shell's faces

        Parameters
        ----------
        shell : coreShell
            The input shell.
        vertex : coreVertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input shell. Returns False otherwise.

        """

        from Wrapper.Face import Face
        if not isinstance(shell, coreShell):
            return None
        if not isinstance(vertex, coreVertex):
            return None
        faces = Shell.Faces(shell)
        for f in faces:
            if Face.IsInside(fface=f, vertex=vertex, tolerance=tolerance):
                return True
        return False
    
    @staticmethod
    def IsOnBoundary(shell: coreShell, vertex: coreVertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input shell. Returns False otherwise. Inside is defined as being inside one of the shell's faces

        Parameters
        ----------
        shell : coreShell
            The input shell.
        vertex : coreVertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input shell. Returns False otherwise.

        """

        from Wrapper.Wire import Wire

        if not isinstance(shell, coreShell):
            return None
        if not isinstance(vertex, coreVertex):
            return None
        boundary = Shell.ExternalBoundary(shell)
        return Wire.IsInside(wire=boundary, vertex=vertex, tolerance=tolerance)
    
    @staticmethod
    def IsOutside(shell: coreShell, vertex: coreVertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input shell. Returns False otherwise. Inside is defined as being inside one of the shell's faces

        Parameters
        ----------
        shell : coreShell
            The input shell.
        vertex : coreVertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input shell. Returns False otherwise.

        """

        if not isinstance(shell, coreShell):
            return None
        if not isinstance(vertex, coreVertex):
            return None
        return not Wire.IsInside(shell=shell, vertex=vertex, tolerance=tolerance)
    
    @staticmethod
    def HyperbolicParaboloidRectangularDomain(origin: coreVertex = None, llVertex: coreVertex = None, lrVertex: coreVertex =None, ulVertex: coreVertex =None, urVertex: coreVertex = None,
                                              uSides: int = 10, vSides: int = 10, direction: list = [0,0,1], placement: str = "bottom") -> coreShell:
        """
        Creates a hyperbolic paraboloid with a rectangular domain.

        Parameters
        ----------
        origin : coreVertex , optional
            The origin of the hyperbolic parabolid. If set to None, it will be placed at the (0,0,0) origin. The default is None.
        llVertex : coreVertex , optional
            The lower left corner of the hyperbolic parabolid. If set to None, it will be set to (-0.5,-0.5,-0.5).
        lrVertex : coreVertex , optional
            The lower right corner of the hyperbolic parabolid. If set to None, it will be set to (0.5,-0.5,0.5).
        ulVertex : coreVertex , optional
            The upper left corner of the hyperbolic parabolid. If set to None, it will be set to (-0.5,0.5,0.5).
        urVertex : coreVertex , optional
            The upper right corner of the hyperbolic parabolid. If set to None, it will be set to (0.5,0.5,-0.5).
        uSides : int , optional
            The number of segments along the X axis. The default is 10.
        vSides : int , optional
            The number of segments along the Y axis. The default is 10.
        direction : list , optional
            The vector representing the up direction of the hyperbolic parabolid. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the hyperbolic parabolid. This can be "center", "lowerleft", "bottom". It is case insensitive. The default is "center".

        Returns
        -------
        coreShell
            The created hyperbolic paraboloid.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        if not isinstance(origin, coreVertex):
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(llVertex, coreVertex):
            llVertex = Vertex.ByCoordinates(-0.5,-0.5,-0.5)
        if not isinstance(lrVertex, coreVertex):
            lrVertex = Vertex.ByCoordinates(0.5,-0.5,0.5)
        if not isinstance(ulVertex, coreVertex):
            ulVertex = Vertex.ByCoordinates(-0.5,0.5,0.5)
        if not isinstance(urVertex, coreVertex):
            urVertex = Vertex.ByCoordinates(0.5,0.5,-0.5)
        e1 = Edge.ByVertices([llVertex, lrVertex])
        e3 = Edge.ByVertices([urVertex, ulVertex])
        edges = []
        for i in range(uSides+1):
            v1 = Edge.VertexByParameter(e1, float(i)/float(uSides))
            v2 = Edge.VertexByParameter(e3, 1.0 - float(i)/float(uSides))
            edges.append(Edge.ByVertices([v1, v2]))
        faces = []
        for i in range(uSides):
            for j in range(vSides):
                v1 = Edge.VertexByParameter(edges[i], float(j)/float(vSides))
                v2 = Edge.VertexByParameter(edges[i], float(j+1)/float(vSides))
                v3 = Edge.VertexByParameter(edges[i+1], float(j+1)/float(vSides))
                v4 = Edge.VertexByParameter(edges[i+1], float(j)/float(vSides))
                faces.append(Face.ByVertices([v1, v2, v4]))
                faces.append(Face.ByVertices([v4, v2, v3]))
        returnTopology = Shell.ByFaces(faces)
        if not returnTopology:
            returnTopology = None
        zeroOrigin = returnTopology.center_of_mass()
        xOffset = 0
        yOffset = 0
        zOffset = 0
        minX = min([llVertex.X(), lrVertex.X(), ulVertex.X(), urVertex.X()])
        maxX = max([llVertex.X(), lrVertex.X(), ulVertex.X(), urVertex.X()])
        minY = min([llVertex.Y(), lrVertex.Y(), ulVertex.Y(), urVertex.Y()])
        maxY = max([llVertex.Y(), lrVertex.Y(), ulVertex.Y(), urVertex.Y()])
        minZ = min([llVertex.Z(), lrVertex.Z(), ulVertex.Z(), urVertex.Z()])
        maxZ = max([llVertex.Z(), lrVertex.Z(), ulVertex.Z(), urVertex.Z()])
        if placement.lower() == "lowerleft":
            xOffset = -minX
            yOffset = -minY
            zOffset = -minZ
        elif placement.lower() == "bottom":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -minZ
        elif placement.lower() == "center":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -(minZ + (maxZ - minZ)*0.5)
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + direction[0]
        y2 = 0 + direction[1]
        z2 = 0 + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        returnTopology = Topology.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = Topology.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = Topology.Translate(returnTopology, zeroOrigin.x()+xOffset, zeroOrigin.y()+yOffset, zeroOrigin.z()+zOffset)
        return returnTopology
    
    @staticmethod
    def HyperbolicParaboloidCircularDomain(origin: coreVertex = None, radius: float = 0.5, sides: int = 36, rings: int = 10, A: float = 1.0, B: float = -1.0, direction: list = [0,0,1], placement: str = "bottom") -> coreShell:
        """
        Creates a hyperbolic paraboloid with a circular domain. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        origin : coreVertex , optional
            The origin of the hyperbolic parabolid. If set to None, it will be placed at the (0,0,0) origin. The default is None.
        radius : float , optional
            The desired radius of the hyperbolic paraboloid. The default is 0.5.
        sides : int , optional
            The desired number of sides of the hyperbolic parabolid. The default is 36.
        rings : int , optional
            The desired number of concentric rings of the hyperbolic parabolid. The default is 10.
        A : float , optional
            The *A* constant in the equation z = A*x^2^ + B*y^2^. The default is 1.0.
        B : float , optional
            The *B* constant in the equation z = A*x^2^ + B*y^2^. The default is -1.0.
        direction : list , optional
            The  vector representing the up direction of the hyperbolic paraboloid. The default is [0,0,1.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "bottom". It is case insensitive. The default is "center".

        Returns
        -------
        coreShell
            The created hyperboloic paraboloid.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Face import Face
        if not isinstance(origin, coreVertex):
            origin = Vertex.ByCoordinates(0,0,0)
        uOffset = float(360)/float(sides)
        vOffset = float(radius)/float(rings)
        faces = []
        for i in range(rings-1):
            r1 = radius - vOffset*i
            r2 = radius - vOffset*(i+1)
            for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x1 = math.sin(a1)*r1
                y1 = math.cos(a1)*r1
                z1 = A*x1*x1 + B*y1*y1
                x2 = math.sin(a1)*r2
                y2 = math.cos(a1)*r2
                z2 = A*x2*x2 + B*y2*y2
                x3 = math.sin(a2)*r2
                y3 = math.cos(a2)*r2
                z3 = A*x3*x3 + B*y3*y3
                x4 = math.sin(a2)*r1
                y4 = math.cos(a2)*r1
                z4 = A*x4*x4 + B*y4*y4
                v1 = coreVertex.by_coordinates(x1,y1,z1)
                v2 = coreVertex.by_coordinates(x2,y2,z2)
                v3 = coreVertex.by_coordinates(x3,y3,z3)
                v4 = coreVertex.by_coordinates(x4,y4,z4)
                f1 = Face.ByVertices([v1,v2,v4])
                f2 = Face.ByVertices([v4,v2,v3])
                faces.append(f1)
                faces.append(f2)
            a1 = math.radians(uOffset)*(sides-1)
            a2 = math.radians(360)
            x1 = math.sin(a1)*r1
            y1 = math.cos(a1)*r1
            z1 = A*x1*x1 + B*y1*y1
            x2 = math.sin(a1)*r2
            y2 = math.cos(a1)*r2
            z2 = A*x2*x2 + B*y2*y2
            x3 = math.sin(a2)*r2
            y3 = math.cos(a2)*r2
            z3 = A*x3*x3 + B*y3*y3
            x4 = math.sin(a2)*r1
            y4 = math.cos(a2)*r1
            z4 = A*x4*x4 + B*y4*y4
            v1 = coreVertex.by_coordinates(x1,y1,z1)
            v2 = coreVertex.by_coordinates(x2,y2,z2)
            v3 = coreVertex.by_coordinates(x3,y3,z3)
            v4 = coreVertex.by_coordinates(x4,y4,z4)
            f1 = Face.ByVertices([v1,v2,v4])
            f2 = Face.ByVertices([v4,v2,v3])
            faces.append(f1)
            faces.append(f2)
        # Special Case: Center triangles
        r = vOffset
        x1 = 0
        y1 = 0
        z1 = 0
        v1 = coreVertex.by_coordinates(x1,y1,z1)
        for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x2 = math.sin(a1)*r
                y2 = math.cos(a1)*r
                z2 = A*x2*x2 + B*y2*y2
                #z2 = 0
                x3 = math.sin(a2)*r
                y3 = math.cos(a2)*r
                z3 = A*x3*x3 + B*y3*y3
                #z3 = 0
                v2 = coreVertex.by_coordinates(x2,y2,z2)
                v3 = coreVertex.by_coordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)
        a1 = math.radians(uOffset)*(sides-1)
        a2 = math.radians(360)
        x2 = math.sin(a1)*r
        y2 = math.cos(a1)*r
        z2 = A*x2*x2 + B*y2*y2
        x3 = math.sin(a2)*r
        y3 = math.cos(a2)*r
        z3 = A*x3*x3 + B*y3*y3
        v2 = coreVertex.by_coordinates(x2,y2,z2)
        v3 = coreVertex.by_coordinates(x3,y3,z3)
        f1 = Face.ByVertices([v2,v1,v3])
        faces.append(f1)
        returnTopology = coreShell.by_faces(faces, 1e-03, False)
        if not returnTopology:
            returnTopology = coreCluster.by_topologies(faces)

        
        # Description: instead of occt_shapes we are expecting a simple list of core.vertex type objects!!!
        # Problem: TopExp_Explorer in static_downward_navigation returns new instances of occt_shapes.
        #          These shapes has not been registered and associated to topologies and core.vertex objects cannot be accessed.
        vertices = returnTopology.vertices() # vertices --> is a TopTools_ListOfShape type object, that contains ooct_shapes.
        xList = []
        yList = []
        zList = []

        # Original version:
        #------------------
        for aVertex in vertices:
            xList.append(aVertex.x())
            yList.append(aVertex.y())
            zList.append(aVertex.z())

        # New approach to iterate over the TopTools_ListOfShapes type object
        #-------------------------------------------------------------------
        # iterator = TopTools_ListIteratorOfListOfShape(vertices)

        # while iterator.More():

        #     occt_vertex = iterator.Value()
        #     aVertex = coreTopology.shapeID_to_topology[occt_vertex.id]

        #     xList.append(aVertex.x())
        #     yList.append(aVertex.y())
        #     zList.append(aVertex.z())

        #     iterator.Next()

        minX = min(xList)
        maxX = max(xList)
        minY = min(yList)
        maxY = max(yList)
        minZ = min(zList)
        maxZ = max(zList)
        zeroOrigin = returnTopology.CenterOfMass()
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = -minX
            yOffset = -minY
            zOffset = -minZ
        elif placement.lower() == "bottom":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -minZ
        elif placement.lower() == "center":
            xOffset = -(minX + (maxX - minX)*0.5)
            yOffset = -(minY + (maxY - minY)*0.5)
            zOffset = -(minZ + (maxZ - minZ)*0.5)
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + direction[0]
        y2 = 0 + direction[1]
        z2 = 0 + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        zeroOrigin = coreVertex.by_coordinates(0,0,0)
        returnTopology = coreTopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 1, 0, theta)
        returnTopology = coreTopologyUtility.Rotate(returnTopology, zeroOrigin, 0, 0, 1, phi)
        returnTopology = coreTopologyUtility.Translate(returnTopology, origin.X()+xOffset, origin.Y()+yOffset, origin.Z()+zOffset)
        return returnTopology
    
    @staticmethod
    def InternalBoundaries(shell: coreShell) -> coreTopology:
        """
        Returns the internal boundaries (closed wires) of the input shell. Internal boundaries are considered holes.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        coreTopology
            The wire if a single hole or a cluster of wires if more than one hole.

        """
        from Wrapper.Cluster import Cluster
        edges = []
        _ = shell.Edges(None, edges)
        ibEdges = []
        for anEdge in edges:
            faces = []
            _ = anEdge.Faces(shell, faces)
            if len(faces) > 1:
                ibEdges.append(anEdge)
        return Cluster.SelfMerge(Cluster.ByTopologies(ibEdges))
    
    @staticmethod
    def IsClosed(shell: coreShell) -> bool:
        """
        Returns True if the input shell is closed. Returns False otherwise.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        bool
            True if the input shell is closed. False otherwise.

        """
        return shell.IsClosed()

    @staticmethod
    def Pie(origin: coreVertex = None, radiusA: float = 0.5, radiusB: float = 0.0, sides: int = 32, rings: int = 1, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreShell:
        """
        Creates a pie shape.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the pie. The default is None which results in the pie being placed at (0,0,0).
        radiusA : float , optional
            The outer radius of the pie. The default is 0.5.
        radiusB : float , optional
            The inner radius of the pie. The default is 0.25.
        sides : int , optional
            The number of sides of the pie. The default is 32.
        rings : int , optional
            The number of rings of the pie. The default is 1.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the pie. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the pie. The default is 360.
        direction : list , optional
            The vector representing the up direction of the pie. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreShell
            The created pie.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            return None
        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) < tolerance:
            return None
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        angleRange = toAngle - fromAngle
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusB > radiusA:
            temp = radiusA
            radiusA = radiusB
            radiusB = temp
        if abs(radiusA - radiusB) < tolerance or radiusA < tolerance:
            return None
        radiusRange = radiusA - radiusB
        sides = int(abs(math.floor(sides)))
        if sides < 3:
            return None
        rings = int(abs(rings))
        if radiusB < tolerance:
            radiusB = 0
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = radiusA
            yOffset = radiusA
        uOffset = float(angleRange)/float(sides)
        vOffset = float(radiusRange)/float(rings)
        faces = []
        if radiusB > tolerance:
            for i in range(rings):
                r1 = radiusA - vOffset*i
                r2 = radiusA - vOffset*(i+1)
                for j in range(sides):
                    a1 = fromAngle + uOffset*j
                    a2 = fromAngle + uOffset*(j+1)
                    x1 = math.sin(a1)*r1
                    y1 = math.cos(a1)*r1
                    z1 = 0
                    x2 = math.sin(a1)*r2
                    y2 = math.cos(a1)*r2
                    z2 = 0
                    x3 = math.sin(a2)*r2
                    y3 = math.cos(a2)*r2
                    z3 = 0
                    x4 = math.sin(a2)*r1
                    y4 = math.cos(a2)*r1
                    z4 = 0
                    v1 = Vertex.ByCoordinates(x1,y1,z1)
                    v2 = Vertex.ByCoordinates(x2,y2,z2)
                    v3 = Vertex.ByCoordinates(x3,y3,z3)
                    v4 = Vertex.ByCoordinates(x4,y4,z4)
                    f1 = Face.ByVertices([v1,v2,v3,v4])
                    faces.append(f1)
        else:
            x1 = 0
            y1 = 0
            z1 = 0
            v1 = Vertex.ByCoordinates(x1,y1,z1)
            for j in range(sides):
                a1 = fromAngle + uOffset*j
                a2 = fromAngle + uOffset*(j+1)
                x2 = math.sin(a1)*radiusA
                y2 = math.cos(a1)*radiusA
                z2 = 0
                x3 = math.sin(a2)*radiusA
                y3 = math.cos(a2)*radiusA
                z3 = 0
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)

        shell = Shell.ByFaces(faces, tolerance)
        if not shell:
            return None
        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0 + direction[0]
        y2 = 0 + direction[1]
        z2 = 0 + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < tolerance:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        shell = Topology.Rotate(shell, origin, 0, 1, 0, theta)
        shell = Topology.Rotate(shell, origin, 0, 0, 1, phi)
        shell = Topology.Translate(shell, origin.x()+xOffset, origin.y()+yOffset, origin.z()+zOffset)
        return shell

    @staticmethod
    def Rectangle(origin: coreVertex = None, width: float = 1.0, length: float = 1.0, uSides: int = 2, vSides: int = 2, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreShell:
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0,0,0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        uSides : int , optional
            The number of sides along the width. The default is 2.
        vSides : int , optional
            The number of sides along the length. The default is 2.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreShell
            The created shell.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            return None
        uOffset = float(width)/float(uSides)
        vOffset = float(length)/float(vSides)
        faces = []
        if placement.lower() == "center":
            wOffset = width*0.5
            lOffset = length*0.5
        else:
            wOffset = 0
            lOffset = 0
        for i in range(uSides):
            for j in range(vSides):
                rOrigin = Vertex.ByCoordinates(i*uOffset - wOffset, j*vOffset - lOffset, 0)
                w = Wire.Rectangle(origin=rOrigin, width=uOffset, length=vOffset, direction=[0,0,1], placement="lowerleft", tolerance=tolerance)
                f = Face.ByWire(w)
                faces.append(f)
        shell = Shell.ByFaces(faces)
        x1 = origin.x()
        y1 = origin.y()
        z1 = origin.z()
        x2 = origin.x() + direction[0]
        y2 = origin.y() + direction[1]
        z2 = origin.z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        shell = Topology.Rotate(shell, origin, 0, 1, 0, theta)
        shell = Topology.Rotate(shell, origin, 0, 0, 1, phi)
        return shell

    def Roof(face, degree=45, angTolerance=2, tolerance=0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : coreFace
            The input face.
        degree : float , optioal
            The desired angle in degrees of the roof. The default is 45.
        angTolerance : float , optional
            The desired angular tolerance. The default is 2. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        coreShell
            The created roof.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Shell import Shell
        from Wrapper.Cell import Cell
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        import topologic
        import math

        def nearest_vertex_2d(v, vertices, tolerance=0.001):
            for vertex in vertices:
                x2 = Vertex.X(vertex)
                y2 = Vertex.Y(vertex)
                temp_v = Vertex.ByCoordinates(x2, y2, Vertex.Z(v))
                if Vertex.Distance(v, temp_v) <= tolerance:
                    return vertex
            return None
        
        if not isinstance(face, coreFace):
            return None
        degree = abs(degree)
        if degree >= 90-tolerance:
            return None
        if degree < tolerance:
            return None
        flat_face = Face.Flatten(face)
        d = Topology.Dictionary(flat_face)
        roof = Wire.Roof(flat_face, degree)
        if not roof:
            return None
        shell = Shell.Skeleton(flat_face)
        faces = Shell.Faces(shell)
        
        if not faces:
            return None
        triangles = []
        for face in faces:
            internalBoundaries = Face.InternalBoundaries(face)
            if len(internalBoundaries) == 0:
                if len(Topology.Vertices(face)) > 3:
                    triangles += Face.Triangulate(face)
                else:
                    triangles += [face]

        roof_vertices = Topology.Vertices(roof)
        flat_vertices = []
        for rv in roof_vertices:
            flat_vertices.append(Vertex.ByCoordinates(Vertex.X(rv), Vertex.Y(rv), 0))

        final_triangles = []
        for triangle in triangles:
            if len(Topology.Vertices(triangle)) > 3:
                triangles = Face.Triangulate(triangle)
            else:
                triangles = [triangle]
            final_triangles += triangles

        final_faces = []
        for triangle in final_triangles:
            face_vertices = Topology.Vertices(triangle)
            top_vertices = []
            for sv in face_vertices:
                temp = nearest_vertex_2d(sv, roof_vertices, tolerance=tolerance)
                if temp:
                    top_vertices.append(temp)
                else:
                    top_vertices.append(sv)
            tri_face = Face.ByVertices(top_vertices)
            final_faces.append(tri_face)

        shell = Shell.ByFaces(final_faces, tolerance=tolerance)
        if not shell:
            shell = Cluster.ByTopologies(final_faces)
        try:
            shell = Topology.RemoveCoplanarFaces(shell, angTolerance=angTolerance)
        except:
            pass
        xTran = Dictionary.ValueAtKey(d,"xTran")
        yTran = Dictionary.ValueAtKey(d,"yTran")
        zTran = Dictionary.ValueAtKey(d,"zTran")
        phi = Dictionary.ValueAtKey(d,"phi")
        theta = Dictionary.ValueAtKey(d,"theta")
        shell = Topology.Rotate(shell, origin=Vertex.Origin(), x=0, y=1, z=0, degree=theta)
        shell = Topology.Rotate(shell, origin=Vertex.Origin(), x=0, y=0, z=1, degree=phi)
        shell = Topology.Translate(shell, xTran, yTran, zTran)
        return shell
    
    @staticmethod
    def SelfMerge(shell: coreShell, angTolerance: float = 0.1) -> coreFace:
        """
        Creates a face by merging the faces of the input shell. The shell must be planar within the input angular tolerance.

        Parameters
        ----------
        shell : coreShell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        coreFace
            The created face.

        """
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Shell import Shell
        from Wrapper.Topology import Topology
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        if not isinstance(shell, coreShell):
            return None
        ext_boundary = Shell.ExternalBoundary(shell)
        if isinstance(ext_boundary, coreWire):
            try:
                return coreFace.ByExternalBoundary(Topology.RemoveCollinearEdges(ext_boundary, angTolerance))
            except:
                try:
                    return coreFace.ByExternalBoundary(Wire.Planarize(Topology.RemoveCollinearEdges(ext_boundary, angTolerance)))
                except:
                    print("FaceByPlanarShell - Error: The input Wire is not planar and could not be fixed. Returning None.")
                    return None
        elif isinstance(ext_boundary, coreCluster):
            wires = []
            _ = ext_boundary.Wires(None, wires)
            faces = []
            areas = []
            for aWire in wires:
                try:
                    aFace = coreFace.ByExternalBoundary(Topology.RemoveCollinearEdges(aWire, angTolerance))
                except:
                    aFace = coreFace.ByExternalBoundary(Wire.Planarize(Topology.RemoveCollinearEdges(aWire, angTolerance)))
                anArea = coreFaceUtility.Area(aFace)
                faces.append(aFace)
                areas.append(anArea)
            max_index = areas.index(max(areas))
            ext_boundary = faces[max_index]
            int_boundaries = list(set(faces) - set([ext_boundary]))
            int_wires = []
            for int_boundary in int_boundaries:
                temp_wires = []
                _ = int_boundary.Wires(None, temp_wires)
                int_wires.append(Topology.RemoveCollinearEdges(temp_wires[0], angTolerance))
            temp_wires = []
            _ = ext_boundary.Wires(None, temp_wires)
            ext_wire = Topology.RemoveCollinearEdges(temp_wires[0], angTolerance)
            try:
                return Face.ByWires(ext_wire, int_wires)
            except:
                return Face.ByWires(Wire.Planarize(ext_wire), planarizeList(int_wires))
        else:
            return None

    def Skeleton(face, tolerance=0.001):
        """
            Creates a shell through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : coreFace
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        coreShell
            The created straight skeleton.

        """
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        import topologic
        import math

        if not isinstance(face, coreFace):
            return None
        roof = Wire.Skeleton(face)
        if not roof:
            return None
        br = Wire.BoundingRectangle(roof) #This works even if it is a Cluster not a Wire
        br = Topology.Scale(br, Topology.Centroid(br), 1.5, 1.5, 1)
        bf = Face.ByWire(br)
        large_shell = Topology.Boolean(bf, roof, operation="slice")
        if not large_shell:
            return None
        faces = Topology.Faces(large_shell)
        if not faces:
            return None
        final_faces = []
        for f in faces:
            internalBoundaries = Face.InternalBoundaries(f)
            if len(internalBoundaries) == 0:
                final_faces.append(f)
        shell = Shell.ByFaces(final_faces)
        return shell
    
    @staticmethod
    def Vertices(shell: coreShell) -> list:
        """
        Returns the vertices of the input shell.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(shell, coreShell):
            return None
        vertices = []
        _ = shell.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Voronoi(vertices: list, face: coreFace = None) -> coreShell:
        """
        Returns a voronoi partitioning of the input face based on the input vertices. The vertices must be coplanar and within the face. See https://en.wikipedia.org/wiki/Voronoi_diagram.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : coreFace , optional
            The input face. If the face is not set an optimised bounding rectangle of the input vertices is used instead. The default is None.

        Returns
        -------
        shell
            A shell representing the voronoi partitioning of the input face.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        import sys
        import subprocess

        try:
            from scipy.spatial import Voronoi
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'scipy', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                from scipy.spatial import Voronoi
            except:
                print("Shell.Voronoi - ERROR: Could not import scipy. Returning None.")
                return None
        
        if not isinstance(face, coreFace):
            cluster = Cluster.ByTopologies(vertices)
            br = Wire.BoundingRectangle(cluster, optimize=5)
            face = Face.ByWire(br)
        if not isinstance(vertices, list):
            return None
        vertices = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertices) < 2:
            return None

        # Flatten the input face
        flatFace = Face.Flatten(face)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        # Create a Vertex at the world's origin (0,0,0)
        world_origin = Vertex.ByCoordinates(0,0,0)

        # Create a cluster of the input vertices
        verticesCluster = Cluster.ByTopologies(vertices)

        # Flatten the cluster using the same transformations
        verticesCluster = Topology.Translate(verticesCluster, -xTran, -yTran, -zTran)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        verticesCluster = Topology.Rotate(verticesCluster, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        flatVertices = Cluster.Vertices(verticesCluster)
        points = []
        for flatVertex in flatVertices:
            points.append([flatVertex.X(), flatVertex.Y()])

        br = Wire.BoundingRectangle(flatFace)
        br_vertices = Wire.Vertices(br)
        br_x = []
        br_y = []
        for br_v in br_vertices:
            x, y = Vertex.Coordinates(br_v, outputType="xy")
            br_x.append(x)
            br_y.append(y)
        min_x = min(br_x)
        max_x = max(br_x)
        min_y = min(br_y)
        max_y = max(br_y)
        br_width = abs(max_x - min_x)
        br_length = abs(max_y - min_y)

        points.append((-br_width*4, -br_length*4))
        points.append((-br_width*4, br_length*4))
        points.append((br_width*4, -br_length*4))
        points.append((br_width*4, br_length*4))

        voronoi = Voronoi(points, furthest_site=False)
        voronoiVertices = []
        for v in voronoi.vertices:
            voronoiVertices.append(Vertex.ByCoordinates(v[0], v[1], 0))

        faces = []
        for region in voronoi.regions:
            tempWire = []
            if len(region) > 1 and not -1 in region:
                for v in region:
                    tempWire.append(Vertex.ByCoordinates(voronoiVertices[v].X(), voronoiVertices[v].Y(),0))
                faces.append(Face.ByWire(Wire.ByVertices(tempWire, close=True)))
        shell = Shell.ByFaces(faces)
        edges = Shell.Edges(shell)
        edgesCluster = Cluster.ByTopologies(edges)
        shell = Topology.Boolean(flatFace,edgesCluster, operation="slice")
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=1, z=0, degree=theta)
        shell = Topology.Rotate(shell, origin=world_origin, x=0, y=0, z=1, degree=phi)
        shell = Topology.Translate(shell, xTran, yTran, zTran)
        return shell

    @staticmethod
    def Wires(shell: coreShell) -> list:
        """
        Returns the wires of the input shell.

        Parameters
        ----------
        shell : coreShell
            The input shell.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(shell, coreShell):
            return None
        wires = []
        _ = shell.Wires(None, wires)
        return wires

    
    
    
    
    