import math

# Core
from Core.Topology import Topology as coreTopology
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Shell import Shell as coreShell
from Core.Cluster import Cluster as coreCluster

from Core.Dictionary import Dictionary as coreDictionary
from Core.Utilities.TopologicUtilities import VertexUtility, FaceUtility

# Wrapper
from Wrapper.Vector import Vector
from Wrapper.Wire import Wire

class Face(coreFace):
    @staticmethod
    def AddInternalBoundaries(face: coreFace, wires: list) -> coreFace:
        """
        Adds internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : coreFace
            The input face.
        wires : list
            The input list of internal boundaries (closed wires).

        Returns
        -------
        coreFace
            The created face with internal boundaries added to it.

        """
        if not face:
            return None
        if not isinstance(face, coreFace):
            return None
        if not wires:
            return face
        if not isinstance(wires, list):
            return face
        wireList = [w for w in wires if isinstance(w, coreWire)]
        if len(wireList) < 1:
            return face
        faceeb = face.external_boundary()
        faceibList = []
        _ = face.internal_boundaries(faceibList)
        for wire in wires:
            faceibList.append(wire)
        return coreFace.by_external_internal_boundaries(faceeb, faceibList)

    @staticmethod
    def AddInternalBoundariesCluster(face: coreFace, cluster: coreCluster) -> coreFace:
        """
        Adds the input cluster of internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : coreFace
            The input face.
        cluster : coreCluster
            The input cluster of internal boundaries (topologic wires).

        Returns
        -------
        coreFace
            The created face with internal boundaries added to it.

        """
        if not face:
            return None
        if not isinstance(face, coreFace):
            return None
        if not cluster:
            return face
        if not isinstance(cluster, coreCluster):
            return face
        wires = []
        _ = cluster.wires(None, wires)
        return Face.AddInternalBoundaries(face, wires)
    
    @staticmethod
    def Angle(faceA: coreFace, faceB: coreFace, mantissa: int = 4) -> float:
        """
        Returns the angle in degrees between the two input faces.

        Parameters
        ----------
        faceA : coreFace
            The first input face.
        faceB : coreFace
            The second input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The angle in degrees between the two input faces.

        """
        from Wrapper.Vector import Vector
        if not faceA or not isinstance(faceA, coreFace):
            return None
        if not faceB or not isinstance(faceB, coreFace):
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return round((Vector.Angle(dirA, dirB)), mantissa)

    @staticmethod
    def Area(face: coreFace, mantissa: int = 4) -> float:
        """
        Returns the area of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The area of the input face.

        """
        if not isinstance(face, coreFace):
            return None
        area = None
        try:
            area = round(FaceUtility.area(face), mantissa)
        except:
            area = None
        return area

    @staticmethod
    def BoundingRectangle(topology: coreTopology, optimize: int = 0) -> coreFace:
        """
        Returns a face representing a bounding rectangle of the input topology. The returned face contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting face will become axis-aligned.

        Parameters
        ----------
        topology : coreTopology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        
        Returns
        -------
        coreFace
            The bounding rectangle of the input topology.

        """
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        def bb(topology: coreTopology):
            vertices = []
            _ = topology.vertices(None, vertices)
            x = []
            y = []
            for aVertex in vertices:
                x.append(aVertex.X())
                y.append(aVertex.Y())
            minX = min(x)
            minY = min(y)
            maxX = max(x)
            maxY = max(y)
            return [minX, minY, maxX, maxY]

        if not isinstance(topology, coreTopology):
            return None
        vertices = Topology.SubTopologies(topology, subTopologyType="vertex")
        topology = Cluster.ByTopologies(vertices)
        boundingBox = bb(topology)
        minX = boundingBox[0]
        minY = boundingBox[1]
        maxX = boundingBox[2]
        maxY = boundingBox[3]
        w = abs(maxX - minX)
        l = abs(maxY - minY)
        best_area = l*w
        orig_area = best_area
        best_z = 0
        best_bb = boundingBox
        origin = Topology.Centroid(topology)
        optimize = min(max(optimize, 0), 10)
        if optimize > 0:
            factor = (round(((11 - optimize)/30 + 0.57), 2))
            flag = False
            for n in range(10,0,-1):
                if flag:
                    break
                za = n
                zb = 90+n
                zc = n
                for z in range(za,zb,zc):
                    if flag:
                        break
                    t = Topology.Rotate(topology, origin=origin, x=0,y=0,z=1, degree=z)
                    minX, minY, maxX, maxY = bb(t)
                    w = abs(maxX - minX)
                    l = abs(maxY - minY)
                    area = l*w
                    if area < orig_area*factor:
                        best_area = area
                        best_z = z
                        best_bb = [minX, minY, maxX, maxY]
                        flag = True
                        break
                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_bb = [minX, minY, maxX, maxY]
                        
        else:
            best_bb = boundingBox

        minX, minY, maxX, maxY = best_bb
        vb1 = coreVertex.by_coordinates(minX, minY, 0)
        vb2 = coreVertex.by_coordinates(maxX, minY, 0)
        vb3 = coreVertex.by_coordinates(maxX, maxY, 0)
        vb4 = coreVertex.by_coordinates(minX, maxY, 0)

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire)
        baseFace = Topology.Rotate(baseFace, origin=origin, x=0,y=0,z=1, degree=-best_z)
        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        baseFace = Topology.SetDictionary(baseFace, dictionary)
        return baseFace
    
    @staticmethod
    def ByEdges(edges: list) -> coreFace:
        """
        Creates a face from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.

        Returns
        -------
        face : coreFace
            The created face.

        """
        from Wrapper.Wire import Wire
        wire = Wire.ByEdges(edges)
        if not wire:
            return None
        if not isinstance(wire, coreWire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def ByEdgesCluster(cluster: coreCluster) -> coreFace:
        """
        Creates a face from the input cluster of edges.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster of edges.

        Returns
        -------
        face : coreFace
            The created face.

        """
        from Wrapper.Cluster import Cluster
        if not isinstance(cluster, coreCluster):
            return None
        edges = Cluster.Edges(cluster)
        return Face.ByEdges(edges)

    @staticmethod
    def ByOffset(face: coreFace, offset: float = 1.0, miter: bool = False, miterThreshold: float = None, offsetKey: str = None, miterThresholdKey: str = None, step: bool = True) -> coreFace:
        """
        Creates an offset wire from the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        offset : float , optional
            The desired offset distance. The default is 1.0.
        miter : bool , optional
            if set to True, the corners will be mitered. The default is False.
        miterThreshold : float , optional
            The distance beyond which a miter should be added. The default is None which means the miter threshold is set to the offset distance multiplied by the square root of 2.
        offsetKey : str , optional
            If specified, the dictionary of the edges will be queried for this key to sepcify the desired offset. The default is None.
        miterThresholdKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to sepcify the desired miter threshold distance. The default is None.
        step : bool , optional
            If set to True, The transition between collinear edges with different offsets will be a step. Otherwise, it will be a continous edge. The default is True.

        Returns
        -------
        coreWire
            The created wire.

        """
        from Wrapper.Wire import Wire

        eb = Face.Wire(face)
        internal_boundaries = Face.InternalBoundaries(face)
        offset_external_boundary = Wire.ByOffset(wire=eb, offset=offset, miter=miter, miterThreshold=miterThreshold, offsetKey=offsetKey, miterThresholdKey=miterThresholdKey, step=step)
        offset_internal_boundaries = []
        for internal_boundary in internal_boundaries:
            offset_internal_boundaries.append(Wire.ByOffset(wire=internal_boundary, offset=offset, miter=miter, miterThreshold=miterThreshold, offsetKey=offsetKey, miterThresholdKey=miterThresholdKey, step=step))
        return Face.ByWires(offset_external_boundary, offset_internal_boundaries)
    
    @staticmethod
    def ByShell(shell: coreShell, angTolerance: float = 0.1)-> coreFace:
        """
        Creates a face by merging the faces of the input shell.

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
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Shell import Shell
        from Wrapper.Topology import Topology
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        
        ext_boundary = Shell.ExternalBoundary(shell)
        ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance)
        if not Topology.IsPlanar(ext_boundary):
            ext_boundary = Wire.Planarize(ext_boundary)

        if isinstance(ext_boundary, coreWire):
            try:
                return coreFace.by_external_boundary(Topology.RemoveCollinearEdges(ext_boundary, angTolerance))
            except:
                try:
                    w = Wire.Planarize(ext_boundary)
                    f = Face.ByWire(w)
                    return f
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
                anArea = FaceUtility.area(aFace)
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
                return coreFace.ByExternalInternalBoundaries(ext_wire, int_wires)
            except:
                return coreFace.ByExternalInternalBoundaries(Wire.Planarize(ext_wire), planarizeList(int_wires))
        else:
            return None
    
    @staticmethod
    def ByVertices(vertices: list) -> coreFace:
        
        """
        Creates a face from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.

        Returns
        -------
        coreFace
            The created face.

        """
        from Wrapper.Topology import Topology
        from Wrapper.Wire import Wire

        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertexList) < 3:
            return None
        
        w = Wire.ByVertices(vertexList)
        f = Face.ByExternalBoundary(w)
        return f

    @staticmethod
    def ByVerticesCluster(cluster: coreCluster) -> coreFace:
        """
        Creates a face from the input cluster of vertices.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster of vertices.

        Returns
        -------
        coreFace
            The crearted face.

        """
        from Wrapper.Cluster import Cluster
        if not isinstance(cluster, coreCluster):
            return None
        vertices = Cluster.Vertices(cluster)
        return Face.ByVertices(vertices)

    @staticmethod
    def ByWire(wire: coreWire) -> coreFace:
        """
        Creates a face from the input closed wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        coreFace or list
            The created face. If the wire is non-planar, the method will attempt to triangulate the wire and return a list of faces.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Shell import Shell
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        import random

        def triangulateWire(wire):
            wire = Topology.RemoveCollinearEdges(wire)
            vertices = Wire.Vertices(wire)
            shell = Shell.Delaunay(vertices)
            if isinstance(shell, coreShell):
                return Shell.Faces(shell)
            else:
                return []
        if not isinstance(wire, coreWire):
            return None
        if not Wire.IsClosed(wire):
            return None
        
        edges = Wire.Edges(wire)
        wire = Topology.SelfMerge(Cluster.ByTopologies(edges))
        vertices = Wire.Vertices(wire)
        try:
            fList = coreFace.ByExternalBoundary(wire)
        except:
            if len(vertices) > 3:
                fList = triangulateWire(wire)
            else:
                fList = []
        
        if not isinstance(fList, list):
            fList = [fList]

        returnList = []
        for f in fList:
            if Face.Area(f) < 0:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                try:
                    f = coreFace.ByExternalBoundary(wire)
                    returnList.append(f)
                except:
                    pass
            else:
                returnList.append(f)
        if len(returnList) == 0:
            return None
        elif len(returnList) == 1:
            return returnList[0]
        else:
            return returnList

    @staticmethod
    def ByWires(externalBoundary: coreWire, internalBoundaries: list = []) -> coreFace:
        """
        Creates a face from the input external boundary (closed wire) and the input list of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : coreWire
            The input external boundary.
        internalBoundaries : list , optional
            The input list of internal boundaries (closed wires). The default is an empty list.

        Returns
        -------
        coreFace
            The created face.

        """
        if not isinstance(externalBoundary, coreWire):
            return None
        if not Wire.IsClosed(externalBoundary):
            return None
        ibList = [x for x in internalBoundaries if isinstance(x, coreWire) and Wire.IsClosed(x)]
        return coreFace.ByExternalInternalBoundaries(externalBoundary, ibList)

    @staticmethod
    def ByWiresCluster(externalBoundary: coreWire, internalBoundariesCluster: coreCluster = None) -> coreFace:
        """
        Creates a face from the input external boundary (closed wire) and the input cluster of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : coreWire
            The input external boundary (closed wire).
        internalBoundariesCluster : coreCluster
            The input cluster of internal boundaries (closed wires). The default is None.

        Returns
        -------
        coreFace
            The created face.

        """
        from Wrapper.Wire import Wire
        from Wrapper.Cluster import Cluster
        if not isinstance(externalBoundary, coreWire):
            return None
        if not Wire.IsClosed(externalBoundary):
            return None
        if not internalBoundariesCluster:
            internalBoundaries = []
        elif not isinstance(internalBoundariesCluster, coreCluster):
            return None
        else:
            internalBoundaries = Cluster.Wires(internalBoundariesCluster)
        return Face.ByWires(externalBoundary, internalBoundaries)
    
    @staticmethod
    def NorthArrow(origin: coreVertex = None, radius: float = 0.5, sides: int = 16, direction: list = [0,0,1], northAngle: float = 0.0,
                   placement: str = "center", tolerance: float = 0.0001) -> coreFace:
        """
        Creates a north arrow.

        Parameters
        ----------
        origin : coreVertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        northAngle : float , optional
            The angular offset in degrees from the positive Y axis direction. The angle is measured in a counter-clockwise fashion where 0 is positive Y, 90 is negative X, 180 is negative Y, and 270 is positive X.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreFace
            The created circle.

        """
        from Wrapper.Topology import Topology
        from Wrapper.Vertex import Vertex
        if not origin:
            origin = Vertex.Origin()
        
        c = Face.Circle(origin=origin, radius=radius, sides=sides, direction=[0,0,1], placement="center", tolerance=tolerance)
        r = Face.Rectangle(origin=origin, width=radius*0.01,length=radius*1.2, placement="lowerleft")
        r = Topology.Translate(r, -0.005*radius,0,0)
        arrow = Topology.Difference(c, r)
        arrow = Topology.Rotate(arrow, Vertex.Origin(), 0,0,1,northAngle)
        if placement.lower() == "lowerleft":
            arrow = Topology.Translate(arrow, radius, radius, 0)
        elif placement.lower() == "upperleft":
            arrow = Topology.Translate(arrow, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            arrow = Topology.Translate(arrow, -radius, radius, 0)
        elif placement.lower() == "upperright":
            arrow = Topology.Translate(arrow, -radius, -radius, 0)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + direction[0]
        y2 = origin.Y() + direction[1]
        z2 = origin.Z() + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        arrow = Topology.Rotate(arrow, origin, 0, 1, 0, theta)
        arrow = Topology.Rotate(arrow, origin, 0, 0, 1, phi)
        return arrow

    @staticmethod
    def Circle(origin: coreVertex = None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0,0,1],
                   placement: str = "center", tolerance: float = 0.0001) -> coreFace:
        """
        Creates a circle.

        Parameters
        ----------
        origin : coreVertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreFace
            The created circle.

        """
        from Wrapper.Wire import Wire
        wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=True, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, coreWire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Compactness(face: coreFace, mantissa: int = 4) -> float:
        """
        Returns the compactness measure of the input face. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        face : coreFace
            The input face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The compactness measure of the input face.

        """
        exb = face.external_boundary()
        edges = []
        _ = exb.Edges(None, edges)
        perimeter = 0.0
        for anEdge in edges:
            perimeter = perimeter + abs(coreEdgeUtility.Length(anEdge))
        area = abs(FaceUtility.area(face))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        if area <= 0:
            return None
        if perimeter <= 0:
            return None
        compactness = (math.pi*(2*math.sqrt(area/math.pi)))/perimeter
        return round(compactness, mantissa)

    @staticmethod
    def CompassAngle(face: coreFace, north: list = None, mantissa: int = 4) -> float:
        """
        Returns the horizontal compass angle in degrees between the normal vector of the input face and the input vector. The angle is measured in counter-clockwise fashion. Only the first two elements of the vectors are considered.

        Parameters
        ----------
        face : coreFace
            The input face.
        north : list , optional
            The second vector representing the north direction. The default is the positive YAxis ([0,1,0]).
        mantissa : int, optional
            The length of the desired mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        float
            The horizontal compass angle in degrees between the direction of the face and the second input vector.

        """
        from Wrapper.Vector import Vector
        if not isinstance(face, coreFace):
            return None
        if not north:
            north = Vector.North()
        dirA = Face.NormalAtParameters(face,mantissa=mantissa)
        return Vector.CompassAngle(vectorA=dirA, vectorB=north, mantissa=mantissa)

    @staticmethod
    def Edges(face: coreFace) -> list:
        """
        Returns the edges of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        list
            The list of edges.

        """
        if not isinstance(face, coreFace):
            return None
        edges = []
        _ = face.Edges(None, edges)
        return edges

    @staticmethod
    def Einstein(origin: coreVertex = None, radius: float = 0.5, direction: list = [0,0,1], placement: str = "center") -> coreFace:
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the tile. The default is None which results in the tiles first vertex being placed at (0,0,0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        
        """
        from Wrapper.Wire import Wire
        wire = Wire.Einstein(origin=origin, radius=radius, direction=direction, placement=placement)
        if not isinstance(wire, coreWire):
            return None
        return Face.ByWire(wire)
    
    @staticmethod
    def ExternalBoundary(face: coreFace) -> coreWire:
        """
        Returns the external boundary (closed wire) of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        coreWire
            The external boundary of the input face.

        """
        return face.ExternalBoundary()
    
    @staticmethod
    def FacingToward(face: coreFace, direction: list = [0,0,-1], asVertex: bool = False, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input face is facing toward the input direction.

        Parameters
        ----------
        face : coreFace
            The input face.
        direction : list , optional
            The input direction. The default is [0,0,-1].
        asVertex : bool , optional
            If set to True, the direction is treated as an actual vertex in 3D space. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the face is facing toward the direction. False otherwise.

        """
        faceNormal = coreFaceUtility.NormalAtParameters(face,0.5, 0.5)
        faceCenter = coreFaceUtility.VertexAtParameters(face,0.5,0.5)
        cList = [faceCenter.X(), faceCenter.Y(), faceCenter.Z()]
        try:
            vList = [direction.X(), direction.Y(), direction.Z()]
        except:
            try:
                vList = [direction[0], direction[1], direction[2]]
            except:
                raise Exception("Face.FacingToward - Error: Could not get the vector from the input direction")
        if asVertex:
            dV = [vList[0]-cList[0], vList[1]-cList[1], vList[2]-cList[2]]
        else:
            dV = vList
        uV = Vector.Normalize(dV)
        dot = sum([i*j for (i, j) in zip(uV, faceNormal)])
        if dot < tolerance:
            return False
        return True
    
    @staticmethod
    def Flatten(face: coreFace, originA: coreVertex = None, originB: coreVertex = None, direction: list = None) -> coreFace:
        """
        Flattens the input face such that its center of mass is located at the origin and its normal is pointed in the positive Z axis.

        Parameters
        ----------
        face : coreFace
            The input face.
        originA : coreVertex , optional
            The old location to use as the origin of the movement. If set to None, the center of mass of the input topology is used. The default is None.
        originB : coreVertex , optional
            The new location at which to place the topology. If set to None, the world origin (0,0,0) is used. The default is None.
        direction : list , optional
            The direction, expressed as a list of [X,Y,Z] that signifies the direction of the face. If set to None, the normal at *u* 0.5 and *v* 0.5 is considered the direction of the face. The deafult is None.

        Returns
        -------
        coreFace
            The flattened face.

        """

        def leftMost(vertices, tolerance = 0.0001):
            xCoords = []
            for v in vertices:
                xCoords.append(Vertex.Coordinates(vertices[0])[0])
            minX = min(xCoords)
            lmVertices = []
            for v in vertices:
                if abs(Vertex.Coordinates(vertices[0])[0] - minX) <= tolerance:
                    lmVertices.append(v)
            return lmVertices
        
        def bottomMost(vertices, tolerance = 0.0001):
            yCoords = []
            for v in vertices:
                yCoords.append(Vertex.Coordinates(vertices[0])[1])
            minY = min(yCoords)
            bmVertices = []
            for v in vertices:
                if abs(Vertex.Coordinates(vertices[0])[1] - minY) <= tolerance:
                    bmVertices.append(v)
            return bmVertices

        def vIndex(v, vList, tolerance):
            for i in range(len(vList)):
                if VertexUtility.distance(v, vList[i]) < tolerance:
                    return i+1
            return None
        
        #  rotate cycle path such that it begins with the smallest node
        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:]+path[:n]
        
        #  rotate vertices list so that it begins with the input vertex
        def rotate_vertices(vertices, vertex):
            n = vertices.index(vertex)
            return vertices[n:]+vertices[:n]
        
        from Wrapper.Vertex import Vertex
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        if not isinstance(face, coreFace):
            return None
        if not isinstance(originA, coreVertex):
            originA = Topology.CenterOfMass(face)
        if not isinstance(originB, coreVertex):
            originB = Vertex.ByCoordinates(0,0,0)
        cm = originA
        world_origin = originB
        if not direction or len(direction) < 3:
            direction = Face.NormalAtParameters(face, 0.5, 0.5)
        x1 = Vertex.X(cm)
        y1 = Vertex.Y(cm)
        z1 = Vertex.Z(cm)
        x2 = Vertex.X(cm) + direction[0]
        y2 = Vertex.Y(cm) + direction[1]
        z2 = Vertex.Z(cm) + direction[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        flatFace = Topology.Translate(face, -cm.X(), -cm.Y(), -cm.Z())
        flatFace = Topology.Rotate(flatFace, world_origin, 0, 0, 1, -phi)
        flatFace = Topology.Rotate(flatFace, world_origin, 0, 1, 0, -theta)
        # Ensure flatness. Force Z to be zero
        flatExternalBoundary = Face.ExternalBoundary(flatFace)
        flatFaceVertices = Topology.SubTopologies(flatExternalBoundary, subTopologyType="vertex")
    
        tempVertices = []
        for ffv in flatFaceVertices:
            tempVertices.append(Vertex.ByCoordinates(ffv.X(), ffv.Y(), 0))
        
        temp_v = bottomMost(leftMost(tempVertices))[0]
        tempVertices = rotate_vertices(tempVertices, temp_v)
        flatExternalBoundary = Wire.ByVertices(tempVertices)

        internalBoundaries = Face.InternalBoundaries(flatFace)
        flatInternalBoundaries = []
        for internalBoundary in internalBoundaries:
            ibVertices = Wire.Vertices(internalBoundary)
            tempVertices = []
            for ibVertex in ibVertices:
                tempVertices.append(Vertex.ByCoordinates(ibVertex.X(), ibVertex.Y(), 0))
            temp_v = bottomMost(leftMost(tempVertices))[0]
            tempVertices = rotate_vertices(tempVertices, temp_v)
            flatInternalBoundaries.append(Wire.ByVertices(tempVertices))
        flatFace = Face.ByWires(flatExternalBoundary, flatInternalBoundaries)
        dictionary = Dictionary.ByKeysValues(["xTran", "yTran", "zTran", "phi", "theta"], [cm.X(), cm.Y(), cm.Z(), phi, theta])
        flatFace = Topology.SetDictionary(flatFace, dictionary)
        return flatFace

    @staticmethod
    def Harmonize(face: coreFace) -> coreFace:
        """
        Returns a harmonized version of the input face such that the *u* and *v* origins are always in the upperleft corner.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        coreFace
            The harmonized face.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary

        if not isinstance(face, coreFace):
            return None
        flatFace = Face.Flatten(face)
        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
        vertices = Wire.Vertices(Face.ExternalBoundary(flatFace))
        harmonizedEB = Wire.ByVertices(vertices)
        internalBoundaries = Face.InternalBoundaries(flatFace)
        harmonizedIB = []
        for ib in internalBoundaries:
            ibVertices = Wire.Vertices(ib)
            harmonizedIB.append(Wire.ByVertices(ibVertices))
        harmonizedFace = Face.ByWires(harmonizedEB, harmonizedIB)
        harmonizedFace = Topology.Rotate(harmonizedFace, origin=world_origin, x=0, y=1, z=0, degree=theta)
        harmonizedFace = Topology.Rotate(harmonizedFace, origin=world_origin, x=0, y=0, z=1, degree=phi)
        harmonizedFace = Topology.Translate(harmonizedFace, xTran, yTran, zTran)
        return harmonizedFace

    @staticmethod
    def InternalBoundaries(face: coreFace) -> list:
        """
        Returns the internal boundaries (closed wires) of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        list
            The list of internal boundaries (closed wires).

        """
        if not isinstance(face, coreFace):
            return None
        wires = []
        _ = face.InternalBoundaries(wires)
        return list(wires)

    @staticmethod
    def InternalVertex(face: coreFace, tolerance: float = 0.0001) -> coreVertex:
        """
        Creates a vertex guaranteed to be inside the input face.

        Parameters
        ----------
        face : coreFace
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreVertex
            The created vertex.

        """
        if not isinstance(face, coreFace):
            return None
        v = coreFaceUtility.InternalVertex(face, tolerance)
        return v

    @staticmethod
    def Invert(face: coreFace) -> coreFace:
        """
        Creates a face that is an inverse (mirror) of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        coreFace
            The inverted face.

        """
        from Wrapper.Wire import Wire

        if not isinstance(face, coreFace):
            return None
        eb = Face.ExternalBoundary(face)
        vertices = Wire.Vertices(eb)
        vertices.reverse()
        inverted_wire = Wire.ByVertices(vertices)
        internal_boundaries = Face.InternalBoundaries(face)
        if not internal_boundaries:
            inverted_face = Face.ByWire(inverted_wire)
        else:
            inverted_face = Face.ByWires(inverted_wire, internal_boundaries)
        return inverted_face

    @staticmethod
    def IsCoplanar(faceA: coreFace, faceB: coreFace, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the two input faces are coplanar. Returns False otherwise.

        Parameters
        ----------
        faceA : coreFace
            The first input face.
        faceB : coreFace
            The second input face
        tolerance : float , optional
            The desired tolerance. The deafault is 0.0001.

        Raises
        ------
        Exception
            Raises an exception if the angle between the two input faces cannot be determined.

        Returns
        -------
        bool
            True if the two input faces are coplanar. False otherwise.

        """
        if not isinstance(faceA, coreFace) or not isinstance(faceB, coreFace):
            return None
        dirA = Face.NormalAtParameters(faceA, 0.5, 0.5, "xyz", 3)
        dirB = Face.NormalAtParameters(faceB, 0.5, 0.5, "xyz", 3)
        return Vector.IsCollinear(dirA, dirB, tolerance)
    
    @staticmethod
    def IsInside(face: coreFace, vertex: coreVertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input face. Returns False otherwise.

        Parameters
        ----------
        face : coreFace
            The input face.
        vertex : coreVertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input face. False otherwise.

        """

        # Ray tracing from https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
        def ray_tracing_method(x,y,poly):
            n = len(poly)
            inside = False

            p1x,p1y = poly[0]
            for i in range(n+1):
                p2x,p2y = poly[i % n]
                if y > min(p1y,p2y):
                    if y <= max(p1y,p2y):
                        if x <= max(p1x,p2x):
                            if p1y != p2y:
                                xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x,p1y = p2x,p2y

            return inside

        from Wrapper.Vertex import Vertex
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary

        if not isinstance(face, coreFace):
            return None
        if not isinstance(vertex, coreVertex):
            return None

        world_origin = Vertex.ByCoordinates(0,0,0)
        # Flatten face and vertex
        flatFace = Face.Flatten(face)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        vertex = Topology.Translate(vertex, -xTran, -yTran, -zTran)
        vertex = Topology.Rotate(vertex, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        vertex = Topology.Rotate(vertex, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        # Test if Vertex is hovering above or below face
        if abs(Vertex.Z(vertex)) > tolerance:
            return False

        # Build 2D poly from flat face
        wire = Face.ExternalBoundary(flatFace)
        vertices = Wire.Vertices(wire)
        poly = []
        for v in vertices:
            poly.append([Vertex.X(v), Vertex.Y(v)])

        # Use ray tracing method to test if vertex is inside the face
        status = ray_tracing_method(Vertex.X(vertex), Vertex.Y(vertex), poly)
        # Vertex is not inside
        if not status:
            return status

        # If it is inside, we must check if it is inside a hole in the face
        internal_boundaries = Face.InternalBoundaries(flatFace)
        if len(internal_boundaries) == 0:
            return status
        
        for ib in internal_boundaries:
            vertices = Wire.Vertices(ib)
            poly = []
            for v in vertices:
                poly.append([Vertex.X(v), Vertex.Y(v)])
            status2 = ray_tracing_method(Vertex.X(vertex), Vertex.Y(vertex), poly)
            if status2:
                return False
        return status

    @staticmethod
    def MedialAxis(face: coreFace, resolution: int = 0, externalVertices: bool = False, internalVertices: bool = False, toLeavesOnly: bool = False, angTolerance: float = 0.1, tolerance: float = 0.0001) -> coreWire:
        """
        Returns a wire representing an approximation of the medial axis of the input topology. See https://en.wikipedia.org/wiki/Medial_axis.

        Parameters
        ----------
        face : coreFace
            The input face.
        resolution : int , optional
            The desired resolution of the solution (range is 0: standard resolution to 10: high resolution). This determines the density of the sampling along each edge. The default is 0.
        externalVertices : bool , optional
            If set to True, the external vertices of the face will be connected to the nearest vertex on the medial axis. The default is False.
        internalVertices : bool , optional
            If set to True, the internal vertices of the face will be connected to the nearest vertex on the medial axis. The default is False.
        toLeavesOnly : bool , optional
            If set to True, the vertices of the face will be connected to the nearest vertex on the medial axis only if this vertex is a leaf (end point). Otherwise, it will connect to any nearest vertex. The default is False.
        angTolerance : float , optional
            The desired angular tolerance in degrees for removing collinear edges. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        coreWire
            The medial axis of the input face.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Wire import Wire
        from Wrapper.Shell import Shell
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary

        def touchesEdge(vertex,edges, tolerance=0.0001):
            if not isinstance(vertex, coreVertex):
                return False
            for edge in edges:
                u = Edge.ParameterAtVertex(edge, vertex, mantissa=4)
                if not u:
                    continue
                if 0<u<1:
                    return True
            return False

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

        faceVertices = Face.Vertices(flatFace)
        faceEdges = Face.Edges(flatFace)
        vertices = []
        resolution = 10 - resolution
        resolution = min(max(resolution, 1), 10)
        for e in faceEdges:
            for n in range(resolution, 100, resolution):
                vertices.append(Edge.VertexByParameter(e,n*0.01))
        
        voronoi = Shell.Voronoi(vertices=vertices, face=flatFace)
        voronoiEdges = Shell.Edges(voronoi)

        medialAxisEdges = []
        for e in voronoiEdges:
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)
            svTouchesEdge = touchesEdge(sv, faceEdges, tolerance=tolerance)
            evTouchesEdge = touchesEdge(ev, faceEdges, tolerance=tolerance)
            #connectsToCorners = (Vertex.Index(sv, faceVertices) != None) or (Vertex.Index(ev, faceVertices) != None)
            #if Face.IsInside(flatFace, sv, tolerance=tolerance) and Face.IsInside(flatFace, ev, tolerance=tolerance):
            if not svTouchesEdge and not evTouchesEdge:
                medialAxisEdges.append(e)

        extBoundary = Face.ExternalBoundary(flatFace)
        extVertices = Wire.Vertices(extBoundary)

        intBoundaries = Face.InternalBoundaries(flatFace)
        intVertices = []
        for ib in intBoundaries:
            intVertices = intVertices+Wire.Vertices(ib)
        
        theVertices = []
        if internalVertices:
            theVertices = theVertices+intVertices
        if externalVertices:
            theVertices = theVertices+extVertices

        tempWire = Cluster.SelfMerge(Cluster.ByTopologies(medialAxisEdges))
        if isinstance(tempWire, coreWire) and angTolerance > 0:
            tempWire = Topology.RemoveCollinearEdges(tempWire, angTolerance=angTolerance)
        medialAxisEdges = Wire.Edges(tempWire)
        for v in theVertices:
            nv = Vertex.NearestVertex(v, tempWire, useKDTree=False)

            if isinstance(nv, coreVertex):
                if toLeavesOnly:
                    adjVertices = Topology.AdjacentTopologies(nv, tempWire)
                    if len(adjVertices) < 2:
                        medialAxisEdges.append(Edge.ByVertices([nv, v]))
                else:
                    medialAxisEdges.append(Edge.ByVertices([nv, v]))
        medialAxis = Cluster.SelfMerge(Cluster.ByTopologies(medialAxisEdges))
        if isinstance(medialAxis, coreWire) and angTolerance > 0:
            medialAxis = Topology.RemoveCollinearEdges(medialAxis, angTolerance=angTolerance)
        medialAxis = Topology.Rotate(medialAxis, origin=world_origin, x=0, y=1, z=0, degree=theta)
        medialAxis = Topology.Rotate(medialAxis, origin=world_origin, x=0, y=0, z=1, degree=phi)
        medialAxis = Topology.Translate(medialAxis, xTran, yTran, zTran)
        return medialAxis

    @staticmethod
    def Normal(face: coreFace, outputType: str = "xyz", mantissa: int = 4) -> list:
        """
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : coreFace
            The input face.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The normal vector to the input face. This is computed at the approximate center of the face.

        """
        return Face.NormalAtParameters(face, u=0.5, v=0.5, outputType=outputType, mantissa=mantissa)

    @staticmethod
    def NormalAtParameters(face: coreFace, u: float = 0.5, v: float = 0.5, outputType: str = "xyz", mantissa: int = 4) -> list:
        """
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : coreFace
            The input face.
        u : float , optional
            The *u* parameter at which to compute the normal to the input face. The default is 0.5.
        v : float , optional
            The *v* parameter at which to compute the normal to the input face. The default is 0.5.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The normal vector to the input face.

        """
        returnResult = []
        try:
            coords = coreFaceUtility.NormalAtParameters(face, u, v)
            x = round(coords[0], mantissa)
            y = round(coords[1], mantissa)
            z = round(coords[2], mantissa)
            outputType = list(outputType.lower())
            for axis in outputType:
                if axis == "x":
                    returnResult.append(x)
                elif axis == "y":
                    returnResult.append(y)
                elif axis == "z":
                    returnResult.append(z)
        except:
            returnResult = None
        return returnResult
    
    @staticmethod
    def NormalEdge(face: coreFace, length: float = 1.0) -> coreEdge:
        """
        Returns the normal vector to the input face as an edge with the desired input length. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : coreFace
            The input face.
        length : float , optional
            The desired length of the normal edge. The default is 1.

        Returns
        -------
        coreEdge
            The created normal edge to the input face. This is computed at the approximate center of the face.

        """
        return Face.NormalEdgeAtParameters(face, u=0.5, v=0.5, length=length)

    @staticmethod
    def NormalEdgeAtParameters(face: coreFace, u: float = 0.5, v: float = 0.5, length: float = 1.0) -> coreEdge:
        """
        Returns the normal vector to the input face as an edge with the desired input length. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : coreFace
            The input face.
        u : float , optional
            The *u* parameter at which to compute the normal to the input face. The default is 0.5.
        v : float , optional
            The *v* parameter at which to compute the normal to the input face. The default is 0.5.
        length : float , optional
            The desired length of the normal edge. The default is 1.

        Returns
        -------
        coreEdge
            The created normal edge to the input face. This is computed at the approximate center of the face.

        """
        from Wrapper.Edge import Edge
        from Wrapper.Topology import Topology
        if not isinstance(face, coreFace):
            return None
        sv = Face.VertexByParameters(face=face, u=u, v=v)
        vec = Face.NormalAtParameters(face, u=u, v=v)
        ev = Topology.TranslateByDirectionDistance(sv, vec, length)
        return Edge.ByVertices([sv, ev])

    @staticmethod
    def Planarize(face: coreFace, origin: coreVertex = None, direction: list = None) -> coreFace:
        """
        Planarizes the input face such that its center of mass is located at the input origin and its normal is pointed in the input direction.

        Parameters
        ----------
        face : coreFace
            The input face.
        origin : coreVertex , optional
            The old location to use as the origin of the movement. If set to None, the center of mass of the input face is used. The default is None.
        direction : list , optional
            The direction, expressed as a list of [X,Y,Z] that signifies the direction of the face. If set to None, the normal at *u* 0.5 and *v* 0.5 is considered the direction of the face. The deafult is None.

        Returns
        -------
        coreFace
            The planarized face.

        """

        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary

        if not isinstance(face, coreFace):
            return None
        if not isinstance(origin, coreVertex):
            origin = Topology.CenterOfMass(face)
        if not isinstance(direction, list):
            direction = Face.NormalAtParameters(face, 0.5, 0.5)
        flatFace = Face.Flatten(face, oldLocation=origin, direction=direction)

        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        planarizedFace = Topology.Rotate(flatFace, origin=world_origin, x=0, y=1, z=0, degree=theta)
        planarizedFace = Topology.Rotate(planarizedFace, origin=world_origin, x=0, y=0, z=1, degree=phi)
        planarizedFace = Topology.Translate(planarizedFace, xTran, yTran, zTran)
        return planarizedFace

    @staticmethod
    def Project(faceA: coreFace, faceB: coreFace, direction : list = None, mantissa: int = 4) -> coreFace:
        """
        Creates a projection of the first input face unto the second input face.

        Parameters
        ----------
        faceA : coreFace
            The face to be projected.
        faceB : coreFace
            The face unto which the first input face will be projected.
        direction : list, optional
            The vector direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        coreFace
            The projected Face.

        """

        from Wrapper.Wire import Wire

        if not faceA:
            return None
        if not isinstance(faceA, coreFace):
            return None
        if not faceB:
            return None
        if not isinstance(faceB, coreFace):
            return None

        eb = faceA.ExternalBoundary()
        ib_list = []
        _ = faceA.InternalBoundaries(ib_list)
        p_eb = Wire.Project(wire=eb, face = faceB, direction=direction, mantissa=mantissa)
        p_ib_list = []
        for ib in ib_list:
            temp_ib = Wire.Project(wire=ib, face = faceB, direction=direction, mantissa=mantissa)
            if temp_ib:
                p_ib_list.append(temp_ib)
        return Face.ByWires(p_eb, p_ib_list)

    @staticmethod
    def Rectangle(origin: coreVertex = None, width: float = 1.0, length: float = 1.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreFace:
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : coreVertex, optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0,0,0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreFace
            The created face.

        """
        from Wrapper.Wire import Wire
        wire = Wire.Rectangle(origin=origin, width=width, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, coreWire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Square(origin: coreVertex = None, size: float = 1.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreFace:
        """
        Creates a square.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the square. The default is None which results in the square being placed at (0,0,0).
        size : float , optional
            The size of the square. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the square. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreFace
            The created square.

        """
        return Face.Rectangle(origin = origin, width = size, length = size, direction = direction, placement = placement, tolerance = tolerance)
    
    @staticmethod
    def Star(origin: coreVertex = None, radiusA: float = 1.0, radiusB: float = 0.4, rays: int = 5, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreFace:
        """
        Creates a star.

        Parameters
        ----------
        origin : coreVertex, optional
            The location of the origin of the star. The default is None which results in the star being placed at (0,0,0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 5.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreFace
            The created face.

        """
        from Wrapper.Wire import Wire
        wire = Wire.Star(origin=origin, radiusA=radiusA, radiusB=radiusB, rays=rays, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, coreWire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Trapezoid(origin: coreVertex = None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreFace:
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : coreVertex, optional
            The location of the origin of the trapezoid. The default is None which results in the trapezoid being placed at (0,0,0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. The default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. The default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. The default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. The default is 0.0.
        length : float , optional
            The length of the trapezoid. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreFace
            The created trapezoid.

        """
        from Wrapper.Wire import Wire
        wire = Wire.Trapezoid(origin=origin, widthA=widthA, widthB=widthB, offsetA=offsetA, offsetB=offsetB, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not isinstance(wire, coreWire):
            return None
        return Face.ByWire(wire)

    @staticmethod
    def Triangulate(face:coreFace) -> list:
        """
        Triangulates the input face and returns a list of faces.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        list
            The list of triangles of the input face.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary

        if not isinstance(face, coreFace):
            return None
        flatFace = Face.Flatten(face)
        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
    
        faceTriangles = []
        for i in range(0,5,1):
            try:
                _ = coreFaceUtility.Triangulate(flatFace, float(i)*0.1, faceTriangles)
                break
            except:
                continue
        if len(faceTriangles) < 1:
            return [face]
        finalFaces = []
        for f in faceTriangles:
            f = Topology.Rotate(f, origin=world_origin, x=0, y=1, z=0, degree=theta)
            f = Topology.Rotate(f, origin=world_origin, x=0, y=0, z=1, degree=phi)
            f = Topology.Translate(f, xTran, yTran, zTran)
            if Face.Angle(face, f) > 90:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                f = coreFace.by_external_boundary(wire)
                finalFaces.append(f)
            else:
                finalFaces.append(f)
        return finalFaces

    @staticmethod
    def TrimByWire(face: coreFace, wire: coreWire, reverse: bool = False) -> coreFace:
        """
        Trims the input face by the input wire.

        Parameters
        ----------
        face : coreFace
            The input face.
        wire : coreWire
            The input wire.
        reverse : bool , optional
            If set to True, the effect of the trim will be reversed. The default is False.

        Returns
        -------
        coreFace
            The resulting trimmed face.

        """
        if not isinstance(face, coreFace):
            return None
        if not isinstance(wire, coreWire):
            return face
        trimmed_face = coreFaceUtility.TrimByWire(face, wire, False)
        if reverse:
            trimmed_face = face.Difference(trimmed_face)
        return trimmed_face
    
    @staticmethod
    def UnFlatten(face: coreFace, dictionary: coreDictionary):
        from Wrapper.Vertex import Vertex
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        theta = Dictionary.ValueAtKey(dictionary, "theta")
        phi = Dictionary.ValueAtKey(dictionary, "phi")
        xTran = Dictionary.ValueAtKey(dictionary, "xTran")
        yTran = Dictionary.ValueAtKey(dictionary, "yTran")
        zTran = Dictionary.ValueAtKey(dictionary, "zTran")
        newFace = Topology.Rotate(face, origin=Vertex.Origin(), x=0, y=1, z=0, degree=theta)
        newFace = Topology.Rotate(newFace, origin=Vertex.Origin(), x=0, y=0, z=1, degree=phi)
        newFace = Topology.Translate(newFace, xTran, yTran, zTran)
        return newFace
    
    @staticmethod
    def VertexByParameters(face: coreFace, u: float = 0.5, v: float = 0.5) -> coreVertex:
        """
        Creates a vertex at the *u* and *v* parameters of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.
        u : float , optional
            The *u* parameter of the input face. The default is 0.5.
        v : float , optional
            The *v* parameter of the input face. The default is 0.5.

        Returns
        -------
        vertex : topologic vertex
            The created vertex.

        """
        if not isinstance(face, coreFace):
            return None
        return coreFaceUtility.VertexAtParameters(face, u, v)
    
    @staticmethod
    def VertexParameters(face: coreFace, vertex: coreVertex, outputType: str = "uv", mantissa: int = 4) -> list:
        """
        Returns the *u* and *v* parameters of the input face at the location of the input vertex.

        Parameters
        ----------
        face : coreFace
            The input face.
        vertex : coreVertex
            The input vertex.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "uv". It is case insensitive. The default is "uv".
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The list of *u* and/or *v* as specified by the outputType input.

        """
        if not isinstance(face, coreFace):
            return None
        if not isinstance(vertex, coreVertex):
            return None
        params = coreFaceUtility.ParametersAtVertex(face, vertex)
        u = round(params[0], mantissa)
        v = round(params[1], mantissa)
        outputType = list(outputType.lower())
        returnResult = []
        for param in outputType:
            if param == "u":
                returnResult.append(u)
            elif param == "v":
                returnResult.append(v)
        return returnResult

    @staticmethod
    def Vertices(face: coreFace) -> list:
        """
        Returns the vertices of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(face, coreFace):
            return None
        vertices = []
        _ = face.Vertices(None, vertices)
        return vertices
    
    @staticmethod
    def Wire(face: coreFace) -> coreWire:
        """
        Returns the external boundary (closed wire) of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        coreWire
            The external boundary of the input face.

        """
        return face.ExternalBoundary()
    
    @staticmethod
    def Wires(face: coreFace) -> list:
        """
        Returns the wires of the input face.

        Parameters
        ----------
        face : coreFace
            The input face.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(face, coreFace):
            return None
        wires = []
        _ = face.Wires(None, wires)
        return wires
