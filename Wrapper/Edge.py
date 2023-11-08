# Core
from Core.Topology import Topology as coreTopology
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Shell import Shell as coreShell
from Core.Cluster import Cluster as coreCluster

from Core.Dictionary import Dictionary as coreDictionary
from Core.Utilities.TopologicUtilities import VertexUtility, EdgeUtility, FaceUtility

# Wrapper
from Wrapper.Vertex import Vertex
from Wrapper.Vector import Vector

class Edge():
    @staticmethod
    def Angle(edgeA: coreEdge, edgeB: coreEdge, mantissa: int = 4, bracket: bool = False) -> float:
        """
        Returns the angle in degrees between the two input edges.

        Parameters
        ----------
        edgeA : coreEdge
            The first input edge.
        edgeB : topologic Edge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.
        bracket : bool
            If set to True, the returned angle is bracketed between 0 and 180. The default is False.

        Returns
        -------
        float
            The angle in degrees between the two input edges.

        """

        if not isinstance(edgeA, coreEdge) or not isinstance(edgeB, coreEdge):
            return None
        dirA = Edge.Direction(edgeA, mantissa)
        dirB = Edge.Direction(edgeB, mantissa)
        ang = Vector.Angle(dirA, dirB)
        if bracket:
            if ang > 90:
                ang = 180 - ang
        return round(ang, mantissa)

    @staticmethod
    def Bisect(edgeA: coreEdge, edgeB: coreEdge, length: float = 1.0, placement: int = 0, tolerance: float = 0.0001) -> coreEdge:
        """
        Creates a bisecting edge between edgeA and edgeB.

        Parameters
        ----------
        edgeA : coreEdge
            The first topologic Edge.
        edgeB : topologic Edge
            The second topologic Edge.
        length : float , optional
            The desired length of the bisecting edge. The default is 1.0.
        placement : int , optional
            The desired placement of the bisecting edge.
            If set to 0, the bisecting edge centroid will be placed at the end vertex of the first edge.
            If set to 1, the bisecting edge start vertex will be placed at the end vertex of the first edge.
            If set to 2, the bisecting edge end vertex will be placed at the end vertex of the first edge.
            If set to any number other than 0, 1, or 2, the bisecting edge centroid will be placed at the end vertex of the first edge. The default is 0.
        tolerance : float , optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.

        Returns
        -------
        coreEdge
            The created bisecting edge.

        """
        if not isinstance(edgeA, coreEdge) or not isinstance(edgeB, coreEdge):
            return None
        if Edge.Length(edgeA) < tolerance or Edge.Length(edgeB) < tolerance:
            return None
        from Wrapper.Topology import Topology
        v1 = Edge.VertexByDistance(edgeA, -1, edgeA.end_vertex(), tolerance=0.0001)
        newEdgeA = Edge.ByVertices([v1, edgeA.end_vertex()])
        v1 = Edge.VertexByDistance(edgeB, 1, edgeB.start_vertex(), tolerance=0.0001)
        newEdgeB = Edge.ByVertices([edgeB.start_vertex(), v1])
        newEdgeB: coreEdge = Topology.Place(newEdgeB, newEdgeB.start_vertex(), newEdgeA.start_vertex())
        bisectingEdge = Edge.ByVertices([newEdgeA.end_vertex(), newEdgeB.end_vertex()])
        bEdgeLength = Edge.Length(bisectingEdge)
        bisectingEdge: coreEdge = Topology.Scale(bisectingEdge, bisectingEdge.start_vertex(), 1/bEdgeLength, 1/bEdgeLength, 1/bEdgeLength)
        if length != 1.0 and length > tolerance:
            bisectingEdge = Topology.Scale(bisectingEdge, bisectingEdge.start_vertex(), length, length, length)
        newLocation = edgeA.end_vertex()
        if placement == 2:
            oldLocation = bisectingEdge.end_vertex()
        elif placement == 1:
            oldLocation = bisectingEdge.start_vertex()
        else:
            oldLocation = bisectingEdge.center_of_mass()
        bisectingEdge = Topology.Place(bisectingEdge, oldLocation, newLocation)
        return bisectingEdge

    @staticmethod
    def ByFaceNormal(face: coreFace, origin: coreVertex = None, length: float = 1.0) -> coreEdge:
        """
        Creates a straight edge representing the normal to the input face.

        Parameters
        ----------
        face : coreFace
            The input face
        origin : toopologic.Vertex , optional
            The desired origin of the edge. If set to None, the centroid of the face is chosen as the origin of the edge. The default is None.
        length : float , optional
            The desired length of the edge. The default is 1.

        Returns
        -------
        edge : coreEdge
            The created edge.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        edge = None
        if not isinstance(face, coreFace):
            return None
        if not isinstance(origin, coreVertex):
            origin = Topology.Centroid(face)
        
        n = Face.Normal(face)
        v2 = Topology.Translate(origin, n[0], n[1], n[2])
        edge = coreEdge.by_start_vertex_end_vertex(origin, v2)
        edge = Edge.SetLength(edge, length, bothSides=False)
        return edge

    @staticmethod
    def ByOffset2D(edge: coreEdge, offset: float = 1.0, tolerance: float = 0.0001) -> coreEdge:
        """
        Creates and edge offset from the input edge. This method is intended for edges that are in the XY plane.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        offset : float , optional
            The desired offset. The default is 1.
        tolerance : float , optiona
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreEdge
            An edge offset from the input edge.

        """
        from Wrapper.Topology import Topology
        n = Edge.Normal2D(edge)
        n = Vector.Normalize(n)
        n = Vector.Multiply(n, offset, tolerance)
        edge2 = Topology.Translate(edge, n[0], n[1], n[2])
        return edge2


    @staticmethod
    def ByStartVertexEndVertex(vertexA: coreVertex, vertexB: coreVertex, tolerance: float = 0.0001) -> coreEdge:
        """
        Creates a straight edge that connects the input vertices.

        Parameters
        ----------
        vertexA : coreVertex
            The first input vertex. This is considered the start vertex.
        vertexB : toopologic.Vertex
            The second input vertex. This is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an Edge can be created. The default is 0.0001.

        Returns
        -------
        edge : coreEdge
            The created edge.

        """
        edge = None
        if not isinstance(vertexA, coreVertex):
            return None
        if not isinstance(vertexB, coreVertex):
            return None
        if coreTopology.is_same(vertexA, vertexB):
            return None
        if VertexUtility.distance(vertexA, vertexB) < tolerance:
            return None
        try:
            edge = coreEdge.by_start_vertex_end_vertex(vertexA, vertexB)
        except:
            edge = None
        return edge
    
    @staticmethod
    def ByVertices(vertices: list, tolerance: float = 0.0001) -> coreEdge:
        """
        Creates a straight edge that connects the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices. The first item is considered the start vertex and the last item is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. The default is 0.0001.

        Returns
        -------
        coreEdge
            The created edge.

        """
        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertexList) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance)
    
    @staticmethod
    def ByVerticesCluster(cluster: coreCluster, tolerance: float = 0.0001) -> coreEdge:
        """
        Creates a straight edge that connects the input cluster of vertices.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster of vertices. The first item is considered the start vertex and the last item is considered the end vertex.
        tolerance : float , optional
            The desired tolerance to decide if an edge can be created. The default is 0.0001.

        Returns
        -------
        coreEdge
            The created edge.

        """
        from Wrapper.Cluster import Cluster
        if not isinstance(cluster, coreCluster):
            return None
        vertices = Cluster.Vertices(cluster)
        vertexList = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertexList) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertexList[0], vertexList[-1], tolerance)

    @staticmethod
    def Direction(edge: coreEdge, mantissa: int = 4) -> list:
        """
        Returns the direction of the input edge expressed as a list of three numbers.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The direction of the input edge.

        """

        from Wrapper.Vector import Vector
        from Core.Vertex import Vertex as coreVertex

        if not isinstance(edge, coreEdge):
            return None
        ev: coreVertex = edge.end_vertex()
        sv: coreVertex = edge.start_vertex()
        x = ev.x() - sv.x()
        y = ev.y() - sv.y()
        z = ev.z() - sv.z()
        uvec = Vector.Normalize([x,y,z])
        x = round(uvec[0], mantissa)
        y = round(uvec[1], mantissa)
        z = round(uvec[2], mantissa)
        return [x, y, z]
    
    @staticmethod
    def EndVertex(edge: coreEdge) -> coreVertex:
        """
        Returns the end vertex of the input edge.

        Parameters
        ----------
        edge : coreEdge
            The input edge.

        Returns
        -------
        coreVertex
            The end vertex of the input edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        vert = None
        try:
            vert = edge.EndVertex()
        except:
            vert = None
        return vert
    
    @staticmethod
    def Extend(edge: coreEdge, distance: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001) -> coreEdge:
        """
        Extends the input edge by the input distance.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        distance : float , optional
            The offset distance. The default is 1.
        bothSides : bool , optional
            If set to True, the edge will be extended by half the distance at each end. The default is False.
        reverse : bool , optional
            If set to True, the edge will be extended from its start vertex. Otherwise, it will be extended from its end vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreEdge
            The extended edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        distance = abs(distance)
        if distance < tolerance:
            return edge
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        if bothSides:
            sve = Edge.VertexByDistance(edge, distance=-distance*0.5, origin=sv, tolerance=tolerance)
            eve = Edge.VertexByDistance(edge, distance=distance*0.5, origin=ev, tolerance=tolerance)
        elif reverse:
            sve = Edge.VertexByDistance(edge, distance=-distance, origin=sv, tolerance=tolerance)
            eve = Edge.EndVertex(edge)
        else:
            sve = Edge.StartVertex(edge)
            eve = Edge.VertexByDistance(edge, distance=distance, origin=ev, tolerance=tolerance)
        return Edge.ByVertices([sve, eve])

    @staticmethod
    def ExtendToEdge2D(edgeA: coreEdge, edgeB: coreEdge) -> coreEdge:
        """
        Extends the first input edge to meet the second input edge. This works only in the XY plane. Z coordinates are ignored.

        Parameters
        ----------
        edgeA : coreEdge
            The first input edge.
        edgeB : coreEdge
            The second input edge.

        Returns
        -------
        coreEdge
            The extended edge.

        """
        from Wrapper.Topology import Topology
        if not isinstance(edgeA, coreEdge):
            return None
        if not isinstance(edgeB, coreEdge):
            return None
        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        intVertex = Edge.Intersect2D(edgeA, edgeB)
        if intVertex and not (Topology.IsInside(edgeA, intVertex)):
            e1 = Edge.ByVertices([sva, intVertex])
            e2 = Edge.ByVertices([eva, intVertex])
            l1 = Edge.Length(e1)
            l2 = Edge.Length(e2)
            if l1 > l2:
                return e1
            else:
                return e2
        return None
    @staticmethod
    def Index(edge: coreEdge, edges: list, strict: bool = False, tolerance: float = 0.0001) -> int:
        """
        Returns index of the input edge in the input list of edges

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        edges : list
            The input list of edges.
        strict : bool , optional
            If set to True, the edge must be strictly identical to the one found in the list. Otherwise, a distance comparison is used. The default is False.
        tolerance : float , optional
            The tolerance for computing if the input edge is identical to an edge from the list. The default is 0.0001.

        Returns
        -------
        int
            The index of the input edge in the input list of edges.

        """
        from Wrapper.Topology import Topology
        if not isinstance(edge, coreEdge):
            return None
        if not isinstance(edges, list):
            return None
        edges = [e for e in edges if isinstance(e, coreEdge)]
        if len(edges) == 0:
            return None
        sva = Edge.StartVertex(edge)
        eva = Edge.EndVertex(edge)
        for i in range(len(edges)):
            if strict:
                if Topology.IsSame(edge, edges[i]):
                    return i
            else:
                svb = Edge.StartVertex(edges[i])
                evb = Edge.EndVertex(edges[i])
                dsvsv = Vertex.Distance(sva, svb)
                devev = Vertex.Distance(eva, evb)
                if dsvsv < tolerance and devev < tolerance:
                    return i
                dsvev = Vertex.Distance(sva, evb)
                devsv = Vertex.Distance(eva, svb)
                if dsvev < tolerance and devsv < tolerance:
                    return i
        return None

    @staticmethod
    def Intersect2D(edgeA: coreEdge, edgeB: coreEdge) -> coreVertex:
        """
        Returns the intersection of the two input edges as a coreVertex. This works only in the XY plane. Z coordinates are ignored.

        Parameters
        ----------
        edgeA : coreEdge
            The first input edge.
        edgeB : coreEdge
            The second input edge.

        Returns
        -------
        coreVertex
            The intersection of the two input edges.

        """
        if not isinstance(edgeA, coreEdge):
            print("Intersect2D: edgeA is not a coreEdge")
            return None
        if not isinstance(edgeB, coreEdge):
            print("Intersect2D: edgeB is not a coreEdge")
            return None
        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        svb = Edge.StartVertex(edgeB)
        evb = Edge.EndVertex(edgeB)
        # Line AB represented as a1x + b1y = c1
        a1 = Vertex.Y(eva) - Vertex.Y(sva)
        b1 = Vertex.X(sva) - Vertex.X(eva)
        c1 = a1*(Vertex.X(sva)) + b1*(Vertex.Y(sva))
 
        # Line CD represented as a2x + b2y = c2
        a2 = Vertex.Y(evb) - Vertex.Y(svb)
        b2 = Vertex.X(svb) - Vertex.X(evb)
        c2 = a2*(Vertex.X(svb)) + b2*(Vertex.Y(svb))
 
        determinant = a1*b2 - a2*b1
 
        if (determinant == 0):
            # The lines are parallel. This is simplified
            # by returning a pair of FLT_MAX
            return None
        else:
            x = (b2*c1 - b1*c2)/determinant
            y = (a1*c2 - a2*c1)/determinant
            return Vertex.ByCoordinates(x,y,0)


    @staticmethod
    def IsCollinear(edgeA: coreEdge, edgeB: coreEdge, mantissa: int = 4, angTolerance: float = 0.1, tolerance: float = 0.0001) -> bool:
        """
        Return True if the two input edges are collinear. Returns False otherwise.

        Parameters
        ----------
        edgeA : coreEdge
            The first input edge.
        edgeB : coreEdge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.
        angTolerance : float , optional
            The angular tolerance used for the test. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two edges are collinear. False otherwise.

        """
        if not isinstance(edgeA, coreEdge) or not isinstance(edgeB, coreEdge):
            return None
        ang = Edge.Angle(edgeA, edgeB, mantissa=mantissa, bracket=True)
        svA = Edge.StartVertex(edgeA)
        evA = Edge.EndVertex(edgeA)
        svB = Edge.StartVertex(edgeB)
        evB = Edge.EndVertex(edgeB)
        d1 = Vertex.Distance(svA, svB)
        d2 = Vertex.Distance(svA, evB)
        d3 = Vertex.Distance(evA, svB)
        d4 = Vertex.Distance(evA, evB)
        if (d1 < tolerance or d2 < tolerance or d3 < tolerance or d4 < tolerance) and (abs(ang) < angTolerance or (abs(180 - ang) < angTolerance)):
            return True
        return False
    
    @staticmethod
    def IsParallel(edgeA: coreEdge, edgeB: coreEdge, mantissa: int = 4, angTolerance: float = 0.1) -> bool:
        """
        Return True if the two input edges are parallel. Returns False otherwise.

        Parameters
        ----------
        edgeA : coreEdge
            The first input edge.
        edgeB : coreEdge
            The second input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.
        angTolerance : float , optional
            The angular tolerance used for the test. The default is 0.1.

        Returns
        -------
        bool
            True if the two edges are collinear. False otherwise.

        """
        if not isinstance(edgeA, coreEdge) or not isinstance(edgeB, coreEdge):
            return None
        ang = Edge.Angle(edgeA, edgeB, mantissa=mantissa, bracket=True)
        if abs(ang) < angTolerance or abs(180 - ang) < angTolerance:
            return True
        return False

    @staticmethod
    def Length(edge: coreEdge, mantissa: int = 4) -> float:
        """
        Returns the length of the input edge.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The length of the input edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        length = None
        try:
            length = round(EdgeUtility.length(edge), mantissa)
        except:
            length = None
        return length

    @staticmethod
    def Normal2D(edge: coreEdge) -> list:
        """
        Returns the normal (perpendicular) vector to the input edge. This method is intended for edges that are in the XY plane. Z is assumed to be zero and ignored.

        Parameters
        ----------
        edge : coreEdge
            The input edge.

        Returns
        -------
        list
            The normal (perpendicular ) vector to the input edge.

        """
        
        from Wrapper.Vector import Vector

        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        x1 = Vertex.X(sv)
        y1 = Vertex.Y(sv)

        x2 = Vertex.X(ev)
        y2 = Vertex.Y(ev)

        dx = x2 - x1
        dy = y2 - y1
        return Vector.Normalize([-dy, dx, 0])

    @staticmethod
    def Normalize(edge: coreEdge, useEndVertex: bool = False) -> coreEdge:
        """
        Creates a normalized edge that has the same direction as the input edge, but a length of 1.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        useEndVertex : bool , optional
            If True the normalized edge end vertex will be placed at the end vertex of the input edge. Otherwise, the normalized edge start vertex will be placed at the start vertex of the input edge. The default is False.

        Returns
        -------
        coreEdge
            The normalized edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        if not useEndVertex:
            sv = edge.StartVertex()
            ev = Edge.VertexByDistance(edge, 1.0, edge.StartVertex())
        else:
            sv = Edge.VertexByDistance(edge, 1.0, edge.StartVertex())
            ev = edge.EndVertex()
        return Edge.ByVertices([sv, ev])

    @staticmethod
    def ParameterAtVertex(edge: coreEdge, vertex: coreVertex, mantissa: int = 4) -> float:
        """
        Returns the *u* parameter along the input edge based on the location of the input vertex.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        vertex : coreVertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The *u* parameter along the input edge based on the location of the input vertex.

        """
        if not isinstance(edge, coreEdge) or not isinstance(vertex, coreVertex):
            return None
        parameter = None
        try:
            parameter = coreEdgeUtility.ParameterAtPoint(edge, vertex)
        except:
            return None
        return round(parameter, mantissa)

    @staticmethod
    def Reverse(edge: coreEdge) -> coreEdge:
        """
        Creates an edge that has the reverse direction of the input edge.

        Parameters
        ----------
        edge : coreEdge
            The input edge.

        Returns
        -------
        coreEdge
            The reversed edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        return Edge.ByVertices([edge.end_vertex(), edge.start_vertex()])
    
    @staticmethod
    def SetLength(edge: coreEdge , length: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001) -> coreEdge:
        """
        Returns an edge with the new length in the same direction as the input edge.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        length : float , optional
            The desired length of the edge. The default is 1.
        bothSides : bool , optional
            If set to True, the edge will be offset symmetrically from each end. The default is True.
        reverse : bool , optional
            If set to True, the edge will be offset from its start vertex. Otherwise, it will be offset from its end vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreEdge
            The extended edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        distance = (length - Edge.Length(edge))
        if distance > 0:
            return Edge.Extend(edge=edge, distance=distance, bothSides=bothSides, reverse=reverse, tolerance=tolerance)
        return Edge.Trim(edge=edge, distance=distance, bothSides=bothSides, reverse=reverse, tolerance=tolerance)

    @staticmethod
    def StartVertex(edge: coreEdge) -> coreVertex:
        """
        Returns the start vertex of the input edge.

        Parameters
        ----------
        edge : coreEdge
            The input edge.

        Returns
        -------
        coreVertex
            The start vertex of the input edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        vert = None
        try:
            vert = edge.StartVertex()
        except:
            vert = None
        return vert

    @staticmethod
    def Trim(edge: coreEdge, distance: float = 0.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001) -> coreEdge:
        """
        Trims the input edge by the input distance.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        distance : float , optional
            The offset distance. The default is 0.
        bothSides : bool , optional
            If set to True, the edge will be trimmed by half the distance at each end. The default is False.
        reverse : bool , optional
            If set to True, the edge will be trimmed from its start vertex. Otherwise, it will be trimmed from its end vertex. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreEdge
            The trimmed edge.

        """
        if not isinstance(edge, coreEdge):
            return None
        distance = abs(distance)
        if distance < tolerance:
            return edge
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        if bothSides:
            sve = Edge.VertexByDistance(edge, distance=distance*0.5, origin=sv, tolerance=tolerance)
            eve = Edge.VertexByDistance(edge, distance=-distance*0.5, origin=ev, tolerance=tolerance)
        elif reverse:
            sve = Edge.VertexByDistance(edge, distance=distance, origin=sv, tolerance=tolerance)
            eve = Edge.EndVertex(edge)
        else:
            sve = Edge.StartVertex(edge)
            eve = Edge.VertexByDistance(edge, distance=-distance, origin=ev, tolerance=tolerance)
        return Edge.ByVertices([sve, eve])

    @staticmethod
    def TrimByEdge2D(edgeA: coreEdge, edgeB: coreEdge, reverse: bool = False) -> coreEdge:
        """
        Trims the first input edge by the second input edge. This works only in the XY plane. Z coordinates are ignored.

        Parameters
        ----------
        edgeA : coreEdge
            The first input edge.
        edgeB : coreEdge
            The second input edge.

        Returns
        -------
        coreEdge
            The trimmed edge.

        """
        from Wrapper.Topology import Topology
        if not isinstance(edgeA, coreEdge):
            return None
        if not isinstance(edgeB, coreEdge):
            return None
        sva = Edge.StartVertex(edgeA)
        eva = Edge.EndVertex(edgeA)
        intVertex = Edge.Intersect2D(edgeA, edgeB)
        if intVertex and (Topology.IsInside(edgeA, intVertex)):
            if reverse:
                return Edge.ByVertices([eva, intVertex])
            else:
                return Edge.ByVertices([sva, intVertex])
        return edgeA

    @staticmethod
    def VertexByDistance(edge: coreEdge, distance: float = 0.0, origin: coreVertex = None, tolerance: float = 0.0001) -> coreVertex:
        """
        Creates a vertex along the input edge offset by the input distance from the input origin.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        distance : float , optional
            The offset distance. The default is 0.
        origin : coreVertex , optional
            The origin of the offset distance. If set to None, the origin will be set to the start vertex of the input edge. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreVertex
            The created vertex.

        """

        if not isinstance(edge, coreEdge):
            return None
        if not origin:
            origin = edge.start_vertex()
        if not isinstance(origin, coreVertex):
            return None
        sv: coreVertex = edge.start_vertex()
        ev: coreVertex = edge.end_vertex()
        vx = ev.x() - sv.x()
        vy = ev.y() - sv.y()
        vz = ev.z() - sv.z()
        vector = Vector.Normalize([vx, vy, vz])
        vector = Vector.Multiply(vector, distance, tolerance)
        return coreVertex.by_coordinates(origin.x()+vector[0], origin.y()+vector[1], origin.z()+vector[2])
    
    @staticmethod
    def VertexByParameter(edge: coreVertex, parameter: float = 0.0) -> coreVertex:
        """
        Creates a vertex along the input edge offset by the input *u* parameter.

        Parameters
        ----------
        edge : coreEdge
            The input edge.
        parameter : float , optional
            The *u* parameter along the input topologic Edge. A parameter of 0 returns the start vertex. A parameter of 1 returns the end vertex. The default is 0.

        Returns
        -------
        coreVertex
            The created vertex.

        """
        if not isinstance(edge, coreEdge):
            return None
        vertex = None
        if parameter == 0:
            vertex = edge.start_vertex()
        elif parameter == 1:
            vertex = edge.end_vertex()
        else:
            try:
                vertex = coreEdgeUtility.PointAtParameter(edge, parameter)
            except:
                vertex = None
        return vertex

    @staticmethod
    def Vertices(edge: coreEdge) -> list:
        """
        Returns the list of vertices of the input edge.

        Parameters
        ----------
        edge : coreEdge
            The input edge.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(edge, coreEdge):
            return None
        vertices = edge.vertices(None)
        return vertices