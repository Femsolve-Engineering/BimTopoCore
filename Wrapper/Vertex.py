
# Core
from Core.Topology import Topology as coreTopology
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Shell import Shell as coreShell
from Core.Cell import Cell as coreCell
from Core.Cluster import Cluster as coreCluster
from Core.Utilities.TopologicUtilities import VertexUtility

# Wrapper
from Wrapper.Face import Face
from Wrapper.Topology import Topology
import collections

class Vertex(Topology):
    @staticmethod
    def AreIpsilateral(vertices: list, face: coreFace) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : coreFace
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        def check(dot_productA, pointB, pointC, normal):
            # Calculate the dot products of the vectors from the surface point to each of the input points.
            dot_productB = (pointB[0] - pointC[0]) * normal[0] + \
                        (pointB[1] - pointC[1]) * normal[1] + \
                        (pointB[2] - pointC[2]) * normal[2]

            # Check if both points are on the same side of the surface.
            if dot_productA * dot_productB > 0:
                return True

            # Check if both points are on opposite sides of the surface.
            elif dot_productA * dot_productB < 0:
                return False

            # Otherwise, at least one point is on the surface.
            else:
                return True
    
        from Wrapper.Vertex import Vertex
        from Wrapper.Face import Face

        if not isinstance(face, coreFace):
            return None
        vertexList = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertexList) < 2:
            return None
        pointA = Vertex.Coordinates(vertexList[0])
        pointC = Vertex.Coordinates(Face.VertexByParameters(face, 0.5, 0.5))
        normal = Face.Normal(face)
        dot_productA = (pointA[0] - pointC[0]) * normal[0] + \
                        (pointA[1] - pointC[1]) * normal[1] + \
                        (pointA[2] - pointC[2]) * normal[2]
        for i in range(1, len(vertexList)):
            pointB = Vertex.Coordinates(vertexList[i])
            if not check(dot_productA, pointB, pointC, normal):
                return False
        return True
    
    @staticmethod
    def AreIpsilateralCluster(cluster: coreCluster, face: coreFace) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        cluster : coreCluster
            The input list of vertices.
        face : coreFace
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        from Wrapper.Topology import Topology
        if not isinstance(cluster, coreTopology):
            return None
        vertices = Topology.SubTopologies(cluster, subTopologyType="vertex")
        return Vertex.AreIpsilateral(vertices, face)
    
    @staticmethod
    def AreOnSameSide(vertices: list, face: Face) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        face : coreFace
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        return Vertex.AreIpsilateral(vertices, face)

    @staticmethod
    def AreOnSameSideCluster(cluster: coreCluster, face: coreFace) -> bool:
        """
        Returns True if the two input vertices are on the same side of the input face. Returns False otherwise. If at least one of the vertices is on the face, this method return True.

        Parameters
        ----------
        cluster : coreCluster
            The input list of vertices.
        face : coreFace
            The input face

        Returns
        -------
        bool
            True if the input vertices are on the same side of the face. False otherwise. If at least one of the vertices is on the face, this method return True.

        """
        from Wrapper.Topology import Topology
        if not isinstance(cluster, coreTopology):
            return None
        vertices = Topology.SubTopologies(cluster, subTopologyType="vertex")
        return Vertex.AreIpsilateral(vertices, face)

    @staticmethod
    def ByCoordinates(x: float = 0, y: float = 0, z: float = 0) -> coreVertex:
        """
        Creates a vertex at the coordinates specified by the x, y, z inputs.

        Parameters
        ----------
        x : float , optional
            The X coordinate. The default is 0.
        y : float , optional
            The Y coordinate. The default is 0.
        z : float , optional
            The Z coordinate. The defaults is 0.

        Returns
        -------
        coreVertex
            The created vertex.

        """
        vertex = None
        try:
            vertex = coreVertex.by_coordinates(x, y, z)
        except:
            vertex = None
        return vertex
    
    @staticmethod
    def Clockwise2D(vertices):
        """
        Sorts the input list of vertices in a clockwise fashion. This method assumes that the vertices are on the XY plane. The Z coordinate is ignored.

        Parameters
        -----------
        vertices : list
            The input list of vertices

        Return
        -----------
        list
            The input list of vertices sorted in a counter clockwise fashion
        
        """
        return list(reversed(Vertex.CounterClockwise2D(vertices)))
    
    @staticmethod
    def Coordinates(vertex: coreVertex, outputType: str = "xyz", mantissa: int = 4) -> list:
        """
        Returns the coordinates of the input vertex.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        outputType : string, optional
            The desired output type. Could be any permutation or substring of "xyz" or the string "matrix". The default is "xyz". The input is case insensitive and the coordinates will be returned in the specified order.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        list
            The coordinates of the input vertex.

        """
        if not isinstance(vertex, coreVertex):
            return None
        x = round(vertex.X(), mantissa)
        y = round(vertex.Y(), mantissa)
        z = round(vertex.Z(), mantissa)
        matrix = [[1,0,0,x],
                [0,1,0,y],
                [0,0,1,z],
                [0,0,0,1]]
        output = []
        outputType = outputType.lower()
        if outputType == "matrix":
            return matrix
        else:
            outputType = list(outputType)
            for axis in outputType:
                if axis == "x":
                    output.append(x)
                elif axis == "y":
                    output.append(y)
                elif axis == "z":
                    output.append(z)
        return output

    @staticmethod
    def CounterClockwise2D(vertices):
        """
        Sorts the input list of vertices in a counterclockwise fashion. This method assumes that the vertices are on the XY plane. The Z coordinate is ignored.

        Parameters
        -----------
        vertices : list
            The input list of vertices

        Return
        -----------
        list
            The input list of vertices sorted in a counter clockwise fashion
        
        """
        import math
        # find the centroid of the points
        cx = sum(Vertex.X(v) for v in vertices) / len(vertices)
        cy = sum(Vertex.Y(v) for v in vertices) / len(vertices)

        # sort the points based on their angle with respect to the centroid
        vertices.sort(key=lambda v: (math.atan2(Vertex.Y(v) - cy, Vertex.X(v) - cx) + 2 * math.pi) % (2 * math.pi))
        return vertices
    
    @staticmethod
    def Distance(vertex: coreVertex, topology: coreTopology, mantissa: int = 4) -> float:
        """
        Returns the distance between the input vertex and the input topology.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        topology : coreTopology
            The input topology.
        mantissa: int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The distance between the input vertex and the input topology.

        """
        if not isinstance(vertex, coreVertex) or not isinstance(topology, coreTopology):
            return None
        return round(VertexUtility.distance(vertex, topology), mantissa)
    
    @staticmethod
    def EnclosingCell(vertex: coreVertex, topology: coreTopology, exclusive: bool = True, tolerance: float = 0.0001) -> list:
        """
        Returns the list of Cells found in the input topology that enclose the input vertex.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        topology : coreTopology
            The input topology.
        exclusive : bool , optional
            If set to True, return only the first found enclosing cell. The default is True.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Returns
        -------
        list
            The list of enclosing cells.

        """
        from Core.Cell import Cell as coreCell
        from Core.CellComplex import CellComplex as coreCellComplex
        
        def boundingBox(cell):
            vertices = []
            _ = cell.Vertices(None, vertices)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(aVertex.X())
                y.append(aVertex.Y())
                z.append(aVertex.Z())
            return ([min(x), min(y), min(z), max(x), max(y), max(z)])
        
        if isinstance(topology, coreCell):
            cells = [topology]
        elif isinstance(topology, coreCluster) or isinstance(topology, coreCellComplex):
            cells = []
            _ = topology.Cells(None, cells)
        else:
            return None
        if len(cells) < 1:
            return None
        enclosingCells = []
        for i in range(len(cells)):
            bbox = boundingBox(cells[i])
            if ((vertex.X() < bbox[0]) or (vertex.Y() < bbox[1]) or (vertex.Z() < bbox[2]) or (vertex.X() > bbox[3]) or (vertex.Y() > bbox[4]) or (vertex.Z() > bbox[5])) == False:
                if Core.CellUtility.Contains(cells[i], vertex, tolerance) == 0:
                    if exclusive:
                        return([cells[i]])
                    else:
                        enclosingCells.append(cells[i])
        return enclosingCells

    @staticmethod
    def Index(vertex: coreVertex, vertices: list, strict: bool = False, tolerance: float = 0.0001) -> int:
        """
        Returns index of the input vertex in the input list of vertices

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        vertices : list
            The input list of vertices.
        strict : bool , optional
            If set to True, the vertex must be strictly identical to the one found in the list. Otherwise, a distance comparison is used. The default is False.
        tolerance : float , optional
            The tolerance for computing if the input vertex is identical to a vertex from the list. The default is 0.0001.

        Returns
        -------
        int
            The index of the input vertex in the input list of vertices.

        """
        from Wrapper.Topology import Topology
        if not isinstance(vertex, coreVertex):
            return None
        if not isinstance(vertices, list):
            return None
        vertices = [v for v in vertices if isinstance(v, coreVertex)]
        if len(vertices) == 0:
            return None
        for i in range(len(vertices)):
            if strict:
                if Topology.IsSame(vertex, vertices[i]):
                    return i
            else:
                d = Vertex.Distance(vertex, vertices[i])
                if d < tolerance:
                    return i
        return None

    @staticmethod
    def InterpolateValue(vertex, vertices, n=3, key="intensity", tolerance=0.0001):
        """
        Interpolates the value of the input vertex based on the values of the *n* nearest vertices.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        vertices : list
            The input list of vertices.
        n : int , optional
            The maximum number of nearest vertices to consider. The default is 3.
        key : str , optional
            The key that holds the value to be interpolated in the dictionaries of the vertices. The default is "intensity".
        tolerance : float , optional
            The tolerance for computing if the input vertex is coincident with another vertex in the input list of vertices. The default is 0.0001.

        Returns
        -------
        coreVertex
            The input vertex with the interpolated value stored in its dictionary at the key specified by the input key. Other keys and values in the dictionary are preserved.

        """

        def interpolate_value(point, data_points, n, tolerance=0.0001):
            """
            Interpolates the value associated with a point in 3D by averaging the values of the n nearest points.
            The influence of the adjacent points is inversely proportional to their distance from the input point.

            Args:
                data_points (list): A list of tuples, each representing a data point in 3D space as (x, y, z, value).
                                    The 'value' represents the value associated with that data point.
                point (tuple): A tuple representing the point in 3D space as (x, y, z) for which we want to interpolate a value.
                n (int): The number of nearest points to consider for interpolation.

            Returns:
                The interpolated value for the input point.
            """
            # Calculate the distances between the input point and all data points
            distances = [(distance(p[:3], point), p[3]) for p in data_points]

            # Sort the distances in ascending order
            sorted_distances = sorted(distances, key=lambda x: x[0])

            # Take the n nearest points
            nearest_points = sorted_distances[:n]

            n_p = nearest_points[0]
            n_d = n_p[0]
            if n_d < tolerance:
                return n_p[1]

            # Calculate the weights for each nearest point based on inverse distance

            weights = [(1/d[0], d[1]) for d in nearest_points]

            # Normalize the weights so they sum to 1
            total_weight = sum(w[0] for w in weights)
            normalized_weights = [(w[0]/total_weight, w[1]) for w in weights]

            # Interpolate the value as the weighted average of the nearest points
            interpolated_value = sum(w[0]*w[1] for w in normalized_weights)

            return interpolated_value

        def distance(point1, point2):
            """
            Calculates the Euclidean distance between two points in 3D space.

            Args:
                point1 (tuple): A tuple representing a point in 3D space as (x, y, z).
                point2 (tuple): A tuple representing a point in 3D space as (x, y, z).

            Returns:
                The Euclidean distance between the two points.
            """
            return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)**0.5
        
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary

        if not isinstance(vertex, coreVertex):
            return None
        if not isinstance(vertices, list):
            return None
        
        vertices = [v for v in vertices if isinstance(v, coreVertex)]
        if len(vertices) == 0:
            return None
        
        point = (Vertex.X(vertex), Vertex.Y(vertex), Vertex.Z(vertex))
        data_points = []
        for v in vertices:
            d = Topology.Dictionary(v)
            value = Dictionary.ValueAtKey(d, key)
            if not value == None:
                if type(value) == int or type(value) == float:
                    data_points.append((Vertex.X(v), Vertex.Y(v), Vertex.Z(v), value))
        if len(data_points) == 0:
            return None
        if n > len(data_points):
            n = len(data_points)
        value = interpolate_value(point, data_points, n, tolerance=0.0001)
        d = Topology.Dictionary(vertex)
        d = Dictionary.SetValueAtKey(d, key, value)
        vertex = Topology.SetDictionary(vertex, d)
        return vertex


    @staticmethod
    def IsInside(vertex: coreVertex, topology: coreTopology, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input topology. Returns False otherwise.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        topology : coreTopology
            The input topology.
        tolerance : float , optional
            The tolerance for computing if the input vertex is enclosed in a cell. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input topology. False otherwise.

        """
        from Wrapper.Wire import Wire
        from Wrapper.Shell import Shell
        from Wrapper.CellComplex import CellComplex
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        if not isinstance(vertex, coreVertex):
            return None
        if not isinstance(topology, coreTopology):
            return None

        if isinstance(topology, coreVertex):
            return VertexUtility.distance(vertex, topology) < tolerance
        elif isinstance(topology, coreEdge):
            try:
                parameter = Core.EdgeUtility.ParameterAtPoint(topology, vertex)
            except:
                parameter = 400 #aribtrary large number greater than 1
            return 0 <= parameter <= 1
        elif isinstance(topology, coreWire):
            edges = Wire.Edges(topology)
            for edge in edges:
                if Vertex.IsInside(vertex, edge, tolerance):
                    return True
            return False
        elif isinstance(topology, coreFace):
            return coreFaceUtility.IsInside(topology, vertex, tolerance)
        elif isinstance(topology, coreShell):
            faces = Shell.Faces(topology)
            for face in faces:
                if Vertex.IsInside(vertex, face, tolerance):
                    return True
            return False
        elif isinstance(topology, ):
            return Core.CellUtility.Contains(topology, vertex, tolerance) == 0
        elif isinstance(topology, Core.CellComplex):
            cells = CellComplex.Cells(topology)
            faces = CellComplex.Faces(topology)
            edges = CellComplex.Edges(topology)
            vertices = CellComplex.Vertices(topology)
            subtopologies = cells + faces + edges + vertices
            for subtopology in subtopologies:
                if Vertex.IsInside(vertex, subtopology, tolerance):
                    return True
            return False
        elif isinstance(topology, coreCluster):
            cells = Cluster.Cells(topology)
            faces = Cluster.Faces(topology)
            edges = Cluster.Edges(topology)
            vertices = Cluster.Vertices(topology)
            subtopologies = cells + faces + edges + vertices
            for subtopology in subtopologies:
                if Vertex.IsInside(vertex, subtopology, tolerance):
                    return True
            return False
        return False
    
    @staticmethod
    def NearestVertex(vertex: coreVertex, topology: coreTopology, useKDTree: bool = True) -> coreVertex:
        """
        Returns the vertex found in the input topology that is the nearest to the input vertex.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        topology : coreTopology
            The input topology to be searched for the nearest vertex.
        useKDTree : bool , optional
            if set to True, the algorithm will use a KDTree method to search for the nearest vertex. The default is True.

        Returns
        -------
        coreVertex
            The nearest vertex.

        """        
        def SED(a, b):
            """Compute the squared Euclidean distance between X and Y."""
            p1 = (a.X(), a.Y(), a.Z())
            p2 = (b.X(), b.Y(), b.Z())
            return sum((i-j)**2 for i, j in zip(p1, p2))
        
        BT = collections.namedtuple("BT", ["value", "left", "right"])
        BT.__doc__ = """
        A Binary Tree (BT) with a node value, and left- and
        right-subtrees.
        """
        def firstItem(v):
            return v.X()
        def secondItem(v):
            return v.Y()
        def thirdItem(v):
            return v.Z()

        def itemAtIndex(v, index):
            if index == 0:
                return v.X()
            elif index == 1:
                return v.Y()
            elif index == 2:
                return v.Z()

        def sortList(vertices, index):
            if index == 0:
                vertices.sort(key=firstItem)
            elif index == 1:
                vertices.sort(key=secondItem)
            elif index == 2:
                vertices.sort(key=thirdItem)
            return vertices
        
        def kdtree(topology):
            assert isinstance(topology, coreTopology), "Vertex.NearestVertex: The input is not a Topology."
            vertices = []
            _ = topology.Vertices(None, vertices)
            assert (len(vertices) > 0), "Vertex.NearestVertex: Could not find any vertices in the input Topology"

            """Construct a k-d tree from an iterable of vertices.

            This algorithm is taken from Wikipedia. For more details,

            > https://en.wikipedia.org/wiki/K-d_tree#Construction

            """
            # k = len(points[0])
            k = 3

            def build(*, vertices, depth):
                if len(vertices) == 0:
                    return None
                #points.sort(key=operator.itemgetter(depth % k))
                vertices = sortList(vertices, (depth % k))

                middle = len(vertices) // 2
                
                return BT(
                    value = vertices[middle],
                    left = build(
                        vertices=vertices[:middle],
                        depth=depth+1,
                    ),
                    right = build(
                        vertices=vertices[middle+1:],
                        depth=depth+1,
                    ),
                )

            return build(vertices=list(vertices), depth=0)
        
        NNRecord = collections.namedtuple("NNRecord", ["vertex", "distance"])
        NNRecord.__doc__ = """
        Used to keep track of the current best guess during a nearest
        neighbor search.
        """

        def find_nearest_neighbor(*, tree, vertex):
            """Find the nearest neighbor in a k-d tree for a given vertex.
            """
            k = 3 # Forcing k to be 3 dimensional
            best = None
            def search(*, tree, depth):
                """Recursively search through the k-d tree to find the nearest neighbor.
                """
                nonlocal best

                if tree is None:
                    return
                distance = SED(tree.value, vertex)
                if best is None or distance < best.distance:
                    best = NNRecord(vertex=tree.value, distance=distance)

                axis = depth % k
                diff = itemAtIndex(vertex,axis) - itemAtIndex(tree.value,axis)
                if diff <= 0:
                    close, away = tree.left, tree.right
                else:
                    close, away = tree.right, tree.left

                search(tree=close, depth=depth+1)
                if diff**2 < best.distance:
                    search(tree=away, depth=depth+1)

            search(tree=tree, depth=0)
            return best.vertex
        
        if useKDTree:
            tree = kdtree(topology)
            return find_nearest_neighbor(tree=tree, vertex=vertex)
        else:
            vertices = []
            _ = topology.Vertices(None, vertices)
            distances = []
            indices = []
            for i in range(len(vertices)):
                distances.append(SED(vertex, vertices[i]))
                indices.append(i)
            sorted_indices = [x for _, x in sorted(zip(distances, indices))]
        return vertices[sorted_indices[0]]

    @staticmethod
    def Origin() -> coreVertex:
        """
        Returns a vertex with coordinates (0,0,0)

        Parameters
        -----------

        Return
        -----------
        coreVertex
        """
        return Vertex.ByCoordinates(0,0,0)

    @staticmethod
    def Project(vertex: coreVertex, face: coreFace, direction: bool = None, mantissa: int = 4, tolerance: float = 0.0001) -> coreVertex:
        """
        Returns a vertex that is the projection of the input vertex unto the input face.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex to project unto the input face.
        face : coreFace
            The input face that receives the projection of the input vertex.
        direction : vector, optional
            The direction in which to project the input vertex unto the input face. If not specified, the direction of the projection is the normal of the input face. The default is None.
        mantissa : int , optional
            The length of the desired mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreVertex
            The projected vertex.

        """
        from Core.Utilities.TopologicUtilities import FaceUtility as coreFaceUtility
        from Core.Utilities.TopologicUtilities import VertexUtility as coreFaceUtility

        from Wrapper.Edge import Edge
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        from Wrapper.Vector import Vector

        if not isinstance(vertex, coreVertex):
            return None
        if not isinstance(face, coreFace):
            return None
        if not direction:
            direction = Vector.Reverse(Face.NormalAtParameters(face, 0.5, 0.5, "XYZ", mantissa))
        if coreFaceUtility. IsInside(face, vertex, tolerance):
            return vertex
        d = VertexUtility.distance(vertex, face)*10
        far_vertex = Core.TopologyUtility.Translate(vertex, direction[0]*d, direction[1]*d, direction[2]*d)
        if VertexUtility.distance(vertex, far_vertex) > tolerance:
            e = coreEdge.by_start_vertex_end_vertex(vertex, far_vertex)
            pv = face.Intersect(e, False)
            if not pv:
                far_vertex = Core.TopologyUtility.Translate(vertex, -direction[0]*d, -direction[1]*d, -direction[2]*d)
                if VertexUtility.distance(vertex, far_vertex) > tolerance:
                    e = coreEdge.by_start_vertex_end_vertex(vertex, far_vertex)
                    pv = face.Intersect(e, False)
            return pv
        else:
            return None

    @staticmethod
    def X(vertex: coreVertex, mantissa: int = 4) -> float:
        """
        Returns the X coordinate of the input vertex.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The X coordinate of the input vertex.

        """
        if not isinstance(vertex, coreVertex):
            return None
        return round(vertex.X(), mantissa)

    @staticmethod
    def Y(vertex: coreVertex, mantissa: int = 4) -> float:
        """
        Returns the Y coordinate of the input vertex.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The Y coordinate of the input vertex.

        """
        if not isinstance(vertex, coreVertex):
            return None
        return round(vertex.Y(), mantissa)

    @staticmethod
    def Z(vertex: coreVertex, mantissa: int = 4) -> float:
        """
        Returns the Z coordinate of the input vertex.

        Parameters
        ----------
        vertex : coreVertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The Z coordinate of the input vertex.

        """
        if not isinstance(vertex, coreVertex):
            return None
        return round(vertex.Z(), mantissa)
           