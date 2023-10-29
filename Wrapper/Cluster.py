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

# Wrapper
from Wrapper.Vector import Vector
from Wrapper.Topology import Topology

class Cluster(coreCluster):
    @staticmethod
    def ByTopologies(topologies: list) -> coreCluster:
        """
        Creates a topologic Cluster from the input list of topologies.

        Parameters
        ----------
        topologies : list
            The list of topologies.

        Returns
        -------
        coreCluster
            The created topologic Cluster.

        """
        assert isinstance(topologies, list), "Cluster.ByTopologies - Error: Input is not a list"
        topologyList = [x for x in topologies if isinstance(x, coreTopology)]
        return coreCluster.ByTopologies(topologyList, False)

    @staticmethod
    def CellComplexes(cluster: coreCluster) -> list:
        """
        Returns the cellComplexes of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of cellComplexes.

        """
        if not isinstance(cluster, coreCluster):
            return None
        cellComplexes = []
        _ = cluster.CellComplexes(None, cellComplexes)
        return cellComplexes

    @staticmethod
    def Cells(cluster: coreCluster) -> list:
        """
        Returns the cells of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of cells.

        """
        if not isinstance(cluster, coreCluster):
            return None
        cells = []
        _ = cluster.Cells(None, cells)
        return cells

    @staticmethod
    def Edges(cluster: coreCluster) -> list:
        """
        Returns the edges of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of edges.

        """ 
        if not isinstance(cluster, coreCluster):
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return edges

    @staticmethod
    def Faces(cluster: coreCluster) -> list:
        """
        Returns the faces of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of faces.

        """
        if not isinstance(cluster, coreCluster):
            return None
        faces = []
        _ = cluster.Faces(None, faces)
        return faces

    @staticmethod
    def FreeCells(cluster: coreCluster) -> list:
        """
        Returns the free cells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of free cells.

        """
        from Wrapper.CellComplex import CellComplex
        from Wrapper.Topology import Topology

        if not isinstance(cluster, coreCluster):
            return None
        allCells = []
        _ = cluster.Cells(None, allCells)
        allCellsCluster = Cluster.ByTopologies(allCells)
        freeCells = []
        cellComplexes = []
        _ = cluster.CellComplexes(None, cellComplexes)
        cellComplexesCells = []
        for cellComplex in cellComplexes:
            tempCells = CellComplex.Cells(cellComplex)
            cellComplexesCells += tempCells
        if len(cellComplexesCells) == 0:
            return allCells
        cellComplexesCluster = Cluster.ByTopologies(cellComplexesCells)
        resultingCluster = Topology.Boolean(allCellsCluster, cellComplexesCluster, operation="Difference")
        if isinstance(resultingCluster, coreCell):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="cell")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeShells(cluster: coreCluster) -> list:
        """
        Returns the free shells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of free shells.

        """
        from Wrapper.Cell import Cell
        from Wrapper.Topology import Topology

        if not isinstance(cluster, coreCluster):
            return None
        allShells = []
        _ = cluster.Shells(None, allShells)
        allShellsCluster = Cluster.ByTopologies(allShells)
        cells = []
        _ = cluster.Cells(None, cells)
        cellsShells = []
        for cell in cells:
            tempShells = Cell.Shells(cell)
            cellsShells += tempShells
        if len(cellsShells) == 0:
            return allShells
        cellsCluster = Cluster.ByTopologies(cellsShells)
        resultingCluster = Topology.Boolean(allShellsCluster, cellsCluster, operation="Difference")
        if isinstance(resultingCluster, coreShell):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="shell")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeFaces(cluster: coreCluster) -> list:
        """
        Returns the free faces of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of free faces.

        """
        from Wrapper.Shell import Shell
        from Wrapper.Topology import Topology
        if not isinstance(cluster, coreCluster):
            return None
        allFaces = []
        _ = cluster.Faces(None, allFaces)
        allFacesCluster = Cluster.ByTopologies(allFaces)
        shells = []
        _ = cluster.Shells(None, shells)
        shellFaces = []
        for shell in shells:
            tempFaces = Shell.Faces(shell)
            shellFaces += tempFaces
        if len(shellFaces) == 0:
            return allFaces
        shellCluster = Cluster.ByTopologies(shellFaces)
        resultingCluster = Topology.Boolean(allFacesCluster, shellCluster, operation="Difference")
        if isinstance(resultingCluster, coreFace):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="face")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result

    @staticmethod
    def FreeWires(cluster: coreCluster) -> list:
        """
        Returns the free wires of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of free wires.

        """
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology

        if not isinstance(cluster, coreCluster):
            return None
        allWires = []
        _ = cluster.Wires(None, allWires)
        allWiresCluster = Cluster.ByTopologies(allWires)
        faces = []
        _ = cluster.Faces(None, faces)
        facesWires = []
        for face in faces:
            tempWires = Face.Wires(face)
            facesWires += tempWires
        if len(facesWires) == 0:
            return allWires
        facesCluster = Cluster.ByTopologies(facesWires)
        resultingCluster = Topology.Boolean(allWiresCluster, facesCluster, operation="Difference")
        if isinstance(resultingCluster, coreWire):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="wire")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeEdges(cluster: coreCluster) -> list:
        """
        Returns the free edges of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of free edges.

        """
        from Wrapper.Wire import Wire
        from Wrapper.Topology import Topology

        if not isinstance(cluster, coreCluster):
            return None
        allEdges = []
        _ = cluster.Edges(None, allEdges)
        allEdgesCluster = Cluster.ByTopologies(allEdges)
        wires = []
        _ = cluster.Wires(None, wires)
        wireEdges = []
        for wire in wires:
            tempEdges = Wire.Edges(wire)
            wireEdges += tempEdges
        if len(wireEdges) == 0:
            return allEdges
        wireCluster = Cluster.ByTopologies(wireEdges)
        resultingCluster = Topology.Boolean(allEdgesCluster, wireCluster, operation="Difference")
        if isinstance(resultingCluster, coreEdge):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="edge")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeVertices(cluster: coreCluster) -> list:
        """
        Returns the free vertices of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of free vertices.

        """
        from Wrapper.Edge import Edge
        from Wrapper.Topology import Topology

        if not isinstance(cluster, coreCluster):
            return None
        allVertices = []
        _ = cluster.Vertices(None, allVertices)
        allVerticesCluster = Cluster.ByTopologies(allVertices)
        edges = []
        _ = cluster.Edges(None, edges)
        edgesVertices = []
        for edge in edges:
            tempVertices = Edge.Vertices(edge)
            edgesVertices += tempVertices
        if len(edgesVertices) == 0:
            return allVertices
        edgesCluster = Cluster.ByTopologies(edgesVertices)
        resultingCluster = Topology.Boolean(allVerticesCluster, edgesCluster, operation="Difference")
        if isinstance(resultingCluster, coreVertex):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="vertex")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def HighestType(cluster: coreCluster) -> int:
        """
        Returns the type of the highest dimension subtopology found in the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        int
            The type of the highest dimension subtopology found in the input cluster.

        """
        if not isinstance(cluster, coreCluster):
            return None
        cellComplexes = Cluster.CellComplexes(cluster)
        if len(cellComplexes) > 0:
            return coreCellComplex.Type()
        cells = Cluster.Cells(cluster)
        if len(cells) > 0:
            return coreCell.Type()
        shells = Cluster.Shells(cluster)
        if len(shells) > 0:
            return coreShell.Type()
        faces = Cluster.Faces(cluster)
        if len(faces) > 0:
            return coreFace.Type()
        wires = Cluster.Wires(cluster)
        if len(wires) > 0:
            return coreWire.Type()
        edges = Cluster.Edges(cluster)
        if len(edges) > 0:
            return coreEdge.Type()
        vertices = Cluster.Vertices(cluster)
        if len(vertices) > 0:
            return coreVertex.Type()
    
    @staticmethod
    def K_Means(topologies, selectors=None, keys=["x", "y", "z"], k=4, maxIterations=100, centroidKey="k_centroid"):
        """
        Clusters the input topologies using K-Means clustering. See https://en.wikipedia.org/wiki/K-means_clustering

        Parameters
        ----------
        topologies : list
            The input list of topologies. If this is not a list of topologic vertices then please provide a list of selectors
        selectors : list , optional
            If the list of topologies are not vertices then please provide a corresponding list of selectors (vertices) that represent the topologies for clustering. For example, these can be the centroids of the topologies.
            If set to None, the list of topologies is expected to be a list of vertices. The default is None.
        keys : list, optional
            The keys in the embedded dictionaries in the topologies. If specified, the values at these keys will be added to the dimensions to be clustered. The values must be numeric. If you wish the x, y, z location to be included,
            make sure the keys list includes "X", "Y", and/or "Z" (case insensitive). The default is ["x", "y", "z"]
        k : int , optional
            The desired number of clusters. The default is 4.
        maxIterations : int , optional
            The desired maximum number of iterations for the clustering algorithm
        centroidKey : str , optional
            The desired dictionary key under which to store the cluster's centroid (this is not to be confused with the actual geometric centroid of the cluster). The default is "k_centroid"

        Returns
        -------
        list
            The created list of clusters.

        """
        from Wrapper.Helper import Helper
        from Wrapper.Vertex import Vertex
        from Wrapper.Dictionary import Dictionary
        from Wrapper.Topology import Topology


        def k_means(data, vertices, k=4, maxIterations=100):
            import random
            def euclidean_distance(p, q):
                return sum((pi - qi) ** 2 for pi, qi in zip(p, q)) ** 0.5

            # Initialize k centroids randomly
            centroids = random.sample(data, k)

            for _ in range(maxIterations):
                # Assign each data point to the nearest centroid
                clusters = [[] for _ in range(k)]
                clusters_v = [[] for _ in range(k)]
                for i, point in enumerate(data):
                    distances = [euclidean_distance(point, centroid) for centroid in centroids]
                    nearest_centroid_index = distances.index(min(distances))
                    clusters[nearest_centroid_index].append(point)
                    clusters_v[nearest_centroid_index].append(vertices[i])

                # Compute the new centroids as the mean of the points in each cluster
                new_centroids = []
                for cluster in clusters:
                    if not cluster:
                        # If a cluster is empty, keep the previous centroid
                        new_centroids.append(centroids[clusters.index(cluster)])
                    else:
                        new_centroids.append([sum(dim) / len(cluster) for dim in zip(*cluster)])

                # Check if the centroids have converged
                if new_centroids == centroids:
                    break

                centroids = new_centroids

            return {'clusters': clusters, 'clusters_v': clusters_v, 'centroids': centroids}



        if topologies == None:
            return None
        topologies = [t for t in topologies if isinstance(t, coreTopology)]
        if len(topologies) == 0:
            return None
        if selectors != None:
            selectors = [s for s in selectors if isinstance(s, coreVertex)]
            if len(selectors) == 0:
                return None
            if not len(selectors) == len(topologies):
                return None
        if keys == None:
            return None
        if k == None:
            return None
        if k < 1:
            return None
        if not isinstance(topologies, list):
            return None
        if len(topologies) < k:
            return None
        if len(topologies) == k:
            t_clusters = []
            for topology in topologies:
                t_cluster = Cluster.ByTopologies([topology])
                for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(t)
                        elif key.lower() == "y":
                            value = Vertex.Y(t)
                        elif key.lower() == "z":
                            value = Vertex.Z(t)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                d = Dictionary.ByKeysValues([centroidKey], [elements])
                t_cluster = Topology.SetDictionary(t_cluster, d)
                t_clusters.append(t_cluster)
            return t_clusters
        
        data = []
        if selectors == None:
            for t in topologies:
                elements = []
                if keys:
                    d = Topology.Dictionary(t)
                    for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(t)
                        elif key.lower() == "y":
                            value = Vertex.Y(t)
                        elif key.lower() == "z":
                            value = Vertex.Z(t)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                data.append(elements)
        else:
            for i, s in enumerate(selectors):
                elements = []
                if keys:
                    d = Topology.Dictionary(topologies[i])
                    for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(s)
                        elif key.lower() == "y":
                            value = Vertex.Y(s)
                        elif key.lower() == "z":
                            value = Vertex.Z(s)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                data.append(elements)
        if len(data) == 0:
            return None
        if selectors:
            dict = k_means(data, selectors, k=k, maxIterations=maxIterations)
        else:
            dict = k_means(data, topologies, k=k, maxIterations=maxIterations)
        clusters = dict['clusters_v']
        centroids = dict['centroids']
        t_clusters = []
        for i, cluster in enumerate(clusters):
            cluster_vertices = []
            for v in cluster:
                if selectors == None:
                    cluster_vertices.append(v)
                else:
                    index = selectors.index(v)
                    cluster_vertices.append(topologies[index])
            cluster = Cluster.ByTopologies(cluster_vertices)
            d = Dictionary.ByKeysValues([centroidKey], [centroids[i]])
            cluster = Topology.SetDictionary(cluster, d)
            t_clusters.append(cluster)
        return t_clusters

    @staticmethod
    def MysticRose(wire: coreWire = None, origin: coreVertex = None, radius: float = 0.5, sides: int = 16, perimeter: bool = True, direction: list = [0,0,1], placement:str = "center", tolerance: float = 0.0001) -> coreCluster:
        """
        Creates a mystic rose.

        Parameters
        ----------
        wire : coreWire , optional
            The input Wire. if set to None, a circle with the input parameters is created. Otherwise, the input parameters are ignored.
        origin : coreVertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the mystic rose. The default is 1.
        sides : int , optional
            The number of sides of the mystic rose. The default is 16.
        perimeter : bool , optional
            If True, the perimeter edges are included in the output. The default is True.
        direction : list , optional
            The vector representing the up direction of the mystic rose. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the mystic rose. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.cluster
            The created mystic rose (cluster of edges).

        """
        import topologicpy
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Wire import Wire
        from Wrapper.Cluster import Cluster
        from itertools import combinations

        if not wire:
            wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=0, toAngle=360, close=True, direction=direction, placement=placement, tolerance=tolerance)
        if not Wire.IsClosed(wire):
            return None
        vertices = Wire.Vertices(wire)
        indices = list(range(len(vertices)))
        combs = [[comb[0],comb[1]] for comb in combinations(indices, 2) if not (abs(comb[0]-comb[1]) == 1) and not (abs(comb[0]-comb[1]) == len(indices)-1)]
        edges = []
        if perimeter:
            edges = Wire.Edges(wire)
        for comb in combs:
            edges.append(Edge.ByVertices([vertices[comb[0]], vertices[comb[1]]]))
        return Cluster.ByTopologies(edges)

    @staticmethod
    def Shells(cluster: coreCluster) -> list:
        """
        Returns the shells of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of shells.

        """
        if not isinstance(cluster, coreCluster):
            return None
        shells = []
        _ = cluster.Shells(None, shells)
        return shells

    @staticmethod
    def Simplify(cluster: coreCluster):
        """
        Simplifies the input cluster if possible. For example, if the cluster contains only one cell, that cell is returned.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        coreTopology or list
            The simplification of the cluster.

        """
        if not isinstance(cluster, coreCluster):
            return None
        resultingTopologies = []
        topCC = []
        _ = cluster.CellComplexes(None, topCC)
        topCells = []
        _ = cluster.Cells(None, topCells)
        topShells = []
        _ = cluster.Shells(None, topShells)
        topFaces = []
        _ = cluster.Faces(None, topFaces)
        topWires = []
        _ = cluster.Wires(None, topWires)
        topEdges = []
        _ = cluster.Edges(None, topEdges)
        topVertices = []
        _ = cluster.Vertices(None, topVertices)
        if len(topCC) == 1:
            cc = topCC[0]
            ccVertices = []
            _ = cc.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cc)
        if len(topCC) == 0 and len(topCells) == 1:
            cell = topCells[0]
            ccVertices = []
            _ = cell.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 1:
            shell = topShells[0]
            ccVertices = []
            _ = shell.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(shell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 1:
            face = topFaces[0]
            ccVertices = []
            _ = face.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(face)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 1:
            wire = topWires[0]
            ccVertices = []
            _ = wire.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(wire)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 1:
            edge = topEdges[0]
            ccVertices = []
            _ = wire.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(edge)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 0 and len(topVertices) == 1:
            vertex = topVertices[0]
            resultingTopologies.append(vertex)
        if len(resultingTopologies) == 1:
            return resultingTopologies[0]
        return cluster

    @staticmethod
    def Vertices(cluster: coreCluster) -> list:
        """
        Returns the vertices of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(cluster, coreCluster):
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices)
        return vertices

    @staticmethod
    def Wires(cluster: coreCluster) -> list:
        """
        Returns the wires of the input cluster.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster.

        Returns
        -------
        list
            The list of wires.

        """
        if not isinstance(cluster, coreCluster):
            return None
        wires = []
        _ = cluster.Wires(None, wires)
        return wires

    