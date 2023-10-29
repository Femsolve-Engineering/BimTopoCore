from binascii import a2b_base64
from re import A
import itertools
import numpy as np
import math
from typing import List

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
from Wrapper.Cluster import Cluster
from Wrapper.Topology import Topology

class Wire(coreWire):
    @staticmethod
    def BoundingRectangle(topology: coreTopology, optimize: int = 0) -> coreWire:
        """
        Returns a wire representing a bounding rectangle of the input topology. The returned wire contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting wire will become axis-aligned.

        Parameters
        ----------
        topology : coreTopology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        
        Returns
        -------
        coreWire
            The bounding rectangle of the input topology.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Wire import Wire
        from Wrapper.Face import Face
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        from random import sample

        def br(topology):
            vertices = []
            _ = topology.Vertices(None, vertices)
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

        world_origin = Vertex.ByCoordinates(0,0,0)

        xTran = None
        # Create a sample face
        while not xTran:
            vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
            v = sample(vertices, 3)
            w = Wire.ByVertices(v)
            f = Face.ByWire(w)
            f = Face.Flatten(f)
            dictionary = Topology.Dictionary(f)
            xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
        
        topology = Topology.Translate(topology, xTran*-1, yTran*-1, zTran*-1)
        topology = Topology.Rotate(topology, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        topology = Topology.Rotate(topology, origin=world_origin, x=0, y=1, z=0, degree=-theta)
        
        boundingRectangle = br(topology)
        minX = boundingRectangle[0]
        minY = boundingRectangle[1]
        maxX = boundingRectangle[2]
        maxY = boundingRectangle[3]
        w = abs(maxX - minX)
        l = abs(maxY - minY)
        best_area = l*w
        orig_area = best_area
        best_z = 0
        best_br = boundingRectangle
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
                    minX, minY, maxX, maxY = br(t)
                    w = abs(maxX - minX)
                    l = abs(maxY - minY)
                    area = l*w
                    if area < orig_area*factor:
                        best_area = area
                        best_z = z
                        best_br = [minX, minY, maxX, maxY]
                        flag = True
                        break
                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_br = [minX, minY, maxX, maxY]
                        
        else:
            best_br = boundingRectangle

        minX, minY, maxX, maxY = best_br
        vb1 = coreVertex.by_coordinates(minX, minY, 0)
        vb2 = coreVertex.by_coordinates(maxX, minY, 0)
        vb3 = coreVertex.by_coordinates(maxX, maxY, 0)
        vb4 = coreVertex.by_coordinates(minX, maxY, 0)

        boundingRectangle = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=origin, x=0,y=0,z=1, degree=-best_z)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=world_origin, x=0, y=1, z=0, degree=theta)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=world_origin, x=0, y=0, z=1, degree=phi)
        boundingRectangle = Topology.Translate(boundingRectangle, xTran, yTran, zTran)

        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        boundingRectangle = Topology.SetDictionary(boundingRectangle, dictionary)
        return boundingRectangle

    @staticmethod
    def ByEdges(edges: list) -> coreWire:
        """
        Creates a wire from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.

        Returns
        -------
        coreWire
            The created wire.

        """
        if not isinstance(edges, list):
            return None
        edgeList = [x for x in edges if isinstance(x, coreEdge)]
        if len(edgeList) < 1:
            return None
        wire = None
        for anEdge in edgeList:
            if anEdge.Type() == 2:
                if wire == None:
                    wire = coreWire.ByEdges([anEdge])
                else:
                    try:
                        wire = wire.Merge(anEdge)
                    except:
                        continue
        if wire.Type() != 4:
            wire = None
        return wire

    @staticmethod
    def ByEdgesCluster(cluster: coreCluster) -> coreWire:
        """
        Creates a wire from the input cluster of edges.

        Parameters
        ----------
        cluster : coreCluster
            The input cluster of edges.

        Returns
        -------
        coreWire
            The created wire.

        """
        if not isinstance(cluster, coreCluster):
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return Wire.ByEdges(edges)

    @staticmethod
    def ByOffset(wire: coreWire, offset: float = 1.0, miter: bool = False, miterThreshold: float = None, offsetKey: str = None, miterThresholdKey: str = None, step: bool = True) -> coreWire:
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
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Face import Face
        from Wrapper.Shell import Shell
        from Wrapper.Cluster import Cluster
        from Wrapper.Dictionary import Dictionary
        from Wrapper.Vector import Vector
        from random import randrange, sample

        if not isinstance(wire, coreWire):
            return None
        if not miterThreshold:
            miterThreshold = offset*math.sqrt(2)
        flatFace = Face.ByWire(wire)
        flatFace = Face.Flatten(flatFace)
        
        world_origin = Vertex.ByCoordinates(0,0,0)
        # Retrieve the needed transformations
        dictionary = Topology.Dictionary(flatFace)
        xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")

        
        
        edges = Wire.Edges(wire)
        vertices = Wire.Vertices(wire)
        flatEdges = []
        flatVertices = []
        newEdges = []
        for i in range(len(vertices)):
            flatVertex = Topology.Translate(vertices[i], -xTran, -yTran, -zTran)
            flatVertex = Topology.Rotate(flatVertex, origin=world_origin, x=0, y=0, z=1, degree=-phi)
            flatVertex = Topology.Rotate(flatVertex, origin=world_origin, x=0, y=1, z=0, degree=-theta)
            flatVertices.append(flatVertex)
        vertices = flatVertices
        for i in range(len(edges)):
            flatEdge = Topology.Translate(edges[i], -xTran, -yTran, -zTran)
            flatEdge = Topology.Rotate(flatEdge, origin=world_origin, x=0, y=0, z=1, degree=-phi)
            flatEdge = Topology.Rotate(flatEdge, origin=world_origin, x=0, y=1, z=0, degree=-theta)
            flatEdges.append(flatEdge)
            if offsetKey:
                d = Topology.Dictionary(edges[i])
                value = Dictionary.ValueAtKey(d, key=offsetKey)
                c = Topology.Centroid(flatEdge)
                if value:
                    finalOffset = value
                else:
                    finalOffset = offset
            else:
                finalOffset = offset
            e1 = Edge.ByOffset2D(flatEdge,finalOffset)
            newEdges.append(e1)
        edges = flatEdges
        newVertices = []
        dupVertices = []
        if Wire.IsClosed(wire):
            e1 = newEdges[-1]
            e2 = newEdges[0]
            intV = Edge.Intersect2D(e1,e2)
            if intV:
                newVertices.append(intV)
                dupVertices.append(vertices[0])
            elif step:
                edgeVertices= Edge.Vertices(e1)
                newVertices.append(Vertex.NearestVertex(vertices[-1], Cluster.ByTopologies(edgeVertices), useKDTree=False))
                edgeVertices= Edge.Vertices(e2)
                newVertices.append(Vertex.NearestVertex(vertices[0], Cluster.ByTopologies(edgeVertices), useKDTree=False))
                dupVertices.append(vertices[0])
                dupVertices.append(vertices[0])
            else:
                tempEdge1 = Edge.ByVertices([Edge.StartVertex(e1), Edge.EndVertex(e2)])
                normal = Edge.Normal(e1)
                normal = [normal[0]*finalOffset*10, normal[1]*finalOffset*10, normal[2]*finalOffset*10]
                tempV = Vertex.ByCoordinates(vertices[0].X()+normal[0], vertices[0].Y()+normal[1], vertices[0].Z()+normal[2])
                tempEdge2 = Edge.ByVertices([vertices[0], tempV])
                intV = Edge.Intersect2D(tempEdge1,tempEdge2)
                newVertices.append(intV)
                dupVertices.append(vertices[0])
        else:
            newVertices.append(Edge.StartVertex(newEdges[0]))
        
        for i in range(len(newEdges)-1):
            e1 = newEdges[i]
            e2 = newEdges[i+1]
            intV = Edge.Intersect2D(e1,e2)
            if intV:
                newVertices.append(intV)
                dupVertices.append(vertices[i+1])
            elif step:
                newVertices.append(Edge.EndVertex(e1))
                newVertices.append(Edge.StartVertex(e2))
                dupVertices.append(vertices[i+1])
                dupVertices.append(vertices[i+1])
            else:
                tempEdge1 = Edge.ByVertices([Edge.StartVertex(e1), Edge.EndVertex(e2)])
                normal = Edge.Normal(e1)
                normal = [normal[0]*finalOffset*10, normal[1]*finalOffset*10, normal[2]*finalOffset*10]
                tempV = Vertex.ByCoordinates(vertices[i+1].X()+normal[0], vertices[i+1].Y()+normal[1], vertices[i+1].Z()+normal[2])
                tempEdge2 = Edge.ByVertices([vertices[i+1], tempV])
                intV = Edge.Intersect2D(tempEdge1,tempEdge2)
                newVertices.append(intV)
                dupVertices.append(vertices[i+1])

        vertices = dupVertices
        if not Wire.IsClosed(wire):
            newVertices.append(Edge.EndVertex(newEdges[-1]))
        newWire = Wire.ByVertices(newVertices, close=Wire.IsClosed(wire))
        
        newVertices = Wire.Vertices(newWire)
        newEdges = Wire.Edges(newWire)
        miterEdges = []
        cleanMiterEdges = []
        # Handle miter
        if miter:
            for i in range(len(newVertices)):
                if miterThresholdKey:
                    d = Topology.Dictionary(vertices[i])
                    value = Dictionary.ValueAtKey(d, key=miterThresholdKey)
                    if value:
                        finalMiterThreshold = value
                    else:
                        finalMiterThreshold = miterThreshold
                else:
                    finalMiterThreshold = miterThreshold
                if Vertex.Distance(vertices[i], newVertices[i]) > abs(finalMiterThreshold):
                    st = Topology.SuperTopologies(newVertices[i], newWire, topologyType="edge")
                    if len(st) > 1:
                        e1 = st[0]
                        e2 = st[1]
                        if not Edge.IsCollinear(e1, e2):
                            e1 = Edge.Reverse(e1)
                            bisector = Edge.ByVertices([vertices[i], newVertices[i]])
                            nv = Edge.VertexByDistance(bisector, distance=finalMiterThreshold, origin=Edge.StartVertex(bisector), tolerance=0.0001)
                            vec = Edge.Normal2D(bisector)
                            nv2 = Topology.Translate(nv, vec[0], vec[1], 0)
                            nv3 = Topology.Translate(nv, -vec[0], -vec[1], 0)
                            miterEdge = Edge.ByVertices([nv2,nv3])
                            if miterEdge:
                                miterEdge = Edge.SetLength(miterEdge, abs(offset)*10)
                                msv = Edge.Intersect2D(miterEdge, e1)
                                mev = Edge.Intersect2D(miterEdge, e2)
                                if (Topology.IsInside(e1, msv,tolerance=0.01) and (Topology.IsInside(e2, mev, tolerance=0.01))):
                                    miterEdge = Edge.ByVertices([msv, mev])
                                    if miterEdge:
                                        cleanMiterEdges.append(miterEdge)
                                        miterEdge = Edge.SetLength(miterEdge, Edge.Length(miterEdge)*1.02)
                                        miterEdges.append(miterEdge)

            c = Cluster.SelfMerge(Cluster.ByTopologies(newEdges+miterEdges))
            vertices = Wire.Vertices(c)
            subtractEdges = []
            for v in vertices:
                edges = Topology.SuperTopologies(v, c, topologyType="edge")
                if len(edges) == 2:
                    if not Edge.IsCollinear(edges[0], edges[1]):
                        adjacentVertices = Topology.AdjacentTopologies(v, c)
                        total = 0
                        for adjV in adjacentVertices:
                            tempEdges = Topology.SuperTopologies(adjV, c, topologyType="edge")
                            total += len(tempEdges)
                        if total == 8:
                            subtractEdges = subtractEdges+edges

            if len(subtractEdges) > 0:
                newWire = Topology.Boolean(newWire, Cluster.ByTopologies(subtractEdges), operation="difference")
                if len(cleanMiterEdges) > 0:
                    newWire = Topology.Boolean(newWire, Cluster.ByTopologies(cleanMiterEdges), operation="merge")

        newWire = Topology.Rotate(newWire, origin=world_origin, x=0, y=1, z=0, degree=theta)
        newWire = Topology.Rotate(newWire, origin=world_origin, x=0, y=0, z=1, degree=phi)
        newWire = Topology.Translate(newWire, xTran, yTran, zTran)
        return newWire

    @staticmethod
    def ByVertices(vertices: list, close: bool = True) -> coreWire:
        """
        Creates a wire from the input list of vertices.

        Parameters
        ----------
        vertices : list
            the input list of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.

        Returns
        -------
        coreWire
            The created wire.

        """
        from Wrapper.Cluster import Cluster
        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if isinstance(x, coreVertex)]
        if len(vertexList) < 2:
            return None
        edges = []
        for i in range(len(vertexList)-1):
            v1 = vertexList[i]
            v2 = vertexList[i+1]
            try:
                e = coreEdge.by_start_vertex_end_vertex(v1, v2)
                if e:
                    edges.append(e)
            except:
                continue
        if close:
            v1 = vertexList[-1]
            v2 = vertexList[0]
            try:
                e = coreEdge.by_start_vertex_end_vertex(v1, v2)
                if e:
                    edges.append(e)
            except:
                pass
        if len(edges) < 1:
            return None
        #return Wire.ByEdges(edges)
        c = Cluster.ByTopologies(edges)
        return Cluster.SelfMerge(c)

    @staticmethod
    def ByVerticesCluster(cluster: coreCluster, close: bool = True) -> coreWire:
        """
        Creates a wire from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic.cluster
            the input cluster of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.

        Returns
        -------
        coreWire
            The created wire.

        """
        if not isinstance(cluster, coreCluster):
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices)
        return Wire.ByVertices(vertices, close)

    @staticmethod
    def Circle(origin: coreVertex = None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreWire:
        """
        Creates a circle.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0,0,0).
        radius : float , optional
            The radius of the circle. The default is 0.5.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreWire
            The created circle.

        """
        if not origin:
            origin = coreVertex.by_coordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        radius = abs(radius)
        if radius < tolerance:
            return None
        
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) < tolerance:
            return None
        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*radius + origin.X()
            y = math.cos(angle)*radius + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append(coreVertex.ByCoordinates(x,y,z))

        baseWire = Wire.ByVertices(baseV[::-1], close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            baseWire = coreTopologyUtility.Translate(baseWire, radius, radius, 0)
        elif placement.lower() == "upperleft":
            baseWire = coreTopologyUtility.Translate(baseWire, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            baseWire = coreTopologyUtility.Translate(baseWire, -radius, radius, 0)
        elif placement.lower() == "upperright":
            baseWire = coreTopologyUtility.Translate(baseWire, -radius, -radius, 0)
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
        baseWire = coreTopologyUtility.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = coreTopologyUtility.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire

    @staticmethod
    def ConvexHull(topology):
        """
        Returns a wire representing the 2D convex hull of the input topology. The vertices of the topology are assumed to be coplanar.

        Parameters
        ----------
        topology : coreTopology
            The input topology.
                
        Returns
        -------
        coreWire
            The convex hull of the input topology.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        from random import sample


        def Left_index(points):
            
            '''
            Finding the left most point
            '''
            minn = 0
            for i in range(1,len(points)):
                if points[i][0] < points[minn][0]:
                    minn = i
                elif points[i][0] == points[minn][0]:
                    if points[i][1] > points[minn][1]:
                        minn = i
            return minn

        def orientation(p, q, r):
            '''
            To find orientation of ordered triplet (p, q, r). 
            The function returns following values 
            0 --> p, q and r are collinear 
            1 --> Clockwise 
            2 --> Counterclockwise 
            '''
            val = (q[1] - p[1]) * (r[0] - q[0]) - \
                (q[0] - p[0]) * (r[1] - q[1])
        
            if val == 0:
                return 0
            elif val > 0:
                return 1
            else:
                return 2
        
        def convex_hull(points, n):
            
            # There must be at least 3 points 
            if n < 3:
                return
        
            # Find the leftmost point
            l = Left_index(points)
        
            hull = []
            
            '''
            Start from leftmost point, keep moving counterclockwise 
            until reach the start point again. This loop runs O(h) 
            times where h is number of points in result or output. 
            '''
            p = l
            q = 0
            while(True):
                
                # Add current point to result 
                hull.append(p)
        
                '''
                Search for a point 'q' such that orientation(p, q, 
                x) is counterclockwise for all points 'x'. The idea 
                is to keep track of last visited most counterclock- 
                wise point in q. If any point 'i' is more counterclock- 
                wise than q, then update q. 
                '''
                q = (p + 1) % n
        
                for i in range(n):
                    
                    # If i is more counterclockwise 
                    # than current q, then update q 
                    if(orientation(points[p], 
                                points[i], points[q]) == 2):
                        q = i
        
                '''
                Now q is the most counterclockwise with respect to p 
                Set p as q for next iteration, so that q is added to 
                result 'hull' 
                '''
                p = q
        
                # While we don't come to first point
                if(p == l):
                    break
        
            # Print Result 
            return hull


        xTran = None
        # Create a sample face and flatten
        while not xTran:
            vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
            v = sample(vertices, 3)
            w = Wire.ByVertices(v)
            f = Face.ByWire(w)
            f = Face.Flatten(f)
            dictionary = Topology.Dictionary(f)
            xTran = Dictionary.ValueAtKey(dictionary,"xTran")
        yTran = Dictionary.ValueAtKey(dictionary,"yTran")
        zTran = Dictionary.ValueAtKey(dictionary,"zTran")
        phi = Dictionary.ValueAtKey(dictionary,"phi")
        theta = Dictionary.ValueAtKey(dictionary,"theta")
        
        world_origin = Vertex.Origin()
        topology = Topology.Translate(topology, xTran*-1, yTran*-1, zTran*-1)
        topology = Topology.Rotate(topology, origin=world_origin, x=0, y=0, z=1, degree=-phi)
        topology = Topology.Rotate(topology, origin=world_origin, x=0, y=1, z=0, degree=-theta)

        vertices = Topology.Vertices(topology)

        points = []
        for v in vertices:
            points.append((Vertex.X(v), Vertex.Y(v)))
        hull = convex_hull(points, len(points))

        hull_vertices = []
        for p in hull:
            hull_vertices.append(Vertex.ByCoordinates(points[p][0], points[p][1], 0))

        ch = Wire.ByVertices(hull_vertices)
        ch = Topology.Rotate(ch, origin=world_origin, x=0, y=1, z=0, degree=theta)
        ch = Topology.Rotate(ch, origin=world_origin, x=0, y=0, z=1, degree=phi)
        ch = Topology.Translate(ch, xTran, yTran, zTran)
        return ch

    @staticmethod
    def Cycles(wire: coreWire, maxVertices: int = 4, tolerance: float = 0.0001) -> list:
        """
        Returns the closed circuits of wires found within the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        maxVertices : int , optional
            The maximum number of vertices of the circuits to be searched. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of circuits (closed wires) found within the input wire.

        """
        
        def vIndex(v, vList, tolerance):
            for i in range(len(vList)):
                if VertexUtility.distance(v, vList[i]) < tolerance:
                    return i+1
            return None
        
        #  rotate cycle path such that it begins with the smallest node
        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:]+path[:n]

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def isNew(cycles, path):
            return not path in cycles

        def visited(node, path):
            return node in path

        def findNewCycles(graph, cycles, path, maxVertices):
            if len(path) > maxVertices:
                return
            start_node = path[0]
            next_node= None
            sub = []

            #visit each edge and each node of each edge
            for edge in graph:
                node1, node2 = edge
                if start_node in edge:
                        if node1 == start_node:
                            next_node = node2
                        else:
                            next_node = node1
                        if not visited(next_node, path):
                                # neighbor node not on path yet
                                sub = [next_node]
                                sub.extend(path)
                                # explore extended path
                                findNewCycles(graph, cycles, sub, maxVertices);
                        elif len(path) > 2  and next_node == path[-1]:
                                # cycle found
                                p = rotate_to_smallest(path);
                                inv = invert(p)
                                if isNew(cycles, p) and isNew(cycles, inv):
                                    cycles.append(p)

        def main(graph, cycles, maxVertices):
            returnValue = []
            for edge in graph:
                for node in edge:
                    findNewCycles(graph, cycles, [node], maxVertices)
            for cy in cycles:
                row = []
                for node in cy:
                    row.append(node)
                returnValue.append(row)
            return returnValue

        tEdges = []
        _ = wire.Edges(None, tEdges)
        tVertices = []
        _ = wire.Vertices(None, tVertices)
        tVertices = tVertices

        graph = []
        for anEdge in tEdges:
            graph.append([vIndex(anEdge.StartVertex(), tVertices, tolerance), vIndex(anEdge.EndVertex(), tVertices, tolerance)])

        cycles = []
        resultingCycles = main(graph, cycles, maxVertices)

        result = []
        for aRow in resultingCycles:
            row = []
            for anIndex in aRow:
                row.append(tVertices[anIndex-1])
            result.append(row)

        resultWires = []
        for i in range(len(result)):
            c = result[i]
            resultEdges = []
            for j in range(len(c)-1):
                v1 = c[j]
                v2 = c[j+1]
                e = coreEdge.ByStartVertexEndVertex(v1, v2)
                resultEdges.append(e)
            e = coreEdge.ByStartVertexEndVertex(c[len(c)-1], c[0])
            resultEdges.append(e)
            resultWire = coreWire.ByEdges(resultEdges)
            resultWires.append(resultWire)
        return resultWires

    @staticmethod
    def Edges(wire: coreWire) -> list:
        """
        Returns the edges of the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        list
            The list of edges.

        """
        if not isinstance(wire, coreWire):
            return None
        edges = wire.edges()
        return edges

    @staticmethod
    def Einstein(origin: coreVertex = None, radius: float = 0.5, direction: list = [0,0,1], placement: str = "center") -> coreWire:
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
        from Wrapper.Vertex import Vertex
        from Wrapper.Topology import Topology
        import math
        def cos(angle):
            return math.cos(math.radians(angle))
        def sin(angle):
            return math.sin(math.radians(angle))
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        d = cos(30)*radius
        v1 = Vertex.ByCoordinates(0,0,0)
        v2 = Vertex.ByCoordinates(cos(30)*d, sin(30)*d, 0)
        v3 = Vertex.ByCoordinates(radius, 0)
        v4 = Vertex.ByCoordinates(2*radius, 0)
        v5 = Vertex.ByCoordinates(2*radius+cos(60)*radius*0.5, sin(30)*d, 0)
        v6 = Vertex.ByCoordinates(1.5*radius, d)
        v7 = Vertex.ByCoordinates(1.5*radius, 2*d)
        v8 = Vertex.ByCoordinates(radius, 2*d)
        v9 = Vertex.ByCoordinates(radius-cos(60)*0.5*radius, 2*d+sin(60)*0.5*radius)
        v10 = Vertex.ByCoordinates(0, 2*d)
        v11 = Vertex.ByCoordinates(0, d)
        v12 = Vertex.ByCoordinates(-radius*0.5, d)
        v13 = Vertex.ByCoordinates(-cos(30)*d, sin(30)*d, 0)
        einstein = Wire.ByVertices([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13], close=True)
        
        if placement.lower() == "lowerleft":
            einstein = Topology.Translate(einstein, radius, d, 0)
        dx = Vertex.X(origin)
        dy = Vertex.Y(origin)
        dz = Vertex.Z(origin)
        einstein = Topology.Translate(einstein, dx, dy, dz)
        if direction != [0,0,1]:
            einstein = Topology.Orient(einstein, origin=origin, dirA=[0,0,1], dirB=direction)
        return einstein
    
    @staticmethod
    def Ellipse(origin: coreVertex = None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: float = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreWire:
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0,0,0).
        inputMode : int , optional
            The method by wich the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreWire
            The created ellipse

        """
        ellipseAll = Wire.EllipseAll(origin=origin, inputMode=inputMode, width=width, length=length, focalLength=focalLength, eccentricity=eccentricity, majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=close, direction=direction, placement=placement, tolerance=tolerance)
        return ellipseAll["ellipse"]

    @staticmethod
    def EllipseAll(origin: coreVertex = None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0,0,1], placement: str ="center", tolerance: float = 0.0001) -> coreWire:
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0,0,0).
        inputMode : int , optional
            The method by wich the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "ellipse" : The ellipse (coreWire)
            2. "foci" : The two focal points (coreCluster containing two vertices)
            3. "a" : The major axis length
            4. "b" : The minor axis length
            5. "c" : The focal length
            6. "e" : The eccentricity
            7. "width" : The width
            8. "length" : The length

        """
        if not origin:
            origin = coreVertex.by_coordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            return None
        if inputMode not in [1,2,3,4]:
            return None
        if placement.lower() not in ["center", "lowerleft"]:
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        width = abs(width)
        length = abs(length)
        focalLength= abs(focalLength)
        eccentricity=abs(eccentricity)
        majorAxisLength=abs(majorAxisLength)
        minorAxisLength=abs(minorAxisLength)
        sides = abs(sides)
        if width < tolerance or length < tolerance or focalLength < tolerance or eccentricity < tolerance or majorAxisLength < tolerance or minorAxisLength < tolerance or sides < 3:
            return None
        if inputMode == 1:
            w = width
            l = length
            a = width/2
            b = length/2
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
        elif inputMode == 2:
            c = focalLength
            e = eccentricity
            a = c/e
            b = math.sqrt(abs(a**2 - c**2))
            w = a*2
            l = b*2
        elif inputMode == 3:
            c = focalLength
            b = minorAxisLength
            a = math.sqrt(abs(b**2 + c**2))
            e = c/a
            w = a*2
            l = b*2
        elif inputMode == 4:
            a = majorAxisLength
            b = minorAxisLength
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
            w = a*2
            l = b*2
        else:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle - fromAngle) < tolerance:
            return None

        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*a + origin.X()
            y = math.cos(angle)*b + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append(coreVertex.by_coordinates(x,y,z))

        ellipse = Wire.ByVertices(baseV[::-1], close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            ellipse = coreTopologyUtility.Translate(ellipse, a, b, 0)
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
        ellipse = coreTopologyUtility.Rotate(ellipse, origin, 0, 1, 0, theta)
        ellipse = coreTopologyUtility.Rotate(ellipse, origin, 0, 0, 1, phi)

        # Create a Cluster of the two foci
        v1 = coreVertex.ByCoordinates(c+origin.X(), 0+origin.Y(),0)
        v2 = coreVertex.ByCoordinates(-c+origin.X(), 0+origin.Y(),0)
        foci = coreCluster.ByTopologies([v1, v2])
        if placement.lower() == "lowerleft":
            foci = coreTopologyUtility.Translate(foci, a, b, 0)
        foci = coreTopologyUtility.Rotate(foci, origin, 0, 1, 0, theta)
        foci = coreTopologyUtility.Rotate(foci, origin, 0, 0, 1, phi)
        d = {}
        d['ellipse'] = ellipse
        d['foci'] = foci
        d['a'] = a
        d['b'] = b
        d['c'] = c
        d['e'] = e
        d['w'] = w
        d['l'] = l
        return d

    @staticmethod
    def Flatten(wire: coreWire, oldLocation: coreVertex =None, newLocation: coreVertex = None, direction: list = None):
        """
        Flattens the input wire such that its center of mass is located at the origin and the specified direction is pointed in the positive Z axis.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        oldLocation : coreVertex , optional
            The old location to use as the origin of the movement. If set to None, the center of mass of the input topology is used. The default is None.
        newLocation : coreVertex , optional
            The new location at which to place the topology. If set to None, the world origin (0,0,0) is used. The default is None.
        direction : list , optional
            The direction, expressed as a list of [X,Y,Z] that signifies the direction of the wire. If set to None, the positive ZAxis direction is considered the direction of the wire. The deafult is None.

        Returns
        -------
        coreWire
            The flattened wire.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        from Wrapper.Vector import Vector
        if not isinstance(wire, coreWire):
            return None
        if direction == None:
            direction = Vector.ZAxis()
        if not isinstance(oldLocation, coreVertex):
            oldLocation = Topology.CenterOfMass(wire)
        if not isinstance(newLocation, coreVertex):
            newLocation = Vertex.ByCoordinates(0,0,0)
        cm = oldLocation
        world_origin = newLocation

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
        flatWire = Topology.Translate(wire, -cm.X(), -cm.Y(), -cm.Z())
        flatWire = Topology.Rotate(flatWire, world_origin, 0, 0, 1, -phi)
        flatWire = Topology.Rotate(flatWire, world_origin, 0, 1, 0, -theta)
        # Ensure flatness. Force Z to be zero
        edges = Wire.Edges(flatWire)
        flatEdges = []
        for edge in edges:
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            sv1 = Vertex.ByCoordinates(Vertex.X(sv), Vertex.Y(sv), 0)
            ev1 = Vertex.ByCoordinates(Vertex.X(ev), Vertex.Y(ev), 0)
            e1 = Edge.ByVertices([sv1, ev1])
            flatEdges.append(e1)
        flatWire = Topology.SelfMerge(Cluster.ByTopologies(flatEdges))
        dictionary = Dictionary.ByKeysValues(["xTran", "yTran", "zTran", "phi", "theta"], [cm.X(), cm.Y(), cm.Z(), phi, theta])
        flatWire = Topology.SetDictionary(flatWire, dictionary)
        return flatWire
    
    @staticmethod
    def Interpolate(wires: list, n: int = 5, outputType: str = "default", replication: str = "default") -> coreTopology:
        """
        Creates *n* number of wires that interpolate between wireA and wireB.

        Parameters
        ----------
        wireA : coreWire
            The first input wire.
        wireB : coreWire
            The second input wire.
        n : int , optional
            The number of intermediate wires to create. The default is 5.
        outputType : str , optional
            The desired type of output. The options are case insensitive. The default is "contour". The options are:
                - "Default" or "Contours" (wires are not connected)
                - "Raster or "Zigzag" or "Toolpath" (the wire ends are connected to create a continous path)
                - "Grid" (the wire ends are connected to create a grid). 
        replication : str , optiona;
            The desired type of replication for wires with different number of vertices. It is case insensitive. The default is "default". The options are:
                - "Default" or "Repeat" which repeats the last vertex of the wire with the least number of vertices
                - "Nearest" which maps the vertices of one wire to the nearest vertex of the next wire creating a list of equal number of vertices.
        Returns
        -------
        toplogic.Topology
            The created interpolated wires as well as the input wires. The return type can be a coreCluster or a coreWire based on options.

        """

        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Face import Face
        from Wrapper.Cluster import Cluster
        from Wrapper.Helper import Helper
        
        outputType = outputType.lower()
        if outputType not in ["default", "contours", "raster", "zigzag", "toolpath", "grid"]:
            return None
        if outputType == "default" or outputType == "contours":
            outputType = "contours"
        if outputType == "raster" or outputType == "zigzag" or outputType == "toolpath":
            outputType = "zigzag"
        
        replication = replication.lower()
        if replication not in ["default", "nearest", "repeat"]:
            return None
        
        def nearestVertex(v, vertices):
            distances = [Vertex.Distance(v, vertex) for vertex in vertices]
            return vertices[distances.index(sorted(distances)[0])]
        
        def replicate(vertices, replication="default"):
            vertices = Helper.Repeat(vertices)
            finalList = vertices
            if replication == "nearest":
                finalList = [vertices[0]]
                for i in range(len(vertices)-1):
                    loopA = vertices[i]
                    loopB = vertices[i+1]
                    nearestVertices = []
                    for j in range(len(loopA)):
                        #clusB = Cluster.ByTopologies(loopB)
                        #nv = Vertex.NearestVertex(loopA[j], clusB, useKDTree=False)
                        nv = nearestVertex(loopA[j], loopB)
                        nearestVertices.append(nv)
                    finalList.append(nearestVertices)
            return finalList
        
        def process(verticesA, verticesB, n=5, outputType="contours", replication="repeat"):
            #if outputType == "zigzag" and Wire.IsClosed(wireA):
                #verticesA.append(verticesA[0])
            #verticesA, verticesB = replicate(verticesA=verticesA, verticesB=verticesB, replication=replication)
            
            contours = [verticesA]
            for i in range(1, n+1):
                u = float(i)/float(n+1)
                temp_vertices = []
                for j in range(len(verticesA)):
                    temp_v = Edge.VertexByParameter(Edge.ByVertices([verticesA[j], verticesB[j]]), u)
                    temp_vertices.append(temp_v)
                contours.append(temp_vertices)
            return contours
        
        if len(wires) < 2:
            return None
        
        vertices = []
        for wire in wires:
            vertices.append(Topology.SubTopologies(wire, subTopologyType="vertex"))
        vertices = replicate(vertices, replication=replication)
        contours = []
        
        finalWires = []
        for i in range(len(vertices)-1):
            verticesA = vertices[i]
            verticesB = vertices[i+1]
            contour = process(verticesA=verticesA, verticesB=verticesB, n=n, outputType=outputType, replication=replication)
            contours += contour
            for c in contour:
                finalWires.append(Wire.ByVertices(c, Wire.IsClosed(wires[i])))

        contours.append(vertices[-1])
        finalWires.append(wires[-1])
        ridges = []
        if outputType == "grid" or outputType == "zigzag":
            for i in range(len(contours)-1):
                verticesA = contours[i]
                verticesB = contours[i+1]
                if outputType == "grid":
                    for j in range(len(verticesA)):
                        ridges.append(Edge.ByVertices([verticesA[j], verticesB[j]]))
                elif outputType == "zigzag":
                    if i%2 == 0:
                        sv = verticesA[-1]
                        ev = verticesB[-1]
                        ridges.append(Edge.ByVertices([sv, ev]))
                    else:
                        sv = verticesA[0]
                        ev = verticesB[0]
                        ridges.append(Edge.ByVertices([sv, ev]))

        return Topology.SelfMerge(Cluster.ByTopologies(finalWires+ridges))
    
    @staticmethod
    def Invert(wire: coreWire) -> coreWire:
        """
        Creates a wire that is an inverse (mirror) of the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        coreWire
            The inverted wire.

        """
        if not isinstance(wire, coreWire):
            return None
        vertices = Wire.Vertices(wire)
        reversed_vertices = vertices[::-1]
        return Wire.ByVertices(reversed_vertices)

    @staticmethod
    def IsClosed(wire: coreWire) -> bool:
        """
        Returns True if the input wire is closed. Returns False otherwise.

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        bool
            True if the input wire is closed. False otherwise.

        """
        status = None
        if wire:
            if isinstance(wire, coreWire):
                status = wire.is_closed()
        return status
    
    @staticmethod
    def IsInside(wire: coreWire, vertex: coreVertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is inside the input wire. Returns False otherwise.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        vertex : coreVertex
            The input Vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input wire. False otherwise.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        
        if not isinstance(wire, coreWire):
            return None
        if not isinstance(vertex, coreVertex):
            return None
        is_inside = False
        edges = Wire.Edges(wire)
        for edge in edges:
            if (Vertex.Distance(vertex, edge) <= tolerance):
                return True
        return False
    
    @staticmethod
    def Isovist(wire: coreWire, viewPoint: coreVertex, obstaclesCluster: coreCluster, tolerance: float = 0.0001) -> list:
        """
        Returns a list of faces representing the isovist projection from the input viewpoint.

        Parameters
        ----------
        wire : coreWire
            The wire representing the external boundary (border) of the isovist.
        viewPoint : coreVertex
            The vertex representing the location of the viewpoint of the isovist.
        obstaclesCluster : coreCluster
            A cluster of wires representing the obstacles within the externalBoundary.

        Returns
        -------
        list
            A list of faces representing the isovist projection from the input viewpoint.

        """
        
        def vertexPartofFace(vertex, face: coreFace, tolerance):
            vertices = []
            _ = face.vertices(None, vertices)
            for v in vertices:
                if VertexUtility.distance(vertex, v) < tolerance:
                    return True
            return False
        
        internalBoundaries = []
        _ = obstaclesCluster.wires(None, internalBoundaries)
        internalVertices = []
        _ = obstaclesCluster.vertices(None, internalVertices)
        # 1. Create a Face with external and internal boundaries
        face = coreFace.by_external_internal_boundaries(wire, internalBoundaries, False)
        # 2. Draw Rays from viewpoint through each Vertex of the obstacles extending to the External Boundary
        #    2.1 Get the Edges and Vertices of the External Boundary
        exBoundaryEdges = []
        _ = wire.edges(None, exBoundaryEdges)
        exBoundaryVertices = []
        _ = wire.vertices(None, exBoundaryVertices)
        testTopologies = exBoundaryEdges+exBoundaryVertices
        #    1.2 Find the maximum distance from the viewpoint to the edges and vertices of the external boundary
        distances = []
        for x in testTopologies:
            distances.append(VertexUtility.distance(viewPoint, x))
        maxDistance = max(distances)*1.5
        #    1.3 Shoot rays and intersect with the external boundary
        rays = []
        for aVertex in (internalVertices+exBoundaryVertices):
            d = VertexUtility.distance(viewPoint, aVertex)
            if d > tolerance:
                scaleFactor = maxDistance/d
                newV = coreTopologyUtility.Scale(aVertex, viewPoint, scaleFactor, scaleFactor, scaleFactor)
                try:
                    ray = coreEdge.by_start_vertex_end_vertex(viewPoint, newV)
                    topologyC = ray.Intersect(wire, False)
                    vertices = []
                    _ = topologyC.Vertices(None, vertices)
                    if topologyC:
                        try:
                            rays.append(coreEdge.by_start_vertex_end_vertex(viewPoint, vertices[0]))
                        except:
                            pass
                    try:
                        rays.append(coreEdge.by_start_vertex_end_vertex(viewPoint, aVertex))
                    except:
                        pass
                except:
                    pass
        rayEdges = []
        for r in rays:
            a: coreTopology = r.Difference(obstaclesCluster, False)
            if a:
                edges = []
                _ = a.edges(None, edges)
                w = None
                try:
                    w = coreWire.by_edges(edges)
                    rayEdges = rayEdges + edges
                except:
                    c = coreCluster.ByTopologies(edges)
                    c = c.SelfMerge()
                    wires: List[coreWire] = []
                    _ = c.Wires(None, wires)
                    if len(wires) > 0:
                        edges: List[coreEdge] = []
                        _ = wires[0].edges(None, edges)
                        rayEdges = rayEdges + edges
                    else:
                        for e in edges:
                            vertices = []
                            e.vertices(None, vertices)
                            for v in vertices:
                                if VertexUtility.distance(viewPoint, v) < tolerance:
                                    rayEdges.append(e)
        rayCluster = coreCluster.ByTopologies(rayEdges)
        #return rayCluster
        shell = face.Slice(rayCluster, False)
        faces = []
        _ = shell.Faces(None, faces)
        finalFaces = []
        for aFace in faces:
            if vertexPartofFace(viewPoint, aFace, 0.001):
                finalFaces.append(aFace)
        return finalFaces

    @staticmethod
    def IsSimilar(wireA: coreWire, wireB: coreWire, angTolerance: float = 0.1, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input wires are similar. Returns False otherwise. The wires must be closed.

        Parameters
        ----------
        wireA : coreWire
            The first input wire.
        wireB : coreWire
            The second input wire.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two input wires are similar. False otherwise.

        """
        
        def isCyclicallyEquivalent(u, v, lengthTolerance, angleTolerance):
            n, i, j = len(u), 0, 0
            if n != len(v):
                return False
            while i < n and j < n:
                if (i % 2) == 0:
                    tol = lengthTolerance
                else:
                    tol = angleTolerance
                k = 1
                while k <= n and math.fabs(u[(i + k) % n]- v[(j + k) % n]) <= tol:
                    k += 1
                if k > n:
                    return True
                if math.fabs(u[(i + k) % n]- v[(j + k) % n]) > tol:
                    i += k
                else:
                    j += k
            return False

        def angleBetweenEdges(e1: coreEdge, e2: coreEdge, tolerance):
            a = e1.end_vertex().X() - e1.start_vertex().X()
            b = e1.end_vertex().Y() - e1.start_vertex().Y()
            c = e1.end_vertex().Z() - e1.start_vertex().Z()
            d = VertexUtility.distance(e1.end_vertex(), e2.start_vertex())
            if d <= tolerance:
                d = e2.start_vertex().X() - e2.end_vertex().X()
                e = e2.start_vertex().Y() - e2.end_vertex().Y()
                f = e2.start_vertex().Z() - e2.end_vertex().Z()
            else:
                d = e2.end_vertex().X() - e2.start_vertex().X()
                e = e2.end_vertex().Y() - e2.start_vertex().Y()
                f = e2.end_vertex().Z() - e2.start_vertex().Z()
            dotProduct = a*d + b*e + c*f
            modOfVector1 = math.sqrt( a*a + b*b + c*c)*math.sqrt(d*d + e*e + f*f) 
            angle = dotProduct/modOfVector1
            angleInDegrees = math.degrees(math.acos(angle))
            return angleInDegrees

        def getInteriorAngles(edges, tolerance):
            angles = []
            for i in range(len(edges)-1):
                e1 = edges[i]
                e2 = edges[i+1]
                angles.append(angleBetweenEdges(e1, e2, tolerance))
            return angles

        def getRep(edges, tolerance):
            angles = getInteriorAngles(edges, tolerance)
            lengths = []
            for anEdge in edges:
                lengths.append(EdgeUtility.length(anEdge))
            minLength = min(lengths)
            normalisedLengths = []
            for aLength in lengths:
                normalisedLengths.append(aLength/minLength)
            return [x for x in itertools.chain(*itertools.zip_longest(normalisedLengths, angles)) if x is not None]
        
        if (wireA.IsClosed() == False):
            return None
        if (wireB.IsClosed() == False):
            return None
        edgesA = []
        _ = wireA.Edges(None, edgesA)
        edgesB = []
        _ = wireB.Edges(None, edgesB)
        if len(edgesA) != len(edgesB):
            return False
        repA = getRep(list(edgesA), tolerance)
        repB = getRep(list(edgesB), tolerance)
        if isCyclicallyEquivalent(repA, repB, tolerance, angTolerance):
            return True
        if isCyclicallyEquivalent(repA, repB[::-1], tolerance, angTolerance):
            return True
        return False

    @staticmethod
    def Length(wire: coreWire, mantissa: int = 4) -> float:
        """
        Returns the length of the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.

        Returns
        -------
        float
            The length of the input wire.

        """
        if not wire:
            return None
        if not isinstance(wire, coreWire):
            return None
        totalLength = None
        try:
            edges = []
            _ = wire.edges(None, edges)
            totalLength = 0
            for anEdge in edges:
                totalLength = totalLength + EdgeUtility.length(anEdge)
            totalLength = round(totalLength, mantissa)
        except:
            totalLength = None
        return totalLength

    @staticmethod
    def Planarize(wire: coreWire) -> coreWire:
        """
        Returns a planarized version of the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        coreWire
            The planarized wire.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Face import Face
        from Wrapper.Topology import Topology
        if not isinstance(wire, coreWire):
            return None
        verts = []
        _ = wire.Vertices(None, verts)
        w = Wire.ByVertices([verts[0], verts[1], verts[2]], close=True)
        f = coreFace.by_external_boundary(w)
        f = Topology.Scale(f, f.Centroid(), 500,500,500)
        proj_verts = []
        direction = Face.NormalAtParameters(f)
        for v in verts:
            v = Vertex.ByCoordinates(v.X()+direction[0]*5, v.Y()+direction[1]*5, v.Z()+direction[2]*5)
            proj_verts.append(Vertex.Project(v, f))
        return Wire.ByVertices(proj_verts, close=True)

    @staticmethod
    def Project(wire: coreWire, face: coreFace, direction: list = None, mantissa: int = 4) -> coreWire:
        """
        Creates a projection of the input wire unto the input face.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        face : coreFace
            The face unto which to project the input wire.
        direction : list, optional
            The vector direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreWire
            The projected wire.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Face import Face
        if not wire:
            return None
        if not isinstance(wire, coreWire):
            return None
        if not face:
            return None
        if not isinstance(face, coreFace):
            return None
        if not direction:
            direction = -1*Face.NormalAtParameters(face, 0.5, 0.5, "XYZ", mantissa)
        large_face: Face = Topology.Scale(face, face.CenterOfMass(), 500, 500, 500)
        edges: List[coreEdge] = wire.edges()
        projected_edges = []

        if large_face:
            if (large_face.Type() == Face.Type()):
                for edge in edges:
                    if edge:
                        if (edge.Type() == coreEdge.Type()):
                            sv = edge.start_vertex()
                            ev = edge.end_vertex()

                            psv = Vertex.Project(vertex=sv, face=large_face, direction=direction)
                            pev = Vertex.Project(vertex=ev, face=large_face, direction=direction)
                            if psv and pev:
                                try:
                                    pe = Edge.ByVertices([psv, pev])
                                    projected_edges.append(pe)
                                except:
                                    continue
        w = Wire.ByEdges(projected_edges)
        return w

    @staticmethod
    def Rectangle(origin: coreVertex = None, width: float = 1.0, length: float = 1.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreWire:
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
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreWire
            The created rectangle.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Topology import Topology
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            print("Wire.Rectangle - Error: specified origin is not a topologic vertex. Retruning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Rectangle - Error: Could not find placement in the list of placements. Retruning None.")
            return None
        width = abs(width)
        length = abs(length)
        if width < tolerance or length < tolerance:
            print("Wire.Rectangle - Error: One or more of the specified dimensions is below the tolerance value. Retruning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            print("Wire.Rectangle - Error: The direction vector magnitude is below the tolerance value. Retruning None.")
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5

        vb1 = Vertex.ByCoordinates(origin.x()-width*0.5+xOffset,origin.y()-length*0.5+yOffset,origin.z())
        vb2 = Vertex.ByCoordinates(origin.x()+width*0.5+xOffset,origin.y()-length*0.5+yOffset,origin.z())
        vb3 = Vertex.ByCoordinates(origin.x()+width*0.5+xOffset,origin.y()+length*0.5+yOffset,origin.z())
        vb4 = Vertex.ByCoordinates(origin.x()-width*0.5+xOffset,origin.y()+length*0.5+yOffset,origin.z())

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], True)
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
        baseWire = Topology.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = Topology.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire
    
    @staticmethod
    def RemoveCollinearEdges(wire: coreWire, angTolerance: float = 0.1) -> coreWire:
        """
        Removes any collinear edges in the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.

        Returns
        -------
        coreWire
            The created wire without any collinear edges.

        """
        from Wrapper.Edge import Edge
        from Wrapper.Wire import Wire
        from Wrapper.Topology import Topology
        def rce(wire, angTolerance=0.1):
            if not isinstance(wire, coreWire):
                return None
            final_wire = None
            vertices: List[coreVertex] = []
            wire_verts = []
            try:
                _ = wire.Vertices(None, vertices)
            except:
                return None
            for aVertex in vertices:
                edges = aVertex.edges(wire)
                if len(edges) > 1:
                    if not Edge.IsCollinear(edges[0], edges[1], angTolerance=angTolerance):
                        wire_verts.append(aVertex)
                else:
                    wire_verts.append(aVertex)
            if len(wire_verts) > 2:
                if wire.is_closed():
                    final_wire = Wire.ByVertices(wire_verts, True)
                else:
                    final_wire = Wire.ByVertices(wire_verts, False)
            elif len(wire_verts) == 2:
                final_wire = coreEdge.by_start_vertex_end_vertex(wire_verts[0], wire_verts[1])
            return final_wire
        
        if not coreTopology.is_manifold(wire, wire):
            wires = Wire.Split(wire)
        else:
            wires = [wire]
        returnWires = []
        for aWire in wires:
            if not isinstance(aWire, coreWire):
                returnWires.append(aWire)
            else:
                returnWires.append(rce(aWire, angTolerance=angTolerance))
        if len(returnWires) == 1:
            returnWire = returnWires[0]
            if isinstance(returnWire, coreEdge):
                return Wire.ByEdges([returnWire])
            elif isinstance(returnWire, coreWire):
                return returnWire
            else:
                return None
        elif len(returnWires) > 1:
            returnWire = coreCluster.ByTopologies(returnWires).SelfMerge()
            if isinstance(returnWire, coreEdge):
                return Wire.ByEdges([returnWire])
            elif isinstance(returnWire, coreWire):
                return returnWire
            else:
                return None
        else:
            return None

    def Roof(face, degree=45, tolerance=0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : coreFace
            The input face.
        degree : float , optioal
            The desired angle in degrees of the roof. The default is 45.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        coreWire
            The created roof. This method returns the roof as a set of edges. No faces are created.

        """
        from Wrapper.Vertex import Vertex
        from Wrapper.Edge import Edge
        from Wrapper.Face import Face
        from Wrapper.Cluster import Cluster
        from Wrapper.Topology import Topology
        from Wrapper.Dictionary import Dictionary
        from Wrapper.Helper import Helper
        import math

        def subtrees_to_edges(subtrees, polygon, slope):
            polygon_z = {}
            for x, y, z in polygon:
                polygon_z[(x, y)] = z

            edges = []
            for subtree in subtrees:
                source = subtree.source
                height = subtree.height
                z = slope * height
                source_vertex = Vertex.ByCoordinates(source.x, source.y, z)

                for sink in subtree.sinks:
                    if (sink.x, sink.y) in polygon_z:
                        z = 0
                    else:
                        z = None
                        for st in subtrees:
                            if st.source.x == sink.x and st.source.y == sink.y:
                                z = slope * st.height
                                break
                            for sk in st.sinks:
                                if sk.x == sink.x and sk.y == sink.y:
                                    z = slope * st.height
                                    break
                        if z is None:
                            height = subtree.height
                            z = slope * height
                    sink_vertex = Vertex.ByCoordinates(sink.x, sink.y, z)
                    if (source.x, source.y) == (sink.x, sink.y):
                        continue
                    if Edge.ByStartVertexEndVertex(source_vertex, sink_vertex) not in edges:
                        edges.append(Edge.ByStartVertexEndVertex(source_vertex, sink_vertex))
            return edges
        
        def face_to_skeleton(face, degree=0):
            normal = Face.Normal(face)
            eb_wire = Face.ExternalBoundary(face)
            ib_wires = Face.InternalBoundaries(face)
            eb_vertices = Topology.Vertices(eb_wire)
            if normal[2] > 0:
                eb_vertices = list(reversed(eb_vertices))
            eb_polygon_coordinates = [(v.X(), v.Y(), v.Z()) for v in eb_vertices]
            eb_polygonxy = [(x[0], x[1]) for x in eb_polygon_coordinates]

            ib_polygonsxy = []
            zero_coordinates = eb_polygon_coordinates
            for ib_wire in ib_wires:
                ib_vertices = Topology.Vertices(ib_wire)
                if normal[2] > 0:
                    ib_vertices = list(reversed(ib_vertices))
                ib_polygon_coordinates = [(v.X(), v.Y(), v.Z()) for v in ib_vertices]
                ib_polygonxy = [(x[0], x[1]) for x in ib_polygon_coordinates]
                ib_polygonsxy.append(ib_polygonxy)
                zero_coordinates += ib_polygon_coordinates
            skeleton = Polyskel.skeletonize(eb_polygonxy, ib_polygonsxy)
            slope = math.tan(math.radians(degree))
            roofEdges = subtrees_to_edges(skeleton, zero_coordinates, slope)
            roofEdges = Helper.Flatten(roofEdges)+Topology.Edges(face)
            roofTopology = Topology.SelfMerge(Cluster.ByTopologies(roofEdges))
            return roofTopology
        
        if not isinstance(face, coreFace):
            return None
        degree = abs(degree)
        if degree >= 90-tolerance:
            return None
        flat_face = Face.Flatten(face)
        d = Topology.Dictionary(flat_face)
        roof = face_to_skeleton(flat_face, degree)
        if not roof:
            return None
        xTran = Dictionary.ValueAtKey(d,"xTran")
        yTran = Dictionary.ValueAtKey(d,"yTran")
        zTran = Dictionary.ValueAtKey(d,"zTran")
        phi = Dictionary.ValueAtKey(d,"phi")
        theta = Dictionary.ValueAtKey(d,"theta")
        roof = Topology.Rotate(roof, origin=Vertex.Origin(), x=0, y=1, z=0, degree=theta)
        roof = Topology.Rotate(roof, origin=Vertex.Origin(), x=0, y=0, z=1, degree=phi)
        roof = Topology.Translate(roof, xTran, yTran, zTran)
        return roof
    
    def Skeleton(face, tolerance=0.001):
        """
            Creates a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel


        Parameters
        ----------
        face : coreFace
            The input face.
       
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        coreWire
            The created straight skeleton.

        """
        if not isinstance(face, coreFace):
            return None
        return Wire.Roof(face, degree=0, tolerance=tolerance)
    
    @staticmethod
    def Split(wire: coreWire) -> list:
        """
        Splits the input wire into segments at its intersections (i.e. at any vertex where more than two edges meet).

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        list
            The list of split wire segments.

        """
        
        def vertexDegree(v: coreVertex, wire: coreWire):
            edges = v.edges(wire, edges)
            return len(edges)
        
        def vertexOtherEdge(vertex: coreVertex, edge: coreEdge, wire: coreWire):
            edges = vertex.edges(wire, edges)
            if coreTopology.is_same(edges[0], edge):
                return edges[-1]
            else:
                return edges[0]
        
        def edgeOtherVertex(edge: coreEdge, vertex: coreVertex):
            vertices = []
            _ = edge.vertices(None, vertices)
            if coreTopology.is_same(vertex, vertices[0]):
                return vertices[-1]
            else:
                return vertices[0]
        
        def edgeInList(edge, edgeList):
            for anEdge in edgeList:
                if coreTopology.IsSame(anEdge, edge):
                    return True
            return False
        
        vertices = wire.vertices(None)
        hubs: List[coreVertex] = []
        for aVertex in vertices:
            if vertexDegree(aVertex, wire) > 2:
                hubs.append(aVertex)
        wires = []
        global_edges = []
        for aVertex in hubs:
            hub_edges = aVertex.edges(wire)
            wire_edges = []
            for hub_edge in hub_edges:
                if not edgeInList(hub_edge, global_edges):
                    current_edge = hub_edge
                    oe = edgeOtherVertex(current_edge, aVertex)
                    while vertexDegree(oe, wire) == 2:
                        if not edgeInList(current_edge, global_edges):
                            global_edges.append(current_edge)
                            wire_edges.append(current_edge)
                        current_edge = vertexOtherEdge(oe, current_edge, wire)
                        oe = edgeOtherVertex(current_edge, oe)
                    if not edgeInList(current_edge, global_edges):
                        global_edges.append(current_edge)
                        wire_edges.append(current_edge)
                    if len(wire_edges) > 1:
                        wires.append(coreCluster.ByTopologies(wire_edges).SelfMerge())
                    else:
                        wires.append(wire_edges[0])
                    wire_edges = []
        if len(wires) < 1:
            return [wire]
        return wires
    
    @staticmethod
    def Square(origin: coreVertex = None, size: float = 1.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreWire:
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
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreWire
            The created square.

        """
        return Wire.Rectangle(origin = origin, width = size, length = size, direction = direction, placement = placement, tolerance = tolerance)
    
    @staticmethod
    def Star(origin: coreWire = None, radiusA: float = 0.5, radiusB: float = 0.2, rays: int = 8, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreWire:
        """
        Creates a star.

        Parameters
        ----------
        origin : coreVertex , optional
            The location of the origin of the star. The default is None which results in the star being placed at (0,0,0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 8.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0,0,1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        coreWire
            The created star.

        """

        if not origin:
            origin = coreVertex.by_coordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            return None
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusA < tolerance or radiusB < tolerance:
            return None
        rays = abs(rays)
        if rays < 3:
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        sides = rays*2 # Sides is double the number of rays
        baseV = []

        xList = []
        yList = []
        for i in range(sides):
            if i%2 == 0:
                radius = radiusA
            else:
                radius = radiusB
            angle = math.radians(360/sides)*i
            x = math.sin(angle)*radius + origin.X()
            y = math.cos(angle)*radius + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append([x,y])

        if placement.lower() == "lowerleft":
            xmin = min(xList)
            ymin = min(yList)
            xOffset = origin.X() - xmin
            yOffset = origin.Y() - ymin
        elif placement.lower() == "upperleft":
            xmin = min(xList)
            ymax = max(yList)
            xOffset = origin.X() - xmin
            yOffset = origin.Y() - ymax
        elif placement.lower() == "lowerright":
            xmax = max(xList)
            ymin = min(yList)
            xOffset = origin.X() - xmax
            yOffset = origin.Y() - ymin
        elif placement.lower() == "upperright":
            xmax = max(xList)
            ymax = max(yList)
            xOffset = origin.X() - xmax
            yOffset = origin.Y() - ymax
        else:
            xOffset = 0
            yOffset = 0
        tranBase = []
        for coord in baseV:
            tranBase.append(coreVertex.by_coordinates(coord[0]+xOffset, coord[1]+yOffset, origin.Z()))
        
        baseWire = Wire.ByVertices(tranBase[::-1], True) #reversing the list so that the normal points up in Blender
        
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
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Z-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Y-Axis
        baseWire = coreTopologyUtility.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = coreTopologyUtility.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire

    @staticmethod
    def Trapezoid(origin: coreVertex = None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0,0,1], placement: str = "center", tolerance: float = 0.0001) -> coreWire:
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : coreVertex , optional
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
        coreWire
            The created trapezoid.

        """
        if not origin:
            origin = coreVertex.ByCoordinates(0,0,0)
        if not isinstance(origin, coreVertex):
            return None
        widthA = abs(widthA)
        widthB = abs(widthB)
        length = abs(length)
        if widthA < tolerance or widthB < tolerance or length < tolerance:
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "center":
            xOffset = -((-widthA*0.5 + offsetA) + (-widthB*0.5 + offsetB) + (widthA*0.5 + offsetA) + (widthB*0.5 + offsetB))/4.0
            yOffset = 0
        elif placement.lower() == "lowerleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -(max((widthA*0.5 + offsetA), (widthB*0.5 + offsetB)))
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -(max((widthA*0.5 + offsetA), (widthB*0.5 + offsetB)))
            yOffset = -length*0.5

        vb1 = coreVertex.by_coordinates(origin.X()-widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = coreVertex.by_coordinates(origin.X()+widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = coreVertex.by_coordinates(origin.X()+widthB*0.5+offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = coreVertex.by_coordinates(origin.X()-widthB*0.5+offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], True)
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
        baseWire = coreTopologyUtility.Rotate(baseWire, origin, 0, 1, 0, theta)
        baseWire = coreTopologyUtility.Rotate(baseWire, origin, 0, 0, 1, phi)
        return baseWire

    @staticmethod
    def Vertices(wire: coreWire) -> list:
        """
        Returns the list of vertices of the input wire.

        Parameters
        ----------
        wire : coreWire
            The input wire.

        Returns
        -------
        list
            The list of vertices.

        """
        if not isinstance(wire, coreWire):
            return None
        vertices: List[coreVertex] = wire.vertices(None)
        return vertices

