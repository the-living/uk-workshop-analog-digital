from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from math import sqrt
from math import fabs

__all__ = [
    'interpolate',
    'vector_length',
    'vector_add',
    'vector_sub',
    'vector_dot',
    'vector_scale',
    'point_distance',
    'interpolate_point',
    'average_point',
    'closest_point_on_line',
    'intersect_line_line',
]

def interpolate(a, b, t):
    return float(a + (fabs(b-a) * t))

def vector_length(vector):
    return sqrt(sum(n**2 for n in vector))

def vector_add(vector_a, vector_b):
    return tuple(a+b for a,b in zip(vector_a, vector_b))

def vector_sub(point_a, point_b):
    """Generate Vector from point A to point B"""
    return tuple([b-a for a,b in zip(point_a, point_b)])

def vector_dot(vector_a, vector_b):
    return sum(a*b for a,b in zip(vector_a, vector_b))

def vector_scale(vector, factor):
    return tuple(v * factor for v in vector)

def point_distance(point_a, point_b):
    return vector_length(vector_sub(point_a, point_b))

def interpolate_point(point_a, point_b, t):
    """Interpolate point at parameter 't' between two input points"""
    return tuple(a + (float(t) * (b-a)) for a,b in zip(point_a, point_b))

def average_point(points, kill_duplicates=True):
    """Compute average point in set of points
    (optional) remove duplicate points
    """
    if kill_duplicates:
        points = list(set(points))
    
    p = len(points)
    x, y, z = zip(*points)
    return tuple(sum(i)/p for i in [x,y,z])

def closest_point_on_line(point, line):
    x,y = point[:2]
    start,end = line
    dx,dy = vector_sub(start, end)
    mag = vector_length((dx,dy))
    assert mag > 0.

    u = (((x - start[0]) * dx) + ((y - start[1]) * dy)) / (mag * mag)
    u = max(0., min(1., u))
    if u <= 0:
        return start
    if u >= 1:
        return end
    return (start[0] + u * dx, start[1] + u * dy)

def intersect_line_line(p1, p2, p3, p4, tolerance=0.00001):
    """Evaluates intersection between two lines.
    If lines are co-planar and intersect within tolerance, returns Point
    if not, returns shortest line connecting two lines
    with start point on line A and end point on line B
    http://paulbourke.net/geometry/pointlineplane/lineline.c
    """
    
    p13 = vector_sub(p3, p1)
    p43 = vector_sub(p3, p4)
    p21 = vector_sub(p1, p2)
    if point_distance(p1,p2) < tolerance or point_distance(p3, p4) < tolerance:
        return None
    d1343 = vector_dot(p13, p43)
    d4321 = vector_dot(p43, p21)
    d1321 = vector_dot(p13, p21)
    d4343 = vector_dot(p43, p43)
    d2121 = vector_dot(p21, p21)
    denom = d2121 * d4343 - d4321 * d4321
    if abs(denom) < tolerance:
        return None
    numer = d1343 * d4321 - d1321 * d4343

    mua = numer / denom
    mub = (d1343 + d4321 * (mua)) / d4343

    pa = vector_add(p1, vector_scale(p21, mua))
    # pa = p1 + (p21 * mua)
    pb = vector_add(p3, vector_scale(p43, mub))
    # pb = b.Start + (p43 * mub)

    if point_distance(pa, pb) < tolerance:
        return pa
    return [pa, pb]