import numpy
import matplotlib.pyplot as plt
import random
import math

# Returns theta in [-pi/2, 3pi/2]
def generate_theta(a, b):
    u = random.random() / 4.0
    theta = numpy.arctan(b/a * numpy.tan(2*numpy.pi*u))

    v = random.random()
    if v < 0.25:
        return theta
    elif v < 0.5:
        return numpy.pi - theta
    elif v < 0.75:
        return numpy.pi + theta
    else:
        return -theta

def radius(a, b, theta):
    return a * b / numpy.sqrt((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)

def random_point(a, b):
    random_theta = generate_theta(a, b)
    max_radius = radius(a, b, random_theta)
    random_radius = max_radius * numpy.sqrt(random.random())

    return numpy.array([
        random_radius * numpy.cos(random_theta),
        random_radius * numpy.sin(random_theta)
    ])

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
    
def get_random_node_ellipsis_sampling(start=[50,50],goal=[470,390]):

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    d = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)

    #parameters of the ellipsis: a is the horizontal radius of the ellipsis (= the distance between start and goal) and b is the vertical radius of the ellipsis (set to 1/2 of a)
    a = d/2
    #b is HARD CODED, could be a parameter!
    b = d/4
    
    random_theta = generate_theta(a, b)
    max_radius = radius(a, b, random_theta)
    random_radius = max_radius * numpy.sqrt(random.random())

    x = random_radius * numpy.cos(random_theta)
    y = random_radius * numpy.sin(random_theta)
    rx, ry = rotate((-a,0), (x,y), theta)
    return rx+(d/2)+start[0], ry+start[1]


# points = numpy.array([random_point(a, b) for _ in range(2000)])
# rotated_points = []
# for point in points:
    # x = point[0]
    # y = point[1]
    # rx, ry = rotate((-a,0), (x,y), theta)
    # rotated_points.append(numpy.array([rx+(d/2)+start[0], ry+start[1]]))
# rotated_points=numpy.array(rotated_points)rotated_points = []
# for point in points:
    # x = point[0]
    # y = point[1]
    # rx, ry = rotate((-a,0), (x,y), theta)
    # rotated_points.append(numpy.array([rx+(d/2)+start[0], ry+start[1]]))
# rotated_points=numpy.array(rotated_points)

# room_area=[0,520,0,440]
# plt.xlim(room_area[:2])
# plt.ylim(room_area[2:])  
# plt.scatter(rotated_points[:,0], rotated_points[:,1])
# plt.show()