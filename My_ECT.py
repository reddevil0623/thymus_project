import numpy as np
import numba as nb
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


@nb.njit()
def euler_critical_values(simp_comp, data, direction):
    """
    Returns an ordered list of Euler critical values for an embedded simplicial complex filtered in given direction
    :param simp_comp: Embedded simplicial complex stored as list of simplices. Each simplex is a list of vertices,
                        given as tuples.
    :param data: Locations of points of simp_comp in Euclidean space.
    :param direction: Direction (i.e. unit vector) to filter in. Given as tuple.
    :return: Ordered list of Euler critical values, filtration as list of elements (simplex, filtration_value)
    """
    """
        1. An empty numpy array named filtration is initialized. It will store the Euler critical values and is of shape (len(simp_comp), 2). 
           The first column of this array will store the size (number of vertices) of each simplex, and the second column will store the filtration value for that simplex.

        2. The function then loops through each simplex in simp_comp.

            a. For each vertex in the simplex, it computes the dot product of the vertex's coordinates (data[v]) and the given direction.

            b. The maximum dot product value among all the vertices of the simplex is taken as the filtration value (fv) for that simplex.

            c. The size of the simplex (i.e., its number of vertices) and the filtration value are stored in the filtration array.
        3. The function returns the filtration array.
    """
    filtration = np.empty((len(simp_comp), 2), dtype=float)
    for i, simplex in enumerate(simp_comp):
        fv = max([np.sum(np.multiply(data[v], direction)) for v in simplex])
        filtration[i] = (float(len(simplex)), fv)
    return filtration



@nb.njit()
def euler_curve(simp_comp, data, direction, interval=(-1., 1.), points: int = 100):
    """
    Returns Euler curve as evenly spaced evaluations on given interval
    :param simp_comp: Simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param direction: Direction (i.e. unit vector) for filtration as tuple
    :param interval: Interval on which to evaluate the Euler curve on
    :param points: Number of evenly spaced points at which to evaluate the Euler curve
    :return: 1-D array representing evenly spaced evaluations of Euler curve on interval
    """
    """
    1. Compute the filtration using the euler_critical_values function from the previous code. 
       This returns a 2D numpy array where the first column contains the size of each simplex and the second column contains the filtration value for that simplex.
    2. Sort the filtration array based on the filtration values (i.e., the second column).
    3. Initialize two variables: value and c. value will store the current value of the Euler characteristic, while c is an index counter for the filtration array.
    4. Compute the step size for the evaluations based on the specified interval and points.
    5. Initialize an empty numpy array chi to store the values of the Euler curve.
    6. Loop over a set of evenly spaced points within the interval:

        a. Check if the current point x is within the interval of a filtration value. If it is, the loop updates the value of the Euler characteristic (value) based on the simplices added during this step of the filtration.

        b. The update to the Euler characteristic is based on the size of the simplex being added. The formula float((-1) ** int(filtration[c, -1])) calculates the change in the Euler characteristic based on the simplex's dimension (i.e., its size). A vertex increases the Euler characteristic by 1, an edge decreases it by 1, a triangle increases it by 1, and so on.

        c. The computed value of the Euler characteristic for the current point is stored in the chi array.
    7. The function returns the chi array, which represents the Euler curve evaluated at the specified points.
    """
    filtration = euler_critical_values(simp_comp, data, direction)
    order = np.argsort(filtration[:, 1])
    filtration = filtration[order]
    value, c = 0., 0
    step_size = (interval[1] - interval[0]) / (points - 1)

    chi = np.empty(points, dtype=float)

    for i, x in enumerate(np.linspace(interval[0], interval[1], points)):
        if x < filtration[c, 1] <= min(x + step_size, interval[1]):
            while filtration[c, 1] <= x + step_size and c < len(filtration):
                
                value -= float((-1) ** int(filtration[c, -2]))
                c += 1

        chi[i] = value

    return chi

def RandomEct_2d(simp_comp, data, k=20, interval=(-1., 1.), points=100, factor=3):
    """
    Computes the ECT of a 2D-embedded simplicial complex by sampling 'k' random
    directions on S¹ and returns the ECT curves.

    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param k: number of random directions on S¹ for ECT
    :param interval: interval over which to construct Euler curves
    :param points: number of evaluations of the Euler curve
    :param factor: use factor * points many evaluations in interval for integration
    :return: ECT curve of given shape
    """
    # Convert simp_comp to a numba-compatible list of lists
    numba_comp_sc = nb.typed.List()
    for s in simp_comp:
        numba_comp_sc.append(nb.typed.List(s))

    # Generate k random angles in [0, 2π)
    thetas = 2 * np.pi * np.random.rand(k)

    # Prepare array to store ECT results
    ect = np.empty((k, points), dtype=np.float64)
    
    # Compute the ECT for each random direction
    for i in nb.prange(k):
        theta = thetas[i]
        direction = np.array((np.sin(theta), np.cos(theta)))
        
        # euler_curve is assumed to be already defined with the signature:
        #    euler_curve(simp_comp, data, direction, interval, total_points)
        ec_values = euler_curve(numba_comp_sc, data, direction,
                                interval, points * factor)
        # Downsample to get 'points' many values
        ect[i] = ec_values[::factor]

    return ect

def ect_2d(simp_comp, data, k=20, interval = (-1., 1.), points: int = 100, factor: int = 3):
    """
    Currently only for shapes embedded into 2D
    Computes the euler_curve for k evenly spaced directions on S¹ and returns the supremum of the absolute difference between the curves
    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param k: number of evenly spaced points on S¹ to use for ECT
    :param interval: interval over which to construct Euler curves
    :param points: number of evaluations of the Euler curve
    :param factor: use factor * points many evaluations in interval for integration
    :return: ECT curve of given shape
    """
    numba_comp_sc = nb.typed.List()
    for s in simp_comp:
        numba_comp_sc.append(nb.typed.List(s))
    thetas = np.linspace(0, 2 * np.pi, k + 1)

    ect = np.empty((k, points), dtype=float)
    for i in nb.prange(k):
        theta = thetas[i]
        direction = np.array((np.sin(theta), np.cos(theta)))
        ect[i] = euler_curve(numba_comp_sc, data, direction, interval, points*factor)[::factor]
    return ect

class EctImg:
    def __init__(self, simp_comp, data, k=20, xinterval=(-1., 1.), xpoints=100, yinterval=(-1., 1.), ypoints=100, factor=3):
        self.xinterval = xinterval
        self.yinterval = yinterval
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.image = self.compute(simp_comp, data, k, xinterval, xpoints, yinterval, ypoints, factor)

    def compute(self, simp_comp, data, k, xinterval, xpoints, yinterval, ypoints, factor):
        ect1 = ect_2d(simp_comp, data, k, xinterval, xpoints, factor)
        image = np.zeros((ypoints, xpoints), dtype=float)
        yvalues = np.linspace(yinterval[0], yinterval[1], ypoints+1, endpoint=True)
        for i in range(xpoints):
            column = ect1[:, i]
            for j in range(ypoints):
                value = 0
                if j < ypoints-1:
                    value = len(np.where((yvalues[j] <= column) & (column < yvalues[j+1]))[0])/k
                else:
                    value = len(np.where((yvalues[j] <= column) & (column <= yvalues[j+1]))[0])/k
                image[j, i] = value
        return image
    
    def plot(self):
        plt.figure(figsize=(10, 8))
        # Using xinterval and yinterval directly in extent
        plt.imshow(self.image, aspect='auto', extent=[self.xinterval[0], self.xinterval[1], self.yinterval[0], self.yinterval[1]], origin='lower', interpolation='none')
        plt.colorbar(label='Intensity')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('ECT Image Plot')
        plt.show()


            


def ect_metric(simp_comp1, data1, simp_comp2, data2, k=20, interval = (-1., 1.), points: int = 100):
    """
    Computes the ECT metric between two shapes
    :param simp_comp1: Embedded simplicial complex of shape 1
    :param data1: Locations of points of simp_comp1 in Euclidean space
    :param simp_comp2: Embedded simplicial complex of shape 2
    :param data2: Locations of points of simp_comp2 in Euclidean space
    :param k: number of evenly spaced points on S¹ to use for ECT
    :param interval: interval over which to construct Euler curves
    :param points: number of evaluations of the Euler curve
    :param factor: use factor * points many evaluations in interval for integration
    :return: ECT metric between the two shapes
    """
    thetas = np.linspace(0, 2 * np.pi, k + 1)
    step_size = (interval[1] - interval[0]) / (points - 1)
    supremum = 0.
    numba_comp_sc1 = nb.typed.List()
    for s in simp_comp1:
        numba_comp_sc1.append(nb.typed.List(s))
        
    numba_comp_sc2 = nb.typed.List()
    for s in simp_comp2:
        numba_comp_sc2.append(nb.typed.List(s))
    thetas = np.linspace(0, 2 * np.pi, k + 1)
    for i in nb.prange(k):
        theta = thetas[i]
        direction = np.array((np.sin(theta), np.cos(theta)))
        ect1 = euler_curve(numba_comp_sc1, data1, direction, interval, points)
        ect2 = euler_curve(numba_comp_sc2, data2, direction, interval, points)
        integral = np.sum(np.abs(ect1 - ect2)) * step_size
        supremum = max(supremum, integral)

    return supremum

def rect_metric(simp_comp1, data1, simp_comp2, data2, k=20, interval = (-1., 1.), points: int = 100):
    """
    Computes the ECT metric between two shapes
    :param simp_comp1: Embedded simplicial complex of shape 1
    :param data1: Locations of points of simp_comp1 in Euclidean space
    :param simp_comp2: Embedded simplicial complex of shape 2
    :param data2: Locations of points of simp_comp2 in Euclidean space
    :param k: number of evenly spaced points on S¹ to use for ECT
    :param interval: interval over which to construct Euler curves
    :param points: number of evaluations of the Euler curve
    :param factor: use factor * points many evaluations in interval for integration
    :return: ECT metric between the two shapes
    """
    thetas = np.linspace(0, 2 * np.pi, k + 1)
    step_size = (interval[1] - interval[0]) / (points - 1)
    numba_comp_sc1 = nb.typed.List()
    for s in simp_comp1:
        numba_comp_sc1.append(nb.typed.List(s))
        
    numba_comp_sc2 = nb.typed.List()
    for s in simp_comp2:
        numba_comp_sc2.append(nb.typed.List(s))
    thetas = np.linspace(0, 2 * np.pi, k + 1)
    ect1 = np.empty((k, points), dtype=float)
    ect2 = np.empty((k, points), dtype=float)
    for i in nb.prange(k):
        theta = thetas[i]
        direction = np.array((np.sin(theta), np.cos(theta)))
        ect1[i] = euler_curve(numba_comp_sc1, data1, direction, interval, points)
        ect2[i] = euler_curve(numba_comp_sc2, data2, direction, interval, points)   
    average_ect1 = np.mean(ect1, axis=0)
    average_ect2 = np.mean(ect2, axis=0)
    integral = np.sum(np.abs(average_ect1-average_ect2)) * step_size

    return integral


    
@nb.njit()
def cumulative_euler_curve(simp_comp, data, direction, interval=(-1., 1.), points: int = 100, factor: int = 3):
    """
    Evaluations of smooth Euler curve (i.e. integral of EC - mean on given interval)
    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param direction: Direction (i.e. unit vector) for filtration as tuple
    :param interval: Interval on which to evaluate the Euler curve on
    :param points: Number of evenly spaced points at which to evaluate the Euler curve
    :param factor: use factor * points many evaluations in interval for integration
    :return: 1-D array representing evenly spaced evaluations of smooth Euler characteristic curve on interval
    """
    """
    1. Compute the Euler curve over the specified interval using the euler_curve function. Note that the number of evaluation points is multiplied by factor to provide a denser set of points for the integration process.
    2. Calculate the mean of the Euler curve multiplied by the length of the interval. This seems to be a normalization step, converting the mean of the Euler curve into a mean rate of change over the interval.
    3. Compute the step size for the evaluations based on the specified interval, points, and factor
    4. Calculate the cumulative sum of the Euler curve using np.cumsum(ec). This essentially integrates the Euler curve over the interval and return an array whose i-th term is the sum of the first i elements of the input.
    5. The result is then scaled by the step size and corrected by subtracting a linear function based on the mean rate of change computed earlier. 
       The [::factor] notation takes every factor-th element of the result.
    """
    ec = euler_curve(simp_comp, data, direction=direction, interval=interval, points=points * factor)
    mean = np.mean(ec)
    step_size = (interval[1] - interval[0]) / (points * factor - 1)
    cec = np.cumsum(ec)[::factor] * step_size - np.linspace(0, interval[1] - interval[0], points) * mean

    return cec

@nb.njit(parallel=True)
def _sect_2d(simp_comp, data, k=20, interval=(-1., 1.), points: int = 100, factor: int = 3):
    """
    Currently only for shapes embedded into 2D
    Integrates over cumulative Euler curves over all directions in S¹
    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param k: number of evenly spaced points on S¹ to use for SECT
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :return: 2D array, integrating the above curves over all directions in S¹
    """
    sect = np.empty((k, points), dtype=float)
    thetas = np.linspace(0, 2 * np.pi, k + 1)

    for i in nb.prange(k):
        theta = thetas[i]
        direction = np.array((np.sin(theta), np.cos(theta)))
        sect[i] = cumulative_euler_curve(simp_comp, data, direction, interval, points, factor)


    return sect



def sect_2d(simp_comp, data, k=20, interval=(-1., 1.), points: int = 100, mode='full', factor: int = 3):
    """
    Currently only for shapes embedded into 2D
    Integrates over cumulative Euler curves over all directions in S¹
    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param k: number of random directions to use for SECT
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :param mode: to return full ect or mean over directions
    :return: 1D array, integrating the above curves over all directions in S¹
    """
    numba_comp_sc = nb.typed.List()
    for s in simp_comp:
        numba_comp_sc.append(nb.typed.List(s))

    sect = _sect_2d(numba_comp_sc, data, k, interval, points, factor)

    if mode == 'full':
        return sect
    elif mode == 'mean':
        return sect.mean(axis=0)
    

def sect_metric(simp_comp1, data1, simp_comp2, data2, k=20, interval=(-1., 1.), points: int = 100, factor: int = 3):
    """
    Computes the SECT metric between two shapes
    :param simp_comp1: Embedded simplicial complex of shape 1
    :param data1: Locations of points of simp_comp1 in Euclidean space
    :param simp_comp2: Embedded simplicial complex of shape 2
    :param data2: Locations of points of simp_comp2 in Euclidean space
    :param k: number of random directions to use for SECT
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :return: SECT metric between the two shapes
    """
    sup = 0
    step_size = (interval[1] - interval[0]) / (points - 1)
    sect1 = sect_2d(simp_comp1, data1, k, interval, points, factor=factor)
    sect2 = sect_2d(simp_comp2, data2, k, interval, points, factor=factor)
    for i in range(k):
        integral = np.sum(np.abs(sect1[i] - sect2[i])) * step_size
        sup = max(sup, integral)
    return sup

def detect_metric(simp_comp1, data1, simp_comp2, data2, k=20, interval=(-1., 1.), points: int = 100, factor: int = 3):
    """
    Computes the SECT metric between two shapes
    :param simp_comp1: Embedded simplicial complex of shape 1
    :param data1: Locations of points of simp_comp1 in Euclidean space
    :param simp_comp2: Embedded simplicial complex of shape 2
    :param data2: Locations of points of simp_comp2 in Euclidean space
    :param k: number of random directions to use for SECT
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :return: SECT metric between the two shapes
    """
    
    step_size = (interval[1] - interval[0]) / (points - 1)
    sect1 = sect_2d(simp_comp1, data1, k, interval, points, mode='mean', factor=factor)
    sect2 = sect_2d(simp_comp2, data2, k, interval, points, mode='mean', factor=factor)
    
    integral = np.sum(np.abs(sect1 - sect2)) * step_size

    return integral


def _sect(simp_comp, data, random_state=None, k=20, z_num=20,interval=(-1., 1.), points: int = 100, directions_state = 'Deterministic'):
    """
    Currently only for shapes embedded into 2D, 3D
    Integrates over cumulative Euler curves over all directions in S¹
    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param random_state: Random state to use to randomly sample directions
    :param k: number of evenly spaced points on S¹ to use for averaging
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :return: 1D array, integrating the above curves over all directions in S¹
    """
    
    rng = np.random.RandomState(random_state)
    
    
    if directions_state == 'Random':
        sect = np.empty((k, points), dtype=float)
        directions = np.empty((k, data.shape[1]), dtype=float)
        for i in nb.prange(k):
            direction = rng.random(data.shape[1])
            # while np.linalg.norm(direction) > 1:
            #     direction = rng.random(data.shape[1])

            direction = direction / np.linalg.norm(direction)
            directions[i] = direction

            sect[i] = cumulative_euler_curve(simp_comp, data, direction, interval=interval, points=points, factor=3)

        return sect, directions
    elif directions_state == 'Deterministic':
        sect = np.empty((k * z_num, points), dtype=float)
        directions = np.empty((k * z_num, data.shape[1]), dtype=float)
        phis = np.linspace(0, np.pi, k, endpoint=False)  # phi values for elevation angle
        thetas = np.linspace(0, 2 * np.pi, z_num, endpoint=False)  # theta values for azimuthal angle
        
        idx = 0  # Index for filling in the sect and directions arrays
        for phi in phis:
            for theta in thetas:
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                direction = np.array([x, y, z])
                directions[idx] = direction
                sect[idx] = cumulative_euler_curve(simp_comp, data, direction, interval=interval, points=points, factor=3)
                idx += 1
                return sect, directions


def sect(simp_comp, data, random_state=None, k=20, z_num=20, interval=(-1., 1.), points: int = 100, mode='full', directions_state = 'Deterministic'):
    """
    Currently only for shapes embedded into 2D, 3D
    Integrates over cumulative Euler curves over all directions in S¹
    :param simp_comp: Embedded simplicial complex
    :param data: Locations of points of simp_comp in Euclidean space
    :param random_state: Random state to use to randomly sample directions
    :param k: number of random directions to use for SECT
    :param z_num: number of z values to use for SECT
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :param mode: to return full ect or mean over directions
    :param directions_state: to use random or deterministic directions
    :return: 1D array, integrating the above curves over all directions in S¹
    """
    numba_comp_sc = nb.typed.List()
    for s in simp_comp:
        numba_comp_sc.append(nb.typed.List(s))

    sect, directions = _sect(numba_comp_sc, data, random_state, k, z_num, interval, points, directions_state)

    if mode == 'full':
        return sect, directions
    elif mode == 'mean':
        return sect.mean(axis=0), directions
