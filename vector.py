#!/usr/bin/env python


# Authors: Dustin Kleckner (2014) and Takumi Matsuzawa (2020)
# =====================================================================
# Vector Operations                                                    |
# =====================================================================


import numpy as np
from scipy.spatial.transform import Rotation as R
import unittest


# ------------------------------------------------------------------------------
# Basic Operations
# ------------------------------------------------------------------------------
def mag(X):
    '''Calculate the length of an array of vectors.'''
    return np.sqrt((np.asarray(X) ** 2).sum(-1))


def mag1(X):
    '''Calculate the length of an array of vectors, keeping the last dimension
    index.'''
    return np.sqrt((np.asarray(X) ** 2).sum(-1))[..., np.newaxis]


def dot(X, Y):
    '''Calculate the dot product of two arrays of vectors.'''
    return (np.asarray(X) * Y).sum(-1)


def dot1(X, Y):
    '''Calculate the dot product of two arrays of vectors, keeping the last
    dimension index'''
    return (np.asarray(X) * Y).sum(-1)[..., np.newaxis]


def norm(X):
    '''Computes a normalized version of an array of vectors.'''
    # The norma of the null vector is fixed to be (1,0,0) this is consistent 
    # with othe function of this module.
    if mag1(X) == 0:
        return np.asarray([1, 0, 0])
    else:
        return np.asarray(X) / mag1(X)


def plus(X):
    '''Return a shifted version of an array of vectors.'''
    return np.roll(X, -1, 0)


def minus(X):
    '''Return a shifted version of an array of vectors.'''
    return np.roll(X, +1, 0)


def cross(X, Y):
    '''Return the cross-product of two vectors.'''
    return np.cross(X, Y)


def proj(X, Y):
    r'''Return the projection of one vector onto another.

    Parameters
    ----------
    X, Y : vector array

    Returns
    -------
    Z : vector array
        :math:`\vec{Z} = \frac{\vec{X} \cdot \vec{Y}}{|Y|^2} \vec{Y}`
    '''
    Yp = norm(Y)
    return dot1(Yp, X) * Yp


def midpoint_delta(X):
    '''Returns center point and vector of each edge of the
     polygon defined by the points.'''
    Xp = plus(X)
    return (Xp + X) / 2., (Xp - X)


def arb_perp(V):
    '''For each vector, return an arbitrary vector that is perpendicular.

    **Note: arbitrary does not mean random!**'''
    p = np.eye(3, dtype=V.dtype)[np.argmin(V, -1)]
    return norm(p - proj(p, V))


def apply_basis(X, B):
    '''Transform each vector into the specified basis/bases.

    Parameters
    ----------
    X : vector array, shape [..., 3]
    B : orthonormal basis array, shape [..., 3, 3]

    Returns
    -------
    Y : vector array, shape [..., 3]
        X transformed into the basis given by B
    '''

    return (np.asarray(X)[..., np.newaxis, :] * B).sum(-1)


# ------------------------------------------------------------------------------
# Building vectors intelligently
# ------------------------------------------------------------------------------
def vec(x=[0], y=[0], z=[0]):
    '''Generate a [..., 3] vector from seperate x, y, z.

    Parameters
    ----------
    x, y, z: array
        coordinates; default to 0, may have any shape

    Returns
    -------
    X : [..., 3] array'''

    x, y, z = map(np.asarray, [x, y, z])

    s = [1] * max([x.ndim, y.ndim, z.ndim])

    for a in (x, y, z):
        s[-a.ndim:] = [max(ss, n) for ss, n in zip(s[-a.ndim:], a.shape)]

    v = np.empty(s + [3], 'd')
    v[..., 0] = x
    v[..., 1] = y
    v[..., 2] = z

    return v


# ------------------------------------------------------------------------------
# Rotations and basis operations
# ------------------------------------------------------------------------------

def rot(a, X=None, cutoff=1E-10):
    '''Rotate points around an arbitrary axis.

    Parameters
    ----------
    a : [..., 3] array
        Rotation vector, will rotate counter-clockwise around axis by an amount
        given be the length of the vector (in radians).  May be a single vector
        or an array of vectors if each point is rotated separately.

    X : [..., 3] array
        Vectors to rotate; if not specified generates a rotation basis instead.

    cutoff : float
        If length of vector less than this value (1E-10 by default), no rotation
        is performed.  (Used to avoid basis errors)

    Returns
    -------
    Y : [..., 3] array
        Rotated vectors or rotation basis.
    '''

    # B = np.eye(3, dtype='d' if X is None else X.dtype)

    a = np.asarray(a)
    if X is None: X = np.eye(3, dtype=a.dtype)

    phi = mag(a)
    if phi.max() < cutoff: return X

    # http://en.wikipedia.org/w/index.php?title=Rotation_matrix#Axis_and_angle
    n = norm(a)
    # The following should not happen anymore since we fixed norm
    if np.isnan(n).any():
        n[np.where(np.isnan(n).any(-1))] = (1, 0, 0)

    B = np.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype)
    c = np.cos(phi)
    s = np.sin(phi)
    C = 1 - c

    for i in range(3):
        for j in range(3):
            if i == j:
                extra = c
            else:
                if (j - i) % 3 == 2:
                    extra = +s * n[..., (j - 1) % 3]
                else:
                    extra = -s * n[..., (j + 1) % 3]

            B[..., i, j] = n[..., i] * n[..., j] * C + extra

    ##Create a new basis where the rotation is simply in x
    # Ba = normalize_basis(a[..., np.newaxis, :])
    #
    ###B = apply_basis(B, Ba) #This was pointless, B was 1
    ##y, z = B[..., 1, :].copy(), B[..., 2, :].copy()
    ##
    ##c, s = np.cos(phi), np.sin(phi)
    ##
    ###Rotate in new basis
    ##B[..., 1, :] = y*c - z*s
    ##B[..., 2, :] = y*s + z*c
    ##
    ##B = apply_basis(B, Ba.T).T
    #
    # B = np.zeros_like(Ba)
    # B[..., 0, 0] = 1
    # B[..., 1, 1] = +cos(phi)
    # B[..., 1, 2] = -sin(phi)
    # B[..., 2, 1] = +sin(phi)
    # B[..., 2, 2] = +cos(phi)
    #
    # B = apply_basis(B, Ba)

    if X is not None:
        return apply_basis(X, B)
    else:
        return B


def normalize_basis(B):
    '''Create right-handed orthonormal basis/bases from input basis.

    Parameters
    ----------
    B : [..., 1-3, 3] array
        input basis, should be at least 2d.  If the second to last axis has
        1 vectors, it will automatically create an arbitrary orthonormal basis
        with the specified first vector.
        (note: even if three bases are specified, the last is always ignored,
        and is generated by a cross product of the first two.)

    Returns
    -------
    NB : [..., 3, 3] array
        orthonormal basis
    '''

    B = np.asarray(B)
    NB = np.empty(B.shape[:-2] + (3, 3), dtype='d')

    v1 = norm(B[..., 0, :])
    if np.isnan(v1).any():
        v1[np.where(np.isnan(v1).any(-1))] = (1, 0, 0)

    v2 = B[..., 1, :] if B.shape[-2] >= 2 else np.eye(3)[np.argmin(abs(v1), axis=-1)]
    v2 = norm(v2 - v1 * dot1(v1, v2))
    v3 = cross(v1, v2)

    for i, v in enumerate([v1, v2, v3]): NB[..., i, :] = v

    return NB


def get_an_orthonormal_basis(dim, v1=None):
    """
    Returns an orthonormal basis of R^dim
    ... One can choose one of the basis vector (v1) to span the R^dim space.

    Parameters
    ----------
    dim: int > 0
    v1: array-like with shape (dim, ) or len(v1)=dim

    Returns
    -------
    basis: numpy array with shape (dim, dim)
        ... basis vectors are stored as basis[0, :],  basis[1, :], ...,  basis[dim-1, :]
        ... If v1 is given, the normalized v1 is stored in basis[0, :]

    """
    import time
    t0 = time.time()
    # INITIALIZATION
    basis = np.zeros((dim, dim))
    if v1 is None:
        # Choose some random vector as the first basis vector
        basis[:, 0] = norm(np.random.random(dim))
    else:
        basis[:, 0] = norm(v1)

    m = 1  # number of basis vectors constructed during this algorithm
    while m < dim:
        # 1. Prepare a random vector, r
        r = np.random.random(dim)
        t1 = time.time()
        if t1 - t0 > 5:
            print('Something failed in get_an_orthonormal_basis. Aborting...')
            print(dim, v1, r)
            sys.exit()
        # 2. Make sure that r and the basis vectors, which are constructed so far, are linearly independent
        checker = False
        while not checker:
            checker = True
            for i in range(m):
                checker *= isLinearlyIndependent(r, basis[:, i])
        # 3. Construct the next basis vector
        for i in range(m):
            r -= dot(r, basis[:, i]) * basis[:, i]
        basis[:, m] = norm(r)
        m += 1
    return basis


def apply_right_handedness(basis, thd=10 ** -10):
    """
    Convert 3D basis to be right-handed
    If 2D basis were given, it converts the basis such that
    the cross product of the two vectors would point out of the plane
    like \hat{r} and \hat{phi} in the polar coordinate system.
    """
    if basis.shape[1] > 3 or basis.shape[1] < 2:
        raise ValueError("given vectors must have dimensions 2 or 3")
    elif basis.shape[1] == 2:
        cpd = cross(norm(basis[:, 0]), norm(basis[:, 1]))
        if np.abs(cpd + 1) < thd:
            basis[:, 1] = -basis[:, 1]
        return basis
    else:
        cpd = cross(norm(basis[:, 0]), norm(basis[:, 1]))
        if np.abs(cpd + norm(basis[:, 2])) < thd:
            basis[:, 2] = -basis[:, 2]
        return basis


def get_perp_vectors_3d(v0, n=10):
    """
    Returns n vectors on the plane defined by the vector v0.
    ... v0 does not have to be normalized.
    ... In other words, this returns n vectors perpendicular to the 3D vector v0.

    Parameters
    ----------
    v0: array-like with shape (3, ) or len(v1)=3
    n: int
        ... number of vectors on the plane defined by the vector v0 to be output.

    Returns
    -------
    v1s: array with shape (n, 3)
        ... 3D vectors on the plane defined by the vector v0.

    """
    # get an orthonormal basis which spans the R^3 space
    basis = get_an_orthonormal_basis(3, v1=v0)
    basis = apply_right_handedness(basis)
    # basis[:, 1]  and basis[:, 2] are the orthonormal vectors that spans the plane (embedded in R^3) defined by v0 = basis[:, 0]
    v1 = basis[:, 1]  # Use basis[:, 1] as a convention.
    v1s = np.empty((n, 3))
    # now rotate v1 about the axis co-directional to the v0 vector
    for i in range(n):
        r = R.from_rotvec(2 * np.pi / n * i * basis[:, 0])  # norm of the rotvec indicates the rotational angle
        v1s[i, :] = r.apply(v1)
    return v1s


# Linear independence of two vectors
def isLinearlyIndependent(X, Y, thd=10 ** -10, verbose=False):
    """ Checks if two vectors X, Y are linearly independent using the Cauchy-Schwarz inequality
    ... If one of the inout vectors were a null vector, returns False (Independent)
    ... If both input vectors were null vectors, it returns False (Dependent)"""
    if mag1(X) == 0 or mag1(Y) == 0:
        # raise ValueError('null vector is entered!')
        if verbose:
            print('null vector is entered! Returning False')
        return False
    X_norm = norm(X)
    Y_norm = norm(Y)

    difference = np.dot(X_norm, Y_norm) - mag1(X_norm) * mag1(Y_norm)
    if np.abs(difference) < thd:
        return False
    else:
        return True


# misc.
def get_rotation_matrix_between_two_vectors(a, b):
    """
    Returns a 3D rotation matrix R that rotates a unit vector of a onto a unit vector of b
    If a, b are normalized, then R is a matrix such that b = Ra.

    Parameters
    ----------
    a: array-like with shape (3,)
    b: array-like with shape (3,)

    Returns
    -------
    R: array with shape (3,3)
    ... rotation matrix R

    """
    a, b = vec.norm(a), vec.norm(b)
    v = vec.cross(a, b)
    s = vec.mag1(v)
    c = vec.dot(a, b)

    A = np.asarray([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    I = np.asarray([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    R = I + A + np.matmul(A, A) * (1 - c) / s ** 2
    return R


class TestCases(unittest.TestCase):
    def test_construct_vec(self):
        x = np.array([[4., 5., 6.], [1., 2., 3.]])
        y = np.array([[1.], [4.]])
        z = np.array([0.])
        v1 = vec(x, y, z)
        self.assertTrue(v1.shape, (2, 2, 3))
        self.assertEqual(v1.tolist(), [[[4., 1., 0.], [5., 1., 0.], [6., 1., 0.]],
                                       [[1., 4., 0.], [2., 4., 0.], [3., 4., 0.]]])

    def test_rotation(self):
        x = np.array([1., 0., 0.])
        a = vec(z=np.linspace(0, np.pi, 5))
        y = rot(a, x)
        rot_x = np.array([[1., 0., 0.], [1. / np.sqrt(2.), 1. / np.sqrt(2), 0.],
                          [0., 1., 0.], [-1. / np.sqrt(2.), 1. / np.sqrt(2.), 0.], [-1., 0., 0.]])
        self.assertTrue((np.round(abs(y - rot_x), 5) == 0).all())

    def test_normalize(self):
        x = np.array([1., 1., 0])
        y = np.array([-1., 0., 0])
        NB = normalize_basis([x, y])
        normalized_basis = np.array([[1. / np.sqrt(2.), 1. / np.sqrt(2.), 0],
                                     [-1. / np.sqrt(2.), 1. / np.sqrt(2.), 0], [0., 0., 1.]])
        self.assertTrue((np.round(abs(NB - normalized_basis), 5) == 0).all())


if __name__ == '__main__':
    unittest.main()