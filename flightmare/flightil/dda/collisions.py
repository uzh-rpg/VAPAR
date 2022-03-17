import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation


def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=-1):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return np.vstack((start, np.vstack((_bresenhamlines(start, end, max_iter)
                                        .reshape(-1, start.shape[-1]), end))))
    # return _bresenhamlines(start, end, max_iter).reshape(-1,start.shape[-1])


def p2vox(p, reso, xl, yl, zl):
    """Return grid coordinates from positions (p), grid x y and z limits,
    and grid resolution (reso)

    p = np.array (n,3), # positions xyz in meters
    reso = np.float : Resolution in meters
    xl = (np.float, np.float),  # X-Axis limits in meters
    yl = (np.float, np.float),  # Y-Axis limits in meters
    zl = (np.float, np.float),  # Z-Axis limits in meters
    """
    return (np.floor((p.reshape((-1, 3)) - np.array([[xl[0], yl[0], zl[0]]]))
                     / reso).astype(int))


def track2colliders(track, reso, xl, yl, zl, pad, zfloor, debug=False):
    """
    Return a 3d occupancy map "colliders" from gate poses (track dataframe).

    reso = np.float : Resolution in meters
    xl = (np.float, np.float),  # X-Axis limits in meters
    yl= (np.float, np.float),  # Y-Axis limits in meters
    zl = (np.float, np.float),  # Z-Axis limits in meters
    pad = np.float,  # Padding to add to the occupancy map in meters
    zfloor = np.float,  # Ground floor level in meters)
    """

    # Make an empty occupancy grid
    colliders = np.zeros((int(np.diff(xl) / reso),
                          int(np.diff(yl) / reso),
                          int(np.diff(zl) / reso),))

    # Enlarge gate dimensions a bit to fall between inner and outer dimensions
    track['dy'] += 0.25
    track['dz'] += 0.25

    # Get gate corners
    p = track.loc[:, ('px', 'py', 'pz')].values
    q = track.loc[:, ('qx', 'qy', 'qz', 'qw')].values
    d = track.loc[:, ('dx', 'dy', 'dz')].values
    cdict = {
        'ul': p + Rotation.from_quat(q).apply(np.array([[0., 0.5, 0.5]]) * d),
        'ur': p + Rotation.from_quat(q).apply(np.array([[0., -0.5, 0.5]]) * d),
        'lr': p + Rotation.from_quat(q).apply(np.array([[0., -0.5, -0.5]]) * d),
        'll': p + Rotation.from_quat(q).apply(np.array([[0., 0.5, -0.5]]) * d),
    }

    # Find occupied coordinates between corner pairs
    c = np.empty((0, 3))
    for c1, c2 in [('ul', 'ur'), ('ur', 'lr'), ('lr', 'll'), ('ll', 'ul')]:
        for i in range(cdict[c1].shape[0]):
            c = np.vstack((c, bresenhamline(
                p2vox(cdict[c1][i:i + 1, :], reso, xl, yl, zl),
                p2vox(cdict[c2][i:i + 1, :], reso, xl, yl, zl))))

    # Fill the occupancy grid
    w = int(pad / reso)
    for i in range(c.shape[0]):
        # print('{}/{}'.format(i,c.shape[0]))
        colliders[int(np.floor(c[i, 0] - w)):int(np.floor(c[i, 0] + w)),
        int(np.floor(c[i, 1] - w)):int(np.floor(c[i, 1] + w)),
        int(np.floor(c[i, 2] - w)):int(np.floor(c[i, 2] + w))] = 1

    # Set voxels below ground floor to occupied
    colliders[:, :, :int((zfloor - zl[0]) / reso)] = 1

    # For debugging: Plot gate corners and occupancy grid
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Plot gate corners
        for n in cdict:
            v = p2vox(cdict[n], reso, xl, yl, zl)
            ax.plot(v[:, 0], v[:, 1], v[:, 2], 'o')
        # Plot occupied coordinates connecting gate corners
        ax.plot(c[:, 0], c[:, 1], c[:, 2], 'xk')
        # Plot the occupancy grid
        ax.voxels(colliders, edgecolors='k', shade=False)
        ax.set_xlim((0, np.diff(xl) / reso))
        ax.set_ylim((0, np.diff(yl) / reso))
        ax.set_zlim((0, np.diff(zl) / reso))
        plt.show()

    return colliders


def check_collision(p, colliders, reso, xl, yl, zl):
    """Returns collision checks (True=collision, False=No collision) for given
    positions (p), collider map (colliders), map resolution and limits."""
    coords = p2vox(p.reshape((-1, 3)), reso, xl, yl, zl)

    in_bounds = np.array([0 <= coords[i, 0] < colliders.shape[0]
                          and 0 <= coords[i, 1] < colliders.shape[1]
                          and 0 <= coords[i, 2] < colliders.shape[2] for i in range(p.shape[0])]).astype(bool)

    return np.array([colliders[coords[i, 0], coords[i, 1], coords[i, 2]] if in_bounds[i] else 1 for i in
                     range(p.shape[0])]).astype(bool) | ~in_bounds


if __name__ == "__main__":
    # Some settings
    reso = 0.1  # Resolution in meters
    xl = (-30, 30)  # X-Axis limits in meters
    yl = (-20, 20)  # Y-Axis limits in meters
    zl = (-1, 5)  # Z-Axis limits in meters
    pad = 0.5  # Padding to add to the occupancy map in meters
    zfloor = 0.  # Ground floor level in meters)
    # Load gate positions
    track = pd.read_csv('track.csv')
    # Compute collider map
    colliders = track2colliders(track, reso, xl, yl, zl, pad, zfloor)
    # Save collider map
    np.save('colliders.npy', colliders)
    # Load collider map
    colliders = np.load('colliders.npy')
    # Load trajectory
    traj = pd.read_csv('trajectory.csv')
    # Compute collision information
    traj['collision'] = check_collision(traj.loc[:, ('px', 'py', 'pz')].values,
                                        colliders, reso, xl, yl, zl)
    # Make a figure showing the collider map (super slow if reso is small)
    to_plot_colliders = False
    if to_plot_colliders:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        fig.set_figwidth(10)
        fig.set_figheight(10)
        ax.voxels(colliders, edgecolors='w', shade=False)
        plt.tight_layout()
        fig.savefig('colliders.jpg')
        plt.close(fig)
        fig = None
        ax = None
    # Make a figure showing collision detection output
    to_plot_output = True
    if to_plot_output:
        plt.figure()
        plt.gcf().set_figwidth(10)
        plt.gcf().set_figheight(8)
        ind = traj['collision'].values == 1
        plt.subplot(3, 1, 1)
        plt.plot(traj['px'].values, traj['py'].values, label='flight path')
        plt.plot(traj.loc[ind, 'px'].values,
                 traj.loc[ind, 'py'].values,
                 'rx', label='collisions')
        plt.xlabel('px [m]')
        plt.ylabel('py [m]')
        plt.legend(loc='upper right')
        plt.subplot(3, 1, 2)
        for n in ['px', 'py', 'pz']:
            plt.plot(traj['t'].values, traj[n].values, label=n)
        plt.legend(loc='upper right')
        plt.ylabel('Position [m]')
        plt.subplot(3, 1, 3)
        plt.plot(traj.t, traj.collision)
        plt.ylabel('Collision (1=True,0=False)')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.savefig('output.jpg')
