import numpy as np

from . import geometry
from .. import config
import jax.numpy as jnp



class CSGUnion(geometry.Geometry):
    """Construct an object by CSG Union."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} | {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(
            geom1.dim,
            (
                np.minimum(geom1.bbox[0], geom2.bbox[0]),
                np.maximum(geom1.bbox[1], geom2.bbox[1]),
            ),
            geom1.diam + geom2.diam,
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_or(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x)),
        )

    def boundary_normal(self, x):
        return np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))[
            :, np.newaxis
        ] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom2.on_boundary(x), ~self.geom1.inside(x)
        )[
            :, np.newaxis
        ] * self.geom2.boundary_normal(
            x
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = (
                np.random.rand(n, self.dim) * (self.bbox[1] - self.bbox[0])
                + self.bbox[0]
            )
            tmp = tmp[self.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                ~self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[
            on_boundary_geom1
        ]
        on_boundary_geom2 = np.logical_and(
            self.geom2.on_boundary(x), ~self.geom1.inside(x)
        )
        x[on_boundary_geom2] = self.geom2.periodic_point(x, component)[
            on_boundary_geom2
        ]
        return x


class CSGDifference(geometry.Geometry):
    """Construct an object by CSG Difference."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} - {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return jnp.logical_and(self.geom1.inside(x), ~self.geom2.inside(x))

    def on_boundary(self, x):
        return jnp.logical_or(
            jnp.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            jnp.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        return jnp.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))[
            :, jnp.newaxis
        ] * self.geom1.boundary_normal(x) + jnp.logical_and(
            self.geom1.inside(x), self.geom2.on_boundary(x)
        )[
            :, jnp.newaxis
        ] * -self.geom2.boundary_normal(
            x
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[
            on_boundary_geom1
        ]
        return x


class CSGIntersection(geometry.Geometry):
    """Construct an object by CSG Intersection."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} & {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(
            geom1.dim,
            (
                np.maximum(geom1.bbox[0], geom2.bbox[0]),
                np.minimum(geom1.bbox[1], geom2.bbox[1]),
            ),
            min(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        return np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))[
            :, np.newaxis
        ] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom1.inside(x), self.geom2.on_boundary(x)
        )[
            :, np.newaxis
        ] * self.geom2.boundary_normal(
            x
        )

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_point(self, x, component):
        x = np.copy(x)
        on_boundary_geom1 = np.logical_and(
            self.geom1.on_boundary(x), self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_point(x, component)[
            on_boundary_geom1
        ]
        on_boundary_geom2 = np.logical_and(
            self.geom2.on_boundary(x), self.geom1.inside(x)
        )
        x[on_boundary_geom2] = self.geom2.periodic_point(x, component)[
            on_boundary_geom2
        ]
        return x


class CSGMultiDifference(geometry.Geometry):
    """Construct an object by CSG Difference."""

    def __init__(self, geom1, geom2_list):
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2_list = geom2_list

    def not_inside_geom2(self, x):
        return ~jnp.any(jnp.stack([geom2.inside(x) for geom2 in self.geom2_list], axis=1), axis=1)

    def inside(self, x):
        not_in_geom2 = self.not_inside_geom2(x)
        return jnp.logical_and(self.geom1.inside(x), not_in_geom2)

    def on_boundary(self, x):
        not_in_geom2 = self.not_inside_geom2(x)
        return jnp.logical_or(
            jnp.logical_and(self.geom1.on_boundary(x), not_in_geom2),
            jnp.logical_and(
                self.geom1.inside(x),
                jnp.stack([geom2.on_boundary(x) for geom2 in self.geom2_list], axis=1).any(axis=1),
            )
        )

    def random_points(self, n, random='pseudo'):
        x = jnp.empty(shape=(n, self.dim), dtype=config.real(jnp))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.not_inside_geom2(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            # JAX arrays are immutable, so use `at` indexing
            x = x.at[i:i + len(tmp)].set(tmp)
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random='pseudo'):
        x = jnp.empty(shape=(n, self.dim), dtype=config.real(jnp))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[self.not_inside_geom2(geom1_boundary_points)]

            geom2_boundary_points = jnp.concatenate([geom2.random_boundary_points(n, random=random) for geom2 in self.geom2_list], axis=0)
            geom2_boundary_points = geom2_boundary_points[self.geom1.inside(geom2_boundary_points)]

            tmp = jnp.concatenate((geom1_boundary_points, geom2_boundary_points))
            # JAX random permutation requires a key, replace with a static permutation if key unavailable
            import jax
            key = jax.random.PRNGKey(0)  # example static key; replace as needed
            tmp = jax.random.permutation(key, tmp)

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x = x.at[i:i + len(tmp)].set(tmp)
            i += len(tmp)
        return x

    def boundary_normal(self, x):
        not_in_geom2 = self.not_inside_geom2(x)
        res = jnp.logical_and(self.geom1.on_boundary(x), not_in_geom2)[:, jnp.newaxis] * self.geom1.boundary_normal(x)
        for geom2 in self.geom2_list:
            res += jnp.logical_and(self.geom1.inside(x), geom2.on_boundary(x))[:, jnp.newaxis] * -geom2.boundary_normal(x)

        return res