
import jax.numpy as jnp

from EAPINN import config
from EAPINN.geometry import geometry


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
