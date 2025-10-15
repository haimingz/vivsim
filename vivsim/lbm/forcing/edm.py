
from ..basic import get_equilibrium


def forcing_edm(f, g, u, rho):
    """Apply Exact Difference Method (EDM) forcing scheme.
    
    The Exact Difference Method applies external forces by computing the difference
    between equilibrium distributions at shifted velocities. 
    This approach ensures that the collision step does not need to be modified to account
    for external force, which is super convenient for advanced collision models other than BGK. 
    
    This function implements a modified version of the original EDM proposed by Kupershtokh et al.
    It can be proven that this method is equivalent to Guo's forcing scheme.
    
    Ref: Khazaeli et al. https://doi.org/10.1016/j.camwa.2019.02.032
    
    Args:
        f (jax.Array): Distribution function with shape (9, NX, NY).
        g (jax.Array): External force density with shape (2, NX, NY), where the first
            dimension represents force components in x and y directions.
        u (jax.Array): Fluid velocity field with shape (2, NX, NY), where the first
            dimension represents velocity components in x and y directions.
        rho (jax.Array): Fluid density field with shape (NX, NY).
    
    Returns:
        jax.Array: Updated distribution function with applied forcing term, shape (9, NX, NY).
    """
    
    return f + get_equilibrium(rho, u + g / 2) - get_equilibrium(rho, u - g / 2)