"""Equivalent hexagonal honeycomb properties; restored original paper formulas."""
import math


def material_property(E, G, rho, t, l1, l2, theta):
    if not all(math.isfinite(v) for v in (E, G, rho, t, l1, l2, theta)):
        raise ValueError('Honeycomb inputs must be finite')
    if min(E, G, rho, t, l1, l2) <= 0 or not -math.pi/2 < theta < math.pi/2:
        raise ValueError('Require positive physical inputs and angle in (-pi/2, pi/2)')
    s, c = math.sin(theta), math.cos(theta)
    e1, e2 = l2/l1, t/l1
    if e1+s <= 0:
        raise ValueError('Cell height must be positive')
    tangent = s/c
    E1 = E*e2**3 / ((tangent**2+e2**2)*(e1+s)*c)
    E2 = E*e2**3*(e1+s) / ((c**2+(e1+s**2)*e2**2)*c)
    nu12 = (1-e2**2)*s / ((tangent**2+e2**2)*(e1+s))
    G12 = E*e2**3*(e1+s) / (e1**2*(1+2*e1)*c)
    G13 = G*e2*c/(e1+s)
    G23 = G*e2/(2*c)*((e1+s)/(1+2*e1)+(e1+2*s**2)/(2*(e1+s)))
    rhoc = rho*e2*(1+e1)/((e1+s)*c)
    result = E1,E2,G23,G13,G12,nu12,rhoc
    if not all(math.isfinite(v) for v in result) or min(E1,E2,G23,G13,G12,rhoc) <= 0:
        raise ValueError('Nonphysical equivalent properties')
    if rhoc >= rho or 1-nu12**2*E2/E1 <= 0:
        raise ValueError('Invalid relative density or plane-stress stiffness')
    return result
