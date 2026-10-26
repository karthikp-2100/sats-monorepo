"""IGRF-14 field in approximate J2000 GCI axes.

Install: pip install numpy 'ppigrf>=2.0'
Inputs: geodetic latitude/longitude in degrees, ellipsoid altitude in km,
and t as UTC Julian Date (not MJD). Outputs are magnetic vectors in nT.
GCI includes Earth rotation and precession; nutation and polar motion are
omitted. UT1 and TT are approximated by UTC. Not an exact GCRS transform.
"""
from datetime import datetime, timedelta
import numpy as np
import ppigrf

lambda
# Coordinate transformations: independent of the magnetic-field model.
def enu_to_ecef(b, lat_deg, lon_deg):
    lat, lon = np.deg2rad([lat_deg, lon_deg])
    s, c = np.sin(lat), np.cos(lat)
    sl, cl = np.sin(lon), np.cos(lon)
    return np.array([
        [-sl, -s*cl, c*cl],
        [ cl, -s*sl, c*sl],
        [0.0,    c,    s],
    ]) @ np.asarray(b, dtype=float)


# Transform ECEF to GCI (J2000 mean axes) at UTC Julian Date t.
# Pretty much straight from Markeley
def ecef_to_gci(b, t):
    T = (t - 2451545.0) / 36525.0
    gmst = np.deg2rad((280.46061837 + 360.98564736629*(t-2451545.0)
                      + 0.000387933*T*T - T**3/38710000.0) % 360.0)

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # IAU 1976 precession: J2000 mean axes -> mean axes of date.
    arcsec = np.pi / (180.0 * 3600.0)
    zeta = (2306.2181*T + 0.30188*T*T + 0.017998*T**3)*arcsec
    z = (2306.2181*T + 1.09468*T*T + 0.018203*T**3)*arcsec
    theta = (2004.3109*T - 0.42665*T*T - 0.041833*T**3)*arcsec
    precession = rz(z) @ ry(-theta) @ rz(zeta)
    return precession.T @ rz(gmst) @ np.asarray(b, dtype=float)


# Magnetic-field calculation: ppigrf handles the geodetic model conversion.
def magnetic_field_enu(lat_deg, lon_deg, alt_km, t):
    if not np.all(np.isfinite([lat_deg, lon_deg, alt_km, t])):
        raise ValueError('Inputs must be finite.')
    if not -90.0 < lat_deg < 90.0:
        raise ValueError('Latitude must be strictly between -90 and 90 degrees.')
    date = datetime(2000, 1, 1, 12) + timedelta(days=float(t)-2451545.0)
    if not datetime(1900, 1, 1) <= date <= datetime(2030, 1, 1):
        raise ValueError('Date must be within the bundled IGRF-14 interval, 1900-2030.')
    east, north, up = ppigrf.igrf(lon_deg, lat_deg, alt_km, date)
    field = np.array([east.item(), north.item(), up.item()])
    if not np.all(np.isfinite(field)):
        raise ValueError('Magnetic model returned a nonfinite field.')
    return field


def compute_b_gci(lat_deg, lon_deg, alt_km, t):
    b_enu = magnetic_field_enu(lat_deg, lon_deg, alt_km, t)
    b_ecef = enu_to_ecef(b_enu, lat_deg, lon_deg)
    return ecef_to_gci(b_ecef, t)


if __name__ == '__main__':
    b_gci = compute_b_gci(45.0, -75.0, 400.0, 2460676.5)
    print('B_GCI [nT]:', b_gci)
    print('Direction:', b_gci / np.linalg.norm(b_gci))
