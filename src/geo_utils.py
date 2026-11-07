"""Country reference points, great-circle distances, and GeoGuessr-style scoring.

One (lat, lon) point per country: the midpoint between its capital and its most
populated city (the capital itself when they are the same city). Where several
cities qualify, the entries below use the municipality with the largest
population (Antwerp, Quezon City, New Taipei) and the executive capital
(Pretoria).
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0088
GEOGUESSR_SCALE_KM = 1492.7
GEOGUESSR_MAX_SCORE = 5000.0

# country: (capital, most populated city or None if it is the capital), as (lat, lon).
# Alphabetical, matching sorted(training countries) i.e. the label indices.
CAPITAL_AND_LARGEST_CITY = {
    "Argentina": ((-34.6, -58.38), None),
    "Australia": ((-35.28, 149.13), (-33.87, 151.21)),
    "Austria": ((48.21, 16.37), None),
    "Belgium": ((50.85, 4.35), (51.22, 4.4)),
    "Brazil": ((-15.79, -47.88), (-23.55, -46.63)),
    "Bulgaria": ((42.7, 23.32), None),
    "Canada": ((45.42, -75.7), (43.65, -79.38)),
    "Chile": ((-33.45, -70.67), None),
    "Colombia": ((4.71, -74.07), None),
    "Czechia": ((50.08, 14.44), None),
    "Finland": ((60.17, 24.94), None),
    "France": ((48.86, 2.35), None),
    "Germany": ((52.52, 13.4), None),
    "Greece": ((37.98, 23.73), None),
    "Indonesia": ((-6.21, 106.85), None),
    "Ireland": ((53.35, -6.26), None),
    "Israel": ((31.77, 35.21), None),
    "Italy": ((41.9, 12.5), None),
    "Japan": ((35.68, 139.69), None),
    "Malaysia": ((3.14, 101.69), None),
    "Mexico": ((19.43, -99.13), None),
    "Netherlands": ((52.37, 4.9), None),
    "New Zealand": ((-41.29, 174.78), (-36.85, 174.76)),
    "Norway": ((59.91, 10.75), None),
    "Peru": ((-12.05, -77.04), None),
    "Philippines": ((14.6, 120.98), (14.68, 121.04)),
    "Poland": ((52.23, 21.01), None),
    "Portugal": ((38.72, -9.14), None),
    "Romania": ((44.43, 26.1), None),
    "Russia": ((55.76, 37.62), None),
    "Singapore": ((1.35, 103.82), None),
    "South Africa": ((-25.75, 28.19), (-26.2, 28.05)),
    "South Korea": ((37.57, 126.98), None),
    "Spain": ((40.42, -3.7), None),
    "Sweden": ((59.33, 18.07), None),
    "Taiwan": ((25.03, 121.57), (25.01, 121.47)),
    "Thailand": ((13.76, 100.5), None),
    "Turkey": ((39.93, 32.86), (41.01, 28.98)),
    "United Kingdom": ((51.51, -0.13), None),
    "United States": ((38.91, -77.04), (40.71, -74.01)),
}


def _midpoint(a, b):
    """Great-circle midpoint of two (lat, lon) points, in degrees."""
    (la1, lo1), (la2, lo2) = np.radians(a), np.radians(b)
    bx, by = np.cos(la2) * np.cos(lo2 - lo1), np.cos(la2) * np.sin(lo2 - lo1)
    lat = np.arctan2(np.sin(la1) + np.sin(la2), np.hypot(np.cos(la1) + bx, by))
    lon = lo1 + np.arctan2(by, np.cos(la1) + bx)
    return float(np.degrees(lat)), float((np.degrees(lon) + 540) % 360 - 180)


COUNTRY_COORDS = {name: capital if big is None else _midpoint(capital, big)
                  for name, (capital, big) in CAPITAL_AND_LARGEST_CITY.items()}
COUNTRIES = sorted(COUNTRY_COORDS)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_matrix(countries=None):
    """(N, N) float32 km table, rows/cols in `countries` order (default: alphabetical)."""
    countries = list(COUNTRIES if countries is None else countries)
    missing = [c for c in countries if c not in COUNTRY_COORDS]
    if missing:
        raise KeyError(f"No coordinates for: {missing}")
    lat, lon = np.array([COUNTRY_COORDS[c] for c in countries]).T
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]).astype(np.float32)


def geoguessr_score(distance_km):
    """5000 * exp(-d / 1492.7), elementwise."""
    return GEOGUESSR_MAX_SCORE * np.exp(-np.asarray(distance_km) / GEOGUESSR_SCALE_KM)
