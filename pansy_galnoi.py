# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from matplotlib.dates import DateFormatter
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Galactic, AltAz, EarthLocation
from astropy_healpix import HEALPix
from pygdsm import GlobalSkyModel
from antarrlib import steering_vector, radial, freq2wnum, dB, idB, spherint


# Operation frequency.
FREQUENCY = 47e6

# Beam direction of the PANSY radar (az, ze)
BEAMS = np.deg2rad([
    (0, 0),
    (0, 10),
    (90, 10),
    (180, 10),
    (270, 10)
])

# Color for each beam.
COLORS = ["r", "g", "b", "m", "c"]

# Label for each beam.
BEAMDIRS = "ZNESW"

# Evaluation grid.
ZE = np.linspace(0, 90, 91)
AZ = np.linspace(-180, 180, 361)

# Antenna position.
df = pd.read_csv("antpos.csv")
r = df.loc[df.Ready == 1, ["X(m)", "Y(m)", "Z(m) "]]

# Element pattern.
# index is ze in deg, column is az in deg.
df = pd.read_csv("antptn.csv", header=0, index_col=0).rename(columns=float)
lut = RegularGridInterpolator(
    (np.deg2rad(df.index.values), np.deg2rad(df.columns.values)),
    idB(df.values))


def element(ze, az):
    "Element pattern function."
    return lut(np.c_[np.ravel(ze), np.ravel(az)]).reshape(np.shape(ze))


# Wave number.
k = freq2wnum(FREQUENCY)
# Evaluation grid.
ze_g, az_g = np.deg2rad(np.meshgrid(ZE, AZ, indexing="ij"))

# Compute antenna pattern
patterns = []
for az, ze in tqdm(BEAMS, desc="Antenna pattern"):
    b = radial(ze, az)
    v = radial(ze_g, az_g)
    w = steering_vector(k, r, b) / np.sqrt(len(r))
    a = steering_vector(k, r, v)
    e = element(ze_g, az_g)
    p = np.reshape(
        np.abs(
            w.conjugate().dot(a.transpose() * np.sqrt(e).ravel())
        )**2,
        np.shape(ze_g)
    )
    patterns.append(p)

# Check plot.
fig = plt.figure(figsize=(12, 6))
for ibeam, pat in enumerate(patterns):
    ax = fig.add_subplot(2, 3, ibeam + 1, projection="polar")
    m = ax.pcolormesh(np.deg2rad(AZ), ZE, dB(pat, "max"), vmin=-30)
    fig.colorbar(ax=ax, mappable=m)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 30)
    ax.grid()
fig.tight_layout()

# %%

# The number of threads to be executed.
ncpus = max(mp.cpu_count() * 2 // 3, 1)

# PANSY radar geolocation.
OBSERVER = -69.0066066316, 39.5930902267

UTC = 0 * u.hour
SYOT = 3 * u.hour  # SYOT

# Timezone.
LOCALTIME = UTC

# Target datetme.
dt_0 = Time("2023-03-01 12:00:00")

# Time span in a single side.
toff = 12

# The number of time points to be evaluated.
nt = 24 * 60 + 1

localhours = np.linspace(-toff, toff, nt)

# Galactic noise model.
gsm = GlobalSkyModel("Hz")
m = gsm.generate(FREQUENCY)
hp = HEALPix(nside=gsm.nside, order="RING", frame=Galactic)


lat, lon = OBSERVER * u.deg
az = az_g*u.rad
alt = (np.pi/2 - ze_g) * u.rad

# %%
def func(it):
    coords = AltAz(
        az=az, alt=alt,
        obstime=it[1],
        location=EarthLocation(lat=lat, lon=lon),
    )
    return it[0], hp.interpolate_bilinear_skycoord(coords, m)


# Spherical integral over product with beam pattern and theretical skymap.
pa = spherint.patch_area(np.deg2rad(ZE), np.deg2rad(AZ))
# Final product. For each time and beam, galactic noise level is estimated.
galnoi = np.zeros((nt, len(BEAMS)))
with mp.Pool(ncpus) as pool:
    for i, s in tqdm(pool.imap_unordered(
        func, enumerate(dt_0 + localhours * u.hour - LOCALTIME)
    ), total=nt):
        for j, pat in enumerate(patterns):
            galnoi[i, j] = np.sum(pa * pat * s, axis=(-1, -2)) / np.sum(pa * pat, axis=(-1, -2))
# %%

# Check plot.
plt.figure(figsize=(16, 7))
plt.gca().set_prop_cycle(color="rgbmc")
l = plt.plot((dt_0 + localhours*u.hour).to_datetime(), galnoi)
plt.grid()
plt.gca().xaxis.set_major_formatter(DateFormatter("%dT%H"))
plt.legend(l, "ZNESW")
plt.tight_layout()
plt.show()

# %%
