"Galactic noise level estimator for antenna array."
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from matplotlib.dates import DateFormatter
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Galactic, AltAz, EarthLocation
from astropy_healpix import HEALPix
from pygdsm import GlobalSkyModel
from antarrlib import steering_vector, radial, freq2wnum, dB, idB, spherint

from argparse import ArgumentParser

DEFAULT_FREQUENCY = 47e6
DEFAULT_BEAMS = "[(0, 0), (0, 10), (90, 10), (180, 10), (270, 10)]"
DEFAULT_COLORS = "rgbmc"
DEFAULT_LABELS = "ZNESW"
DEFAULT_ZE = "(0, 90, 91)"
DEFAULT_AZ = "(-180, 180, 361)"
DEFAULT_LAT = -69.0066066316
DEFAULT_LON = 39.5930902267
DEFAULT_TIMETICKS = 24*60+1
DEFAULT_DURATION = 24
DEFAULT_LOCALTIME = 0
DEFAULT_ANTPOS = "pansy-antpos.csv"
DEFAULT_ANTPTN = "pansy-antptn.csv"


class Functor:
    def __init__(self, lat, lon, az, alt, hp, m):
        self._lat = lat
        self._lon = lon
        self._az = az
        self._alt = alt
        self._hp = hp
        self._m = m

    def __call__(self, it):
        coords = AltAz(
            az=self._az, alt=self._alt,
            obstime=it[1],
            location=EarthLocation(lat=self._lat, lon=self._lon),
        )
        return it[0], self._hp.interpolate_bilinear_skycoord(coords, self._m)


if __name__ == "__main__":
    argp = ArgumentParser(
        prog="antarr-galnoi",
    )
    argp.add_argument(
        "-f", "--frequency",
        action="store",
        type=float,
        help=f"Radar frequency in Hz. Default is '{DEFAULT_FREQUENCY}'.",
        default=DEFAULT_FREQUENCY)
    argp.add_argument(
        "-b", "--beams",
        action="store",
        type=str,
        help=(
            "Beam directions in degrees. Must be a valid Python expression "
            "for a list of two-element tuple [(az, ze), (az, ze), ...]. "
            f"Default is \"{DEFAULT_BEAMS}\"."
        ),
        default=DEFAULT_BEAMS
    )
    argp.add_argument(
        "-c", "--colors",
        action="store",
        type=str,
        help=(
            "Color specification for each beam. "
            "Must be a valid Python expression for a list of matplotlib's "
            "color spec. "
            f"Default is \"{DEFAULT_COLORS}\"."
        ),
        default=DEFAULT_COLORS
    )
    argp.add_argument(
        "-l", "--labels",
        action="store",
        type=str,
        help=(
            "Label names for each beam. "
            "Must be a valid Python expression for a list of str. "
            f"Default is \"{DEFAULT_LABELS}\"."
        ),
        default=DEFAULT_LABELS
    )
    argp.add_argument(
        "--ze",
        action="store",
        type=str,
        help=(
            "Zenith directions for evaluating the whole sky. "
            "Must be a valid Python expression for a tuple for "
            "`numpy.linspace()` argument: (start, stop, count)."
            f"Default is \"{DEFAULT_ZE}\"."
        ),
        default=DEFAULT_ZE
    )
    argp.add_argument(
        "--az",
        action="store",
        type=str,
        help=(
            "Azimuth directions for evaluating the whole sky. "
            "Must be a valid Python expression for a tuple for "
            "`numpy.linspace()` argument: (start, stop, count)."
            f"Default is \"{DEFAULT_AZ}\"."
        ),
        default=DEFAULT_AZ
    )
    argp.add_argument(
        "-o", "--savecsv",
        action="store",
        type=str,
        help=(
            "Output path for evaluated galactic noise level measured by this "
            "antenna array. "
            "By default, no CSV file is generated and only a graph shows up."
        ),
        default=None
    )
    argp.add_argument(
        "--savefig",
        action="store",
        type=str,
        help=(
            "Output path for evaluated galactic noise level measured by this "
            "antenna array. "
            "If specified, instead of displaying a graph, a quicklook in PNG "
            "format is stored at the specified path."
        ),
        default=None
    )
    argp.add_argument(
        "--show",
        action="store_true",
        help=(
            "Show figure window."
        ),
        default=None
    )
    argp.add_argument(
        "--lat",
        action="store",
        type=float,
        help=(
            "Latitude of the observer."
            f"Default is {DEFAULT_LAT}."
        ),
        default=DEFAULT_LAT
    )
    argp.add_argument(
        "--lon",
        action="store",
        type=float,
        help=(
            "Longitude of the observer."
            f"Default is {DEFAULT_LON}."
        ),
        default=DEFAULT_LON
    )
    argp.add_argument(
        "-j", "--jobs",
        action="store",
        type=float,
        help=(
            "The number of processes. Default is 1."
        ),
        default=1
    )
    argp.add_argument(
        "-0", "--timezero",
        action="store",
        type=str,
        help=(
            "Time origin of observation. Default is now."
        ),
        default=Time(datetime.now())
    )
    argp.add_argument(
        "-d", "--duration",
        action="store",
        type=str,
        help=(
            "Observation duration in hours."
        ),
        default=DEFAULT_DURATION
    )
    argp.add_argument(
        "--nt",
        action="store",
        type=int,
        help=(
            "The number of time ticks of observation. Must be in an integer."
            f" Default is {DEFAULT_TIMETICKS}."
        ),
        default=DEFAULT_TIMETICKS
    )
    argp.add_argument(
        "--localtime",
        action="store",
        type=float,
        help=(
            f"Localtime offset in hours. Default is {DEFAULT_LOCALTIME}."
        ),
        default=DEFAULT_LOCALTIME
    )
    argp.add_argument(
        "--antpos",
        action="store",
        type=str,
        help=(
            f"Path to antenna position file. Default is \"{DEFAULT_ANTPOS}\"."
        ),
        default=DEFAULT_ANTPOS
    )
    argp.add_argument(
        "--antptn",
        action="store",
        type=str,
        help=(
            f"Path to antenna pattern file. Default is \"{DEFAULT_ANTPTN}\"."
        ),
        default=DEFAULT_ANTPTN
    )
    args = argp.parse_args()

    beams = np.deg2rad(eval(args.beams))
    ze = np.linspace(*eval(args.ze))
    az = np.linspace(*eval(args.az))

    # Antenna position.
    df = pd.read_csv(args.antpos)
    try:
        r = df.loc[df.Ready == 1, ["X(m)", "Y(m)", "Z(m) "]]
    except AttributeError:
        r = df.iloc[:, [0, 1, 2]]

    # Element pattern.
    # index is ze in deg, column is az in deg.
    df = pd.read_csv(args.antptn, header=0, index_col=0).rename(columns=float)
    lut = RegularGridInterpolator(
        (np.deg2rad(df.index.values), np.deg2rad(df.columns.values)),
        idB(df.values))


    def element(ze, az):
        "Element pattern function."
        return lut(np.c_[np.ravel(ze), np.ravel(az)]).reshape(np.shape(ze))


    # Wave number.
    k = freq2wnum(args.frequency)
    # Evaluation grid.
    ze_g, az_g = np.deg2rad(np.meshgrid(ze, az, indexing="ij"))

    # Compute antenna pattern
    patterns = []
    for az1, ze1 in tqdm(beams, desc="Antenna pattern"):
        b = radial(ze1, az1)
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
    if args.savefig or args.show:
        fig = plt.figure(figsize=(12, 6))
        for ibeam, pat in enumerate(patterns):
            ax = fig.add_subplot(2, 3, ibeam + 1, projection="polar")
            m = ax.pcolormesh(np.deg2rad(az), ze, dB(pat, "max"), vmin=-30)
            fig.colorbar(ax=ax, mappable=m)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_rlim(0, 30)
            ax.grid()
        fig.tight_layout()
        if args.savefig:
            fig.savefig(args.savefig)

    # The number of threads to be executed.
    if args.jobs in [-1, "all", 0]:
        args.jobs = mp.cpu_count()
    elif not args.jobs.is_integer():
        args.jobs = max(int(mp.cpu_count() * args.jobs), 1)
    ncpus = int(args.jobs)

    # Timezone.
    localtime = args.localtime * u.hour

    # Target datetme.
    timezero = Time(args.timezero)

    timeticks = np.linspace(0, args.duration, args.nt)

    # Galactic noise model.
    gsm = GlobalSkyModel("Hz")
    m = gsm.generate(args.frequency)
    hp = HEALPix(nside=gsm.nside, order="RING", frame=Galactic)

    lat = args.lat * u.deg
    lon = args.lon * u.deg
    az_g = az_g*u.rad
    alt_g = (np.pi/2 - ze_g) * u.rad

    # Spherical integral over product with beam pattern and theretical skymap.
    pa = spherint.patch_area(np.deg2rad(ze), np.deg2rad(az))
    # Final product. For each time and beam, galactic noise level is estimated.
    galnoi = np.zeros((args.nt, len(beams)))
    with mp.Pool(ncpus) as pool:
        for i, s in tqdm(pool.imap_unordered(
            Functor(lat, lon, az_g, alt_g, hp, m),
            enumerate(timezero + timeticks * u.hour - localtime)
        ), total=args.nt):
            for j, pat in enumerate(patterns):
                galnoi[i, j] = np.sum(pa * pat * s, axis=(-1, -2)) / np.sum(pa * pat, axis=(-1, -2))

    # Check plot.
    if args.savefig or args.show:
        plt.figure(figsize=(16, 7))
        plt.gca().set_prop_cycle(color=args.colors)
        l = plt.plot((timezero + timeticks*u.hour).to_datetime(), galnoi)
        plt.grid()
        plt.gca().xaxis.set_major_formatter(DateFormatter("%dT%H"))
        plt.legend(l, args.labels)
        plt.tight_layout()
        if args.savefig:
            plt.savefig(args.savefig)

    if args.show:
        plt.show()
