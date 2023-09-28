"Galactic noise level estimator for antenna array."
import pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from argparse import ArgumentParser, RawTextHelpFormatter
from os import makedirs
from os.path import join
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
DEFAULT_ANTPOS = pkg_resources.resource_filename("pansy_galnoi", "pansy-antpos.csv")
DEFAULT_ANTPTN = pkg_resources.resource_filename("pansy_galnoi", "pansy-antptn.csv")


class functor_rp:
    def __init__(self, ze_g, az_g, k, r, lut):
        self._ze = ze_g
        self._az = az_g
        self._k = k
        self._r = r
        self._lut = lut

    def _element(self, ze, az):
        "Element pattern function."
        return self._lut(np.c_[np.ravel(ze), np.ravel(az)]).reshape(np.shape(ze))

    def __call__(self, it):
        i, (az1, ze1) = it
        b = radial(ze1, np.pi/2 - az1)
        v = radial(self._ze, np.pi/2 - self._az)
        w = steering_vector(self._k, self._r, b) / np.sqrt(len(self._r))
        a = steering_vector(self._k, self._r, v)
        e = self._element(self._ze, self._az)
        p = np.reshape(
            np.abs(
                w.conjugate().dot(a.transpose() * np.sqrt(e).ravel())
            )**2,
            np.shape(self._ze)
        )
        return i, p


class functor_si:
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


def main():
    argp = ArgumentParser(
        prog="pansy-galnoi",
        description=(
            "Compute the theoretical galactic noise temperature observed by "
            "the specified antenna array "
            "(by default assuming the PANSY radar*)."
        ),
        epilog="*PANSY radar - https://pansy.eps.s.u-tokyo.ac.jp/en/index.html",
        formatter_class=RawTextHelpFormatter
    )
    # 
    # Antenna array settings.
    # 
    argp.add_argument(
        "-f", "--frequency",
        action="store",
        type=float,
        help=f"Radar frequency in Hz. Default is '{DEFAULT_FREQUENCY}'.",
        default=DEFAULT_FREQUENCY)
    argp.add_argument(
        "--lat",
        action="store",
        type=float,
        help=(
            "Latitude of the observer.\n"
            f"Default is {DEFAULT_LAT}."
        ),
        default=DEFAULT_LAT
    )
    argp.add_argument(
        "--lon",
        action="store",
        type=float,
        help=(
            "Longitude of the observer.\n"
            f"Default is {DEFAULT_LON}."
        ),
        default=DEFAULT_LON
    )
    argp.add_argument(
        "--antpos",
        action="store",
        type=str,
        help=(
            f"Path to antenna position file.\nDefault is \"{DEFAULT_ANTPOS}\"."
        ),
        default=DEFAULT_ANTPOS
    )
    argp.add_argument(
        "--antptn",
        action="store",
        type=str,
        help=(
            f"""Path to antenna pattern file.\n
            Default is \"{DEFAULT_ANTPTN}\".\n
            Row corresponds to zenith angle, while column is azimuth angle.
            Angles are measured CW from North (same hereafter).
            """
        ),
        default=DEFAULT_ANTPTN
    )
    argp.add_argument(
        "-b", "--beams",
        action="store",
        type=str,
        help=(
            "Beam directions in degrees.\n"
            "Must be a valid Python expression for a list of two-element "
            "tuple [(az, ze), (az, ze), ...].\n"
            f"Default is \"{DEFAULT_BEAMS}\"."
        ),
        default=DEFAULT_BEAMS
    )
    argp.add_argument(
        "-c", "--colors",
        action="store",
        type=str,
        help=(
            "Color specification for each beam.\n"
            "Must be a valid Python expression for a list of matplotlib's "
            "color spec.\n"
            f"Default is \"{DEFAULT_COLORS}\"."
        ),
        default=DEFAULT_COLORS
    )
    argp.add_argument(
        "-l", "--labels",
        action="store",
        type=str,
        help=(
            "Label names for each beam.\n"
            "Must be a valid Python expression for a list of str.\n"
            f"Default is \"{DEFAULT_LABELS}\"."
        ),
        default=DEFAULT_LABELS
    )
    # 
    # Observation time settings.
    # 
    argp.add_argument(
        "-t", "--timezero",
        action="store",
        type=str,
        help=(
            "Time origin in astropy Time format.\nDefault is now in UTC."
        ),
        default=datetime.utcnow().isoformat()
    )
    argp.add_argument(
        "-d", "--duration",
        action="store",
        type=int,
        help=(
            f"Duration in hours.\nDefault is {DEFAULT_DURATION} h."
        ),
        default=DEFAULT_DURATION
    )
    argp.add_argument(
        "-n", "--nt",
        action="store",
        type=int,
        help=(
            "The number of time ticks. Must be in an integer.\n"
            f"Default is {DEFAULT_TIMETICKS}."
        ),
        default=DEFAULT_TIMETICKS
    )
    argp.add_argument(
        "--localtime",
        action="store",
        type=float,
        help=(
            f"Localtime offset in hours.\nDefault is {DEFAULT_LOCALTIME}."
        ),
        default=DEFAULT_LOCALTIME
    )
    # 
    # Miscellaneous settings.
    # 
    argp.add_argument(
        "--ze",
        action="store",
        type=str,
        help=(
            "Zenith directions for evaluating the whole sky.\n"
            "Must be a valid Python expression for a tuple for "
            "`numpy.linspace()` argument: (start, stop, count).\n"
            f"Default is \"{DEFAULT_ZE}\"."
        ),
        default=DEFAULT_ZE
    )
    argp.add_argument(
        "--az",
        action="store",
        type=str,
        help=(
            "Azimuth directions for evaluating the whole sky.\n"
            "Must be a valid Python expression for a tuple for "
            "`numpy.linspace()` argument: (start, stop, count).\n"
            f"Default is \"{DEFAULT_AZ}\"."
        ),
        default=DEFAULT_AZ
    )
    argp.add_argument(
        "-j", "--jobs",
        action="store",
        type=float,
        nargs="?",
        help=(
            f"The number of processes.\n"
            "Default is 1, max is {mp.cpu_count()}."
        ),
        default=1,
        const=mp.cpu_count()
    )
    argp.add_argument(
        "-o", "--output",
        action="store",
        type=str,
        nargs="?",
        help=(
            "Output directory for evaluated galactic noise level measured by "
            "this antenna array.\n"
            "By default, no CSV file is generated and only a graph shows up."
        ),
        default=None,
        const=".",
    )
    argp.add_argument(
        "--check-patterns",
        action="store_true",
        help="Show antenna patterns for debugging."
    )
    argp.add_argument(
        "--show",
        action="store_true",
        help=(
            "Show figure window."
        ),
        default=None
    )
    args = argp.parse_args()

    args.labels = list(eval(f"\"{args.labels}\""))
    args.colors = list(eval(f"\"{args.colors}\""))

    if args.output:
        makedirs(args.output, exist_ok=True)

    if not args.show and not args.output:
        args.show = True

    # The number of threads to be executed.
    if args.jobs in [-1, 0]:
        args.jobs = mp.cpu_count()
    elif not float(args.jobs).is_integer():
        args.jobs = max(int(mp.cpu_count() * args.jobs), 1)
    ncpus = int(args.jobs)

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

    # Wave number.
    k = freq2wnum(args.frequency)
    # Evaluation grid.
    ze_g, az_g = np.deg2rad(np.meshgrid(ze, az, indexing="ij"))

    # Timezone.
    localtime = args.localtime * u.hour

    # Target datetme.
    timezero = Time(args.timezero, scale="utc")

    timeticks = np.linspace(0, args.duration, args.nt)

    # Galactic noise model.
    gsm = GlobalSkyModel("Hz")
    m = gsm.generate(args.frequency)
    hp = HEALPix(nside=gsm.nside, order="RING", frame=Galactic)

    lat = args.lat * u.deg
    lon = args.lon * u.deg
    az0_g = az_g*u.rad
    alt_g = (np.pi/2 - ze_g) * u.rad

    # Spherical integral over product with beam pattern and theretical skymap.
    pa = spherint.patch_area(np.deg2rad(ze), np.deg2rad(az))

    patterns = [None for _ in range(len(beams))]

    # Final product. For each time and beam, galactic noise level is estimated.
    galnoi = np.zeros((args.nt, len(beams)))

    with mp.Pool(ncpus) as pool:
        # Compute antenna pattern
        for i, p in tqdm(
            pool.imap_unordered(
                functor_rp(ze_g, az_g, k, r, lut),
                enumerate(beams)
            ), desc="Antenna pattern", total=len(beams)
        ):
            patterns[i] = p

        for i, s in tqdm(pool.imap_unordered(
            functor_si(lat, lon, az0_g, alt_g, hp, m),
            enumerate(timezero + timeticks * u.hour - localtime)
        ), total=args.nt, desc="Spherical integral"):
            for j, pat in enumerate(patterns):
                galnoi[i, j] = np.sum(pa * pat * s, axis=(-1, -2)) / \
                    np.sum(pa * pat, axis=(-1, -2))

    # Check plot.
    if args.check_patterns:
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
        if args.output:
            fig.savefig(join(args.output, "pattern.png"))

    # Check plot.
    if args.output or args.show:
        fig, ax = plt.subplots(1, 1, figsize=(16, 7))
        ax.set_prop_cycle(color=args.colors)
        l = ax.plot((timezero + timeticks*u.hour).to_datetime(), galnoi)
        ax.grid()
        ax.xaxis.set_major_formatter(DateFormatter("%dT%H"))
        ax.yaxis.set_major_formatter("{x:.0f} K")
        ax.legend(l, args.labels)
        ax.set_title(" - ".join(
            map(
                lambda x: x.strftime("%Y/%m/%d %H:%M:%S"),
                [timezero, timezero + args.duration*u.hour])) + " (UTC" +
                (f"+{args.localtime:g}" if args.localtime != 0 else "") + ")"
            )
        fig.tight_layout()
        if args.output:
            fig.savefig(
                join(args.output, timezero.strftime("%Y%m%d.%H%M%S.png")))

    if args.output:
        pd.DataFrame(galnoi, index=timeticks, columns=args.labels).to_csv(
            join(args.output, timezero.strftime("%Y%m%d.%H%M%S.csv")))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
