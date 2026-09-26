"""
Convert Filterbank files to PSRFITS file

Original Source: https://github.com/rwharton/fil2psrfits
"""

import logging
import os
from operator import index

import astropy.coordinates as coord
import astropy.time as time
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)
import json


class ObsInfo(object):
    """
    Class to setup observation info for psrfits header

    """

    def __init__(self):
        self.file_date = self.format_date(time.Time.now().isot)
        self.observer = ""
        self.proj_id = ""
        self.obs_date = ""
        self.fcenter = 0.0
        self.bw = 0.0
        self.nchan = 0
        self.src_name = ""
        self.ra_str = "00:00:00"
        self.dec_str = "+00:00:00"
        self.bmaj_deg = None
        self.bmin_deg = None
        self.bpa_deg = None
        self.scan_len = 0
        self.stt_imjd = 0
        self.stt_smjd = 0
        self.stt_offs = 0.0
        self.stt_lst = None

        self.dt = 0.0
        self.nbits = 16
        self.nsuboffs = 0.0
        self.chan_bw = 0.0
        self.nsblk = 0

        self.telescope = ""
        self.ant_x = None
        self.ant_y = None
        self.ant_z = None
        self.longitude = None
        self.frontend = ""
        self.nrcvr = None
        self.fd_poln = ""
        self.fd_hand = None
        self.fd_sang = None
        self.fd_xyph = None
        self.backend = ""
        self.beconfig = ""
        self.be_phase = None
        self.be_dcc = None
        self.be_delay = None
        self.tcycle = None
        self.npoln = 1
        self.poln_order = "AA+BB"

    def calc_longitude(self):
        xyz = (self.ant_x, self.ant_y, self.ant_z)
        try:
            if any(value is None or not np.isfinite(value) for value in xyz):
                return None
        except TypeError:
            return None
        return coord.EarthLocation.from_geocentric(*xyz, unit="m").lon.deg

    def fill_from_mjd(self, mjd):
        stt_imjd = int(mjd)
        stt_smjd = int((mjd - stt_imjd) * 24 * 3600)
        stt_offs = ((mjd - stt_imjd) * 24 * 3600.0) - stt_smjd
        self.stt_imjd = stt_imjd
        self.stt_smjd = stt_smjd
        self.stt_offs = stt_offs
        self.obs_date = self.format_date(time.Time(mjd, format="mjd").isot)

    def fill_freq_info(self, fcenter, nchan, chan_bw):
        self.fcenter = fcenter
        self.bw = nchan * chan_bw
        self.nchan = nchan
        self.chan_bw = chan_bw

    def fill_source_info(self, src_name, ra_str, dec_str):
        self.src_name = src_name
        self.ra_str = ra_str
        self.dec_str = dec_str

    def fill_beam_info(self, bmaj_deg, bmin_deg, bpa_deg):
        self.bmaj_deg = bmaj_deg
        self.bmin_deg = bmin_deg
        self.bpa_deg = bpa_deg

    def fill_data_info(self, dt, nbits):
        self.dt = dt
        self.nbits = nbits

    def calc_start_lst(self, mjd):
        self.stt_lst = (
            None if self.longitude is None else self.calc_lst(mjd, self.longitude)
        )

    def calc_lst(self, mjd, longitude):
        gfac0 = 6.697374558
        gfac1 = 0.06570982441908
        gfac2 = 1.00273790935
        gfac3 = 0.000026
        mjd0 = 51544.5  # MJD at 2000 Jan 01 12h
        H = (mjd - int(mjd)) * 24  # Hours since previous 0h
        D = mjd - mjd0  # Days since MJD0
        D0 = int(mjd) - mjd0  # Days between MJD0 and prev 0h
        T = D / 36525.0  # Number of centuries since MJD0
        gmst = gfac0 + gfac1 * D0 + gfac2 * H + gfac3 * T**2.0
        lst = ((gmst + longitude / 15.0) % 24.0) * 3600.0
        return lst

    def format_date(self, date_str):
        # Strip out the decimal seconds
        out_str = date_str.split(".")[0]
        return out_str

    def set_pol(self, npol=1, poln_order="AA+BB"):
        # set number of polarisations and type
        self.npoln = npol
        self.poln_order = poln_order

    def fill_primary_header(self):
        p_hdr = fits.Header()
        p_hdr["HDRVER"] = (
            "3.4             ",
            "Header version                               ",
        )
        p_hdr["FITSTYPE"] = ("PSRFITS", "FITS definition for pulsar data files        ")
        p_hdr["DATE"] = (
            self.file_date,
            "File creation date (YYYY-MM-DDThh:mm:ss UTC) ",
        )
        p_hdr["OBSERVER"] = (
            self.observer,
            "Observer name(s)                             ",
        )
        p_hdr["PROJID"] = (
            self.proj_id,
            "Project name                                 ",
        )
        p_hdr["TELESCOP"] = (
            self.telescope,
            "Telescope name                               ",
        )
        p_hdr["ANT_X"] = (self.ant_x, "[m] Antenna ITRF X-coordinate (D)            ")
        p_hdr["ANT_Y"] = (self.ant_y, "[m] Antenna ITRF Y-coordinate (D)            ")
        p_hdr["ANT_Z"] = (self.ant_z, "[m] Antenna ITRF Z-coordinate (D)            ")
        p_hdr["FRONTEND"] = (
            self.frontend,
            "Rx and feed ID                               ",
        )
        p_hdr["NRCVR"] = (self.nrcvr, "Number of receiver polarisation channels     ")
        p_hdr["FD_POLN"] = (
            self.fd_poln,
            "LIN or CIRC                                  ",
        )
        p_hdr["FD_HAND"] = (
            self.fd_hand,
            "+/- 1. +1 is LIN:A=X,B=Y, CIRC:A=L,B=R (I)   ",
        )
        p_hdr["FD_SANG"] = (
            self.fd_sang,
            "[deg] FA of E vect for equal sigma in A&B (E)  ",
        )
        p_hdr["FD_XYPH"] = (
            self.fd_xyph,
            "[deg] Phase of A^* B for injected cal (E)    ",
        )
        p_hdr["BACKEND"] = (
            self.backend,
            "Backend ID                                   ",
        )
        p_hdr["BECONFIG"] = (
            self.beconfig,
            "Backend configuration file name              ",
        )
        p_hdr["BE_PHASE"] = (
            self.be_phase,
            "0/+1/-1 BE cross-phase:0 unknown,+/-1 std/rev",
        )
        p_hdr["BE_DCC"] = (self.be_dcc, "0/1 BE downconversion conjugation corrected  ")
        p_hdr["BE_DELAY"] = (
            self.be_delay,
            "[s] Backend propn delay from digitiser input ",
        )
        p_hdr["TCYCLE"] = (self.tcycle, "[s] On-line cycle time (D)                   ")
        p_hdr["OBS_MODE"] = ("SEARCH", "(PSR, CAL, SEARCH)                           ")
        p_hdr["DATE-OBS"] = (
            self.obs_date,
            "Date of observation (YYYY-MM-DDThh:mm:ss UTC)",
        )
        p_hdr["OBSFREQ"] = (
            self.fcenter,
            "[MHz] Centre frequency for observation       ",
        )
        p_hdr["OBSBW"] = (self.bw, "[MHz] Bandwidth for observation              ")
        p_hdr["OBSNCHAN"] = (
            self.nchan,
            "Number of frequency channels (original)      ",
        )
        p_hdr["CHAN_DM"] = (0.0, "DM used to de-disperse each channel (pc/cm^3)")
        p_hdr["SRC_NAME"] = (
            self.src_name,
            "Source or scan ID                            ",
        )
        p_hdr["COORD_MD"] = ("J2000", "Coordinate mode (J2000, GAL, ECLIP, etc.)    ")
        p_hdr["EQUINOX"] = (2000.0, "Equinox of coords (e.g. 2000.0)              ")
        p_hdr["RA"] = (self.ra_str, "Right ascension (hh:mm:ss.ssss)              ")
        p_hdr["DEC"] = (self.dec_str, "Declination (-dd:mm:ss.sss)                  ")
        p_hdr["BMAJ"] = (self.bmaj_deg, "[deg] Beam major axis length                 ")
        p_hdr["BMIN"] = (self.bmin_deg, "[deg] Beam minor axis length                 ")
        p_hdr["BPA"] = (self.bpa_deg, "[deg] Beam position angle                    ")
        p_hdr["STT_CRD1"] = (
            self.ra_str,
            "Start coord 1 (hh:mm:ss.sss or ddd.ddd)      ",
        )
        p_hdr["STT_CRD2"] = (
            self.dec_str,
            "Start coord 2 (-dd:mm:ss.sss or -dd.ddd)     ",
        )
        p_hdr["TRK_MODE"] = ("TRACK", "Track mode (TRACK, SCANGC, SCANLAT)          ")
        p_hdr["STP_CRD1"] = (
            self.ra_str,
            "Stop coord 1 (hh:mm:ss.sss or ddd.ddd)       ",
        )
        p_hdr["STP_CRD2"] = (
            self.dec_str,
            "Stop coord 2 (-dd:mm:ss.sss or -dd.ddd)      ",
        )
        p_hdr["SCANLEN"] = (
            self.scan_len,
            "[s] Requested scan length (E)                ",
        )
        p_hdr["FD_MODE"] = ("FA", "Feed track mode - FA, CPA, SPA, TPA          ")
        p_hdr["FA_REQ"] = (0.0, "[deg] Feed/Posn angle requested (E)          ")
        p_hdr["CAL_MODE"] = ("OFF", "Cal mode (OFF, SYNC, EXT1, EXT2)             ")
        p_hdr["CAL_FREQ"] = (0.0, "[Hz] Cal modulation frequency (E)            ")
        p_hdr["CAL_DCYC"] = (0.0, "Cal duty cycle (E)                           ")
        p_hdr["CAL_PHS"] = (0.0, "Cal phase (wrt start time) (E)               ")
        p_hdr["STT_IMJD"] = (
            self.stt_imjd,
            "Start MJD (UTC days) (J - long integer)      ",
        )
        p_hdr["STT_SMJD"] = (
            self.stt_smjd,
            "[s] Start time (sec past UTC 00h) (J)        ",
        )
        p_hdr["STT_OFFS"] = (
            self.stt_offs,
            "[s] Start time offset (D)                    ",
        )
        p_hdr["STT_LST"] = (
            self.stt_lst,
            "[s] Start LST (D)                            ",
        )
        return p_hdr

    def fill_table_header(self):
        t_hdr = fits.Header()
        t_hdr["INT_TYPE"] = ("TIME", "Time axis (TIME, BINPHSPERI, BINLNGASC, etc)   ")
        t_hdr["INT_UNIT"] = ("SEC", "Unit of time axis (SEC, PHS (0-1), DEG)        ")
        t_hdr["SCALE"] = ("FluxDen", "Intensity units (FluxDen/RefFlux/Jansky)       ")
        t_hdr["NPOL"] = (self.npoln, "Nr of polarisations                            ")
        t_hdr["POL_TYPE"] = (
            self.poln_order,
            "Polarisation identifier (e.g., AABBCRCI, AA+BB)",
        )
        t_hdr["TBIN"] = (self.dt, "[s] Time per bin or sample                     ")
        t_hdr["NBIN"] = (1, "Nr of bins (PSR/CAL mode; else 1)              ")
        t_hdr["NBIN_PRD"] = (0, "Nr of bins/pulse period (for gated data)       ")
        t_hdr["PHS_OFFS"] = (0.0, "Phase offset of bin 0 for gated data           ")
        t_hdr["NBITS"] = (self.nbits, "Nr of bits/datum (SEARCH mode 'X' data, else 1)")
        t_hdr["NSUBOFFS"] = (
            self.nsuboffs,
            "Subint offset (Contiguous SEARCH-mode files)   ",
        )
        t_hdr["NCHAN"] = (self.nchan, "Number of channels/sub-bands in this file      ")
        t_hdr["CHAN_BW"] = (
            self.chan_bw,
            "[MHz] Channel/sub-band width                   ",
        )
        t_hdr["NCHNOFFS"] = (0, "Channel/sub-band offset for split files        ")
        t_hdr["NSBLK"] = (self.nsblk, "Samples/row (SEARCH mode, else 1)              ")
        return t_hdr


def initialize_psrfits(
    outfile,
    your_object,
    npsub=-1,
    nstart=None,
    nsamp=None,
    chan_freqs=None,
    npoln=1,
    poln_order="AA+BB",
    data_reader=None,
    chunk_rows=10,
):
    """Create a PSRFITS file, optionally filling DATA as bounded row chunks.

    ``data_reader`` receives ``(start_sample, nsamp)`` and must return data
    shaped ``(nsamp, npoln, nchans)``. Without it this keeps the public
    initializer's zero-filled DATA behavior.
    """
    nbits = your_object.your_header.nbits
    tsamp = your_object.your_header.tsamp
    nstart = 0 if nstart is None else index(nstart)
    nsamp = None if nsamp is None else index(nsamp)
    npsub = index(npsub)
    chunk_rows = index(chunk_rows)
    if npsub != -1 and npsub <= 0:
        raise ValueError("npsub must be positive or -1 for automatic sizing")
    sources = getattr(your_object, "your_file", your_object.your_header.filename)
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    for source in sources:
        if os.path.realpath(outfile) == os.path.realpath(source) or (
            os.path.exists(outfile)
            and os.path.exists(source)
            and os.path.samefile(outfile, source)
        ):
            raise ValueError("PSRFITS output must not overwrite its input")
    if nstart < 0:
        raise ValueError("nstart must be non-negative")
    available = max(0, your_object.your_header.nspectra - nstart)
    nsamps = available if nsamp is None else min(nsamp, available)
    if nsamps < 0:
        raise ValueError("nsamp must be non-negative")
    if nsamp is not None and nsamp > available:
        logging.warning(
            "Data requested exceeds the length of file. Reading data till end of file."
        )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    mjd = your_object.your_header.tstart + nstart * tsamp / (24 * 60 * 60)

    if chan_freqs is None:
        chan_freqs = your_object.chan_freqs
    chan_freqs = np.asarray(chan_freqs)
    if not len(chan_freqs):
        raise ValueError("chan_freqs must contain at least one channel")
    nchans = len(chan_freqs)
    fcenter = (chan_freqs[0] + chan_freqs[-1]) / 2
    foff = chan_freqs[1] - chan_freqs[0] if nchans > 1 else your_object.your_header.foff

    if npoln == 4:
        if your_object.your_header.npol == 4:
            nifs = 4
        else:
            logger.warning(
                f"Number of polarisations in the data {your_object.your_header.npol} is not equal to 4."
                "Only writing 1 polarisation."
            )
            nifs = 1
    elif npoln == 1:
        nifs = 1
    else:
        raise ValueError(
            "npoln can only be 1 (for one polarisation) or 4 (for all polarisations)."
        )

    src_name = your_object.your_header.source_name
    from astropy.coordinates import SkyCoord

    ra = your_object.your_header.ra_deg
    dec = your_object.your_header.dec_deg
    ra = 0.0 if ra is None else ra
    dec = 0.0 if dec is None else dec
    loc = SkyCoord(ra, dec, unit="deg")
    ra_str = loc.ra.to_string(unit="hourangle", sep=":", pad=True, precision=4)
    dec_str = loc.dec.to_string(
        unit="deg", sep=":", pad=True, alwayssign=True, precision=4
    )

    d = ObsInfo()
    input_header = your_object.fits[0].header if your_object.format == "fits" else None
    provenance_cards = (
        "OBSERVER",
        "PROJID",
        "TELESCOP",
        "ANT_X",
        "ANT_Y",
        "ANT_Z",
        "FRONTEND",
        "NRCVR",
        "FD_POLN",
        "FD_HAND",
        "FD_SANG",
        "FD_XYPH",
        "BACKEND",
        "BECONFIG",
        "BE_PHASE",
        "BE_DCC",
        "BE_DELAY",
        "TCYCLE",
        "BMAJ",
        "BMIN",
        "BPA",
    )
    if input_header is not None:
        d.ant_x, d.ant_y, d.ant_z = (
            input_header.get(k) for k in ("ANT_X", "ANT_Y", "ANT_Z")
        )
        d.longitude = d.calc_longitude()
    else:
        d.telescope = {1: "Arecibo", 6: "GBT"}.get(
            getattr(your_object, "telescope_id", None), ""
        )
    d.fill_from_mjd(mjd)
    d.fill_freq_info(fcenter, nchans, foff)
    d.fill_source_info(src_name, ra_str, dec_str)
    d.fill_beam_info(None, None, None)
    d.fill_data_info(tsamp, nbits)
    d.calc_start_lst(mjd)
    d.set_pol(npol=nifs, poln_order=poln_order)

    n_per_subint = npsub if npsub > 0 else max(1, int(1.0 / tsamp))
    n_subints = (nsamps + n_per_subint - 1) // n_per_subint
    t_subint = n_per_subint * tsamp
    d.nsblk = n_per_subint
    d.scan_len = t_subint * n_subints

    logger.info(
        f"Setting the following info to be written in {outfile} \n {json.dumps(vars(d), indent=4, sort_keys=True)}"
    )
    phdr = d.fill_primary_header()
    if input_header is not None:
        for card in provenance_cards:
            if card in input_header:
                phdr[card] = input_header[card]
        for card in (
            "OBSERVER",
            "PROJID",
            "TELESCOP",
            "FRONTEND",
            "FD_POLN",
            "BACKEND",
            "BECONFIG",
        ):
            if phdr[card] is None:
                phdr[card] = ""
    else:
        phdr.add_history(
            "SIGPROC telescope_id=%r machine_id=%r"
            % (
                getattr(your_object, "telescope_id", None),
                getattr(your_object, "machine_id", None),
            )
        )
    thdr = d.fill_table_header()
    ra_deg, dec_deg = ra, dec
    l_deg, b_deg = your_object.your_header.gl, your_object.your_header.gb
    l_deg = np.nan if l_deg is None else l_deg
    b_deg = np.nan if b_deg is None else b_deg

    dtype = np.dtype(your_object.your_header.dtype)
    data_format = {
        np.dtype(np.uint8): "B",
        np.dtype(np.int16): "I",
        np.dtype(np.int32): "J",
        np.dtype(np.int64): "K",
        np.dtype(np.float32): "E",
        np.dtype(np.float64): "D",
    }.get(dtype, "E")
    columns = [
        fits.Column(name="TSUBINT", format="1D", unit="s"),
        fits.Column(name="OFFS_SUB", format="1D", unit="s"),
        fits.Column(name="LST_SUB", format="1D", unit="s"),
        fits.Column(name="RA_SUB", format="1D", unit="deg"),
        fits.Column(name="DEC_SUB", format="1D", unit="deg"),
        fits.Column(name="GLON_SUB", format="1D", unit="deg"),
        fits.Column(name="GLAT_SUB", format="1D", unit="deg"),
        fits.Column(name="FD_ANG", format="1E", unit="deg"),
        fits.Column(name="POS_ANG", format="1E", unit="deg"),
        fits.Column(name="PAR_ANG", format="1E", unit="deg"),
        fits.Column(name="TEL_AZ", format="1E", unit="deg"),
        fits.Column(name="TEL_ZEN", format="1E", unit="deg"),
        fits.Column(name="DAT_FREQ", format=f"{nchans}E", unit="MHz"),
        fits.Column(name="DAT_WTS", format=f"{nchans}E"),
        fits.Column(name="DAT_OFFS", format=f"{nchans}E"),
        fits.Column(name="DAT_SCL", format=f"{nchans}E"),
        fits.Column(
            name="DATA",
            format=f"{nifs * nchans * n_per_subint}{data_format}",
            dim=f"({nchans}, {nifs}, {n_per_subint})",
        ),
    ]
    table_hdu = fits.BinTableHDU.from_columns(columns, header=thdr, nrows=0)
    table_hdu.header["EXTNAME"] = "SUBINT"
    table_hdu.header["NAXIS2"] = n_subints
    primary_hdu = fits.PrimaryHDU(header=phdr)
    primary_hdu.header["EXTEND"] = True

    record_dtype = table_hdu.data.dtype.newbyteorder(">")

    logging.info(f"Writing PSRFITS table to file: {outfile}")
    with open(outfile, "wb") as fits_file:
        fits_file.write(
            primary_hdu.header.tostring(sep="", endcard=True, padding=True).encode(
                "ascii"
            )
        )
        fits_file.write(
            table_hdu.header.tostring(sep="", endcard=True, padding=True).encode(
                "ascii"
            )
        )
        for first_row in range(0, n_subints, chunk_rows):
            rows = min(chunk_rows, n_subints - first_row)
            records = np.zeros(rows, dtype=record_dtype)
            offs_sub = (np.arange(first_row, first_row + rows) + 0.5) * t_subint
            records["TSUBINT"] = t_subint
            records["OFFS_SUB"] = offs_sub
            records["LST_SUB"] = [
                np.nan
                if d.longitude is None
                else d.calc_lst(mjd + offset / (24.0 * 3600.0), d.longitude)
                for offset in offs_sub
            ]
            records["RA_SUB"] = ra_deg
            records["DEC_SUB"] = dec_deg
            records["GLON_SUB"] = l_deg
            records["GLAT_SUB"] = b_deg
            records["DAT_FREQ"] = chan_freqs
            records["DAT_WTS"] = 1
            records["DAT_SCL"] = 1

            samples = min(rows * n_per_subint, nsamps - first_row * n_per_subint)
            if data_reader is not None and samples:
                data = data_reader(nstart + first_row * n_per_subint, samples)
                if data.shape != (samples, nifs, nchans):
                    raise ValueError(
                        "data_reader returned data with shape "
                        f"{data.shape}, expected {(samples, nifs, nchans)}"
                    )
                full_rows, tail = divmod(samples, n_per_subint)
                if full_rows:
                    records["DATA"][:full_rows] = data[
                        : full_rows * n_per_subint
                    ].reshape(full_rows, n_per_subint, nifs, nchans)
                if tail:
                    records["DATA"][full_rows, :tail] = data[full_rows * n_per_subint :]
            records.tofile(fits_file)
        padding = (-n_subints * record_dtype.itemsize) % 2880
        if padding:
            fits_file.write(b"\0" * padding)
    logging.info(f"Header information written in {outfile}")
