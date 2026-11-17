"""Real flights (ADS-B) turned into radar tracks.

ADS-B aircraft broadcast their own GPS position, so a recorded flight is a trustworthy
"truth" with real, human-flown maneuvers. We keep that truth and simulate only the radar:
the flight is moved so it lies 20-60 km from a radar at the origin, and the usual range and
bearing noise and missed detections are added. The trackers then see real flying instead of
invented maneuvers.

Pipeline:  CSV -> flights -> gap-free 1 Hz segments -> 120-scan windows -> placed near the
radar -> `Track` objects (the same type the simulator makes, so every tracker works on them).

Only the standard library `csv` module is used for reading (no pandas).
"""
import csv
from dataclasses import dataclass

import numpy as np

from .simulate import RadarConfig, TargetConfig, Track, track_from_truth

EARTH_RADIUS = 6_371_000.0   # metres

# Accepted column names (OpenSky exports differ): canonical name -> alternatives, lower case.
ALIASES = {
    "time": ("time", "timestamp", "time_position", "lastposupdate"),
    "icao24": ("icao24", "icao"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "lng", "longitude"),
    "velocity": ("velocity", "groundspeed", "gs", "speed"),
    "heading": ("heading", "track", "true_track", "truetrack"),
    "onground": ("onground", "on_ground"),
}


@dataclass
class Flight:
    """One aircraft's recorded positions (irregular times)."""
    icao24: str
    t: np.ndarray        # (n,) seconds, increasing
    lat: np.ndarray      # (n,) degrees
    lon: np.ndarray      # (n,) degrees
    velocity: np.ndarray  # (n,) m/s, NaN when not reported
    heading: np.ndarray   # (n,) degrees, NaN when not reported


@dataclass
class Segment:
    """A gap-free stretch of a flight, resampled to 1 Hz, in local metres."""
    icao24: str
    t: np.ndarray        # (T,) seconds, spaced exactly 1 s apart
    xy: np.ndarray       # (T, 2) metres east and north of the segment's own origin


def _find_columns(header):
    lower = {h.strip().lower(): h for h in header}
    cols = {}
    for name, options in ALIASES.items():
        cols[name] = next((lower[o] for o in options if o in lower), None)
    missing = [n for n in ("time", "icao24", "lat", "lon") if cols[n] is None]
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}; found {list(header)}. "
                         "Expected time (epoch seconds), icao24, lat, lon and optionally velocity, heading.")
    return cols


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_opensky_csv(path) -> list:
    """Read a CSV of aircraft positions into one `Flight` per aircraft.

    Rows without a time or position are skipped, as are rows marked on the ground (taxiing is
    not flying) and repeated times for the same aircraft.
    """
    per_aircraft = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = _find_columns(reader.fieldnames or [])
        for row in reader:
            t, lat, lon = _float(row[cols["time"]]), _float(row[cols["lat"]]), _float(row[cols["lon"]])
            if not (np.isfinite(t) and np.isfinite(lat) and np.isfinite(lon)):
                continue
            if cols["onground"] and str(row[cols["onground"]]).strip().lower() in ("true", "1", "t"):
                continue
            vel = _float(row[cols["velocity"]]) if cols["velocity"] else float("nan")
            hdg = _float(row[cols["heading"]]) if cols["heading"] else float("nan")
            per_aircraft.setdefault(row[cols["icao24"]].strip().lower(), []).append((t, lat, lon, vel, hdg))

    flights = []
    for icao, rows in per_aircraft.items():
        rows.sort(key=lambda r: r[0])
        arr = np.array(rows)
        arr = arr[np.concatenate([[True], np.diff(arr[:, 0]) > 0])]        # drop repeated times
        flights.append(Flight(icao, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]))
    return flights


def to_local_xy(lat, lon, lat0, lon0) -> np.ndarray:
    """Equirectangular projection: metres east (x) and north (y) of (lat0, lon0).

    Accurate to well under a metre per kilometre over the few tens of kilometres of one
    segment, which is all we need.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS
    y = np.radians(lat - lat0) * EARTH_RADIUS
    return np.stack([x, y], axis=-1)


def split_flight(flight: Flight, max_gap: float = 5.0, max_speed: float = 400.0,
                 min_speed: float = 50.0, min_duration: float = 120.0) -> list:
    """Cut a flight into gap-free, plausible, 1 Hz `Segment`s.

    A cut is made wherever two reports are more than `max_gap` seconds apart, or imply a speed
    above `max_speed` (a position glitch). Stretches shorter than `min_duration` seconds, or
    whose median speed is below `min_speed` (not flying), are dropped. Each stretch is resampled
    to exactly 1 Hz by linear interpolation.
    """
    if len(flight.t) < 2:
        return []
    lat0, lon0 = float(np.mean(flight.lat)), float(np.mean(flight.lon))
    xy = to_local_xy(flight.lat, flight.lon, lat0, lon0)
    dt = np.diff(flight.t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cut = (dt > max_gap) | (step / dt > max_speed)
    edges = np.flatnonzero(cut) + 1
    segments = []
    for lo, hi in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(flight.t)]])):
        t, p = flight.t[lo:hi], xy[lo:hi]
        if len(t) < 2:
            continue
        grid = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1.0)
        if len(grid) < min_duration:
            continue
        resampled = np.stack([np.interp(grid, t, p[:, 0]), np.interp(grid, t, p[:, 1])], axis=1)
        speed = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        if np.median(speed) < min_speed:
            continue
        # re-centre on the segment's own start so coordinates stay small
        segments.append(Segment(flight.icao24, grid, resampled - resampled[0]))
    return segments


def windows(segment: Segment, n_steps: int = 120, stride: int = 60) -> list:
    """Cut a segment into windows of exactly `n_steps` scans (the length the trackers expect)."""
    n = len(segment.t)
    return [Segment(segment.icao24, segment.t[i:i + n_steps], segment.xy[i:i + n_steps])
            for i in range(0, n - n_steps + 1, stride)]


def place_radar(xy, rng, min_range: float = 20_000.0, max_range: float = 60_000.0,
                outer: float = 70_000.0, tries: int = 50):
    """Shift a window so it lies near a radar at the origin, or return None if it cannot.

    The window's centre is put at a random distance in [min_range, max_range] and a random
    bearing. The shift is only accepted if the whole window then stays within `outer` of the
    radar and never closer than 5 km (a real radar loses aircraft overhead), so one flight is
    placed the same way a simulated one would be.
    """
    xy = np.asarray(xy, dtype=float)
    centre = xy.mean(axis=0)
    for _ in range(tries):
        r = rng.uniform(min_range, max_range)
        th = rng.uniform(-np.pi, np.pi)
        placed = xy - centre + r * np.array([np.cos(th), np.sin(th)])
        ranges = np.hypot(placed[:, 0], placed[:, 1])
        if ranges.min() >= 5_000.0 and ranges.max() <= outer:
            return placed
    return None


def truth_from_positions(xy, dt: float = 1.0) -> np.ndarray:
    """(T, 4) [x, y, vx, vy]: velocity by finite differences (central inside, one-sided at the ends)."""
    return np.concatenate([xy, np.gradient(xy, dt, axis=0)], axis=1)


def segments_to_tracks(segments, radar: RadarConfig, rng, n_steps: int = 120, stride: int = 60,
                       max_tracks=None) -> list:
    """Turn segments into `Track`s: real truth, simulated radar measurements.

    Every segment is cut into windows of `n_steps` scans, each window is placed near the radar,
    and range/bearing noise and missed detections are added exactly as the simulator does.
    """
    tracks = []
    for seg in segments:
        for win in windows(seg, n_steps, stride):
            placed = place_radar(win.xy, rng)
            if placed is None:
                continue
            tracks.append(track_from_truth(rng, truth_from_positions(placed, radar.dt), radar))
            if max_tracks is not None and len(tracks) >= max_tracks:
                return tracks
    return tracks


def load_tracks(path, radar: RadarConfig = RadarConfig(), seed: int = 0, n_steps: int = 120,
                stride: int = 60, max_tracks=None, **split_kwargs) -> list:
    """CSV file -> list of `Track`s in one call."""
    segments = [s for f in load_opensky_csv(path) for s in split_flight(f, **split_kwargs)]
    return segments_to_tracks(segments, radar, np.random.default_rng(seed), n_steps, stride, max_tracks)
