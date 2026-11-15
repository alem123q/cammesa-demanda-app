"""Download the external variables used to extend the CAMMESA demand models.

Run it from the repository root on a machine with internet access:

    python scripts/download_external.py

It writes to data/external/:
  - weather_daily_cities.csv      daily temperature per city (raw, from Open-Meteo)
  - weather_monthly_regions.csv   monthly weather per CAMMESA region (average of its cities)
  - emae_monthly.csv              INDEC monthly economic activity estimator (EMAE)
  - prices_monthly_regions.csv    INDEC CPI by region: housing & utilities division vs. general level
                                  (proxy for the real electricity tariff)
  - price_series_candidates.csv   every CPI series the search found, to check the matching
  - sources.json                  URLs, series IDs and download date

Needs pandas. If you get "CERTIFICATE_VERIFY_FAILED", install truststore (uses the
operating system's certificates) or certifi:  pip install -U truststore certifi
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "external"
START = "2016-01-01"  # one year before the demand data, so 12-month lags exist from January 2017

# Representative cities for each CAMMESA region (keys match the "region" column of the CAMMESA file).
# The regional value is the simple average of its cities.
CITIES = {
    "GRAN BS.AS.": {"Buenos Aires": (-34.61, -58.38)},
    "BUENOS AIRES": {"La Plata": (-34.92, -57.95), "Mar del Plata": (-38.00, -57.56), "Bahia Blanca": (-38.72, -62.27)},
    "LITORAL": {"Rosario": (-32.95, -60.65), "Santa Fe": (-31.63, -60.70), "Parana": (-31.73, -60.52)},
    "CENTRO": {"Cordoba": (-31.42, -64.18), "Rio Cuarto": (-33.12, -64.35), "San Luis": (-33.30, -66.34)},
    "CUYO": {"Mendoza": (-32.89, -68.83), "San Juan": (-31.54, -68.52)},
    "NOROESTE": {"Tucuman": (-26.82, -65.22), "Salta": (-24.78, -65.41), "Santiago del Estero": (-27.78, -64.26)},
    "NORESTE": {"Resistencia": (-27.45, -58.99), "Corrientes": (-27.47, -58.83), "Posadas": (-27.37, -55.90)},
    "COMAHUE": {"Neuquen": (-38.95, -68.06), "Santa Rosa": (-36.62, -64.29), "Bariloche": (-41.13, -71.31)},
    "PATAGONICA": {"Comodoro Rivadavia": (-45.86, -67.48), "Trelew": (-43.25, -65.31), "Rio Gallegos": (-51.62, -69.22)},
}
BASE_TEMP = 18.0  # base temperature (°C) for heating and cooling degree-days

OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
SERIES_API = "https://apis.datos.gob.ar/series/api/series/"
SEARCH_API = "https://apis.datos.gob.ar/series/api/search/"
EMAE_ID = "143.3_NO_PR_2004_A_21"  # EMAE, original series (base 2004 = 100); checked at run time


def _ssl_context() -> ssl.SSLContext:
    """Prefer the OS certificate store (truststore), then certifi, then Python's default.

    Some Python installs (e.g. an old Miniconda on Windows) ship an outdated certificate
    bundle and fail with "certificate has expired" even though the site is fine.
    """
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_context()


def get(url: str, params: dict, retries: int = 4) -> bytes:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "cammesa-demand-portfolio"})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                return r.read()
        except Exception as e:  # network hiccups or rate limits
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise RuntimeError(
                    f"SSL certificate check failed for {url}.\n"
                    "Fix: pip install -U truststore certifi   (and check that the PC's date and time are right)"
                ) from e
            if attempt == retries - 1:
                raise RuntimeError(f"Request failed: {full}\n{e}") from e
            time.sleep(5 * (attempt + 1))


# ----------------------------------------------------------------- weather
def last_complete_month_end() -> date:
    return date.today().replace(day=1) - timedelta(days=1)


def download_city(name: str, lat: float, lon: float, end: str) -> pd.DataFrame:
    params = dict(latitude=lat, longitude=lon, start_date=START, end_date=end,
                  daily="temperature_2m_mean,temperature_2m_max,temperature_2m_min",
                  timezone="America/Argentina/Buenos_Aires")
    d = json.loads(get(OPEN_METEO, params))["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "city": name,
                         "temp_mean": d["temperature_2m_mean"], "temp_max": d["temperature_2m_max"],
                         "temp_min": d["temperature_2m_min"]})


def weather_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily city temperatures -> monthly values per region."""
    d = daily.copy()
    d["hdd"] = (BASE_TEMP - d["temp_mean"]).clip(lower=0)  # heating degree-days
    d["cdd"] = (d["temp_mean"] - BASE_TEMP).clip(lower=0)  # cooling degree-days
    d["month"] = d["date"].dt.strftime("%Y-%m")
    city = d.groupby(["region", "city", "month"]).agg(
        temp_mean=("temp_mean", "mean"), temp_max=("temp_max", "mean"), temp_min=("temp_min", "mean"),
        hdd=("hdd", "sum"), cdd=("cdd", "sum"), days=("temp_mean", "count")).reset_index()
    region = city.groupby(["region", "month"]).agg(
        temp_mean=("temp_mean", "mean"), temp_max=("temp_max", "mean"), temp_min=("temp_min", "mean"),
        hdd=("hdd", "mean"), cdd=("cdd", "mean"), days=("days", "min")).reset_index()
    return region.rename(columns={"month": "indice_tiempo"}).round(2)


def download_weather() -> dict:
    end = last_complete_month_end().isoformat()
    frames = []
    for region, cities in CITIES.items():
        for name, (lat, lon) in cities.items():
            print(f"  weather  {region:<13} {name}")
            frames.append(download_city(name, lat, lon, end).assign(region=region))
            time.sleep(1)  # be polite with the free API
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(OUT / "weather_daily_cities.csv", index=False)
    monthly = weather_monthly(daily)
    monthly.to_csv(OUT / "weather_monthly_regions.csv", index=False)
    print(f"  -> {len(monthly)} region-months, {monthly['indice_tiempo'].min()} to {monthly['indice_tiempo'].max()}")
    return {"url": OPEN_METEO, "start": START, "end": end, "base_temp_c": BASE_TEMP,
            "cities": {r: {c: list(ll) for c, ll in cs.items()} for r, cs in CITIES.items()}}


# ----------------------------------------------------------------- EMAE
def find_emae_id() -> str:
    """Return a valid EMAE series ID: the default one if it works, otherwise search the API."""
    try:
        get(SERIES_API, dict(ids=EMAE_ID, limit=1, format="json"))
        return EMAE_ID
    except RuntimeError:
        print("  default EMAE ID failed; searching the series API...")
    res = json.loads(get(SEARCH_API, dict(q="emae original", limit=20)))["data"]
    for r in res:
        f = r["field"]
        print(f"    candidate {f['id']}: {f.get('description') or f.get('title')}")
    for r in res:
        text = (r["field"].get("description", "") + r["field"].get("title", "")).lower()
        if "original" in text and "emae" in (text + r["dataset"].get("title", "").lower()):
            return r["field"]["id"]
    raise RuntimeError("No EMAE series found; pass one with --emae-id (see the candidates above).")


def download_emae(series_id: str | None) -> dict:
    sid = series_id or find_emae_id()
    raw = get(SERIES_API, dict(ids=sid, format="csv", limit=5000, start_date=START))
    OUT.joinpath("_emae_raw.csv").write_bytes(raw)
    df = pd.read_csv(OUT / "_emae_raw.csv")
    OUT.joinpath("_emae_raw.csv").unlink()
    df = df.rename(columns={df.columns[0]: "indice_tiempo", df.columns[1]: "emae"})
    df["indice_tiempo"] = pd.to_datetime(df["indice_tiempo"]).dt.strftime("%Y-%m")
    df["emae_yoy_pct"] = (df["emae"] / df["emae"].shift(12) - 1).mul(100).round(2)
    df.to_csv(OUT / "emae_monthly.csv", index=False)
    print(f"  -> EMAE {sid}: {len(df)} months, {df['indice_tiempo'].min()} to {df['indice_tiempo'].max()}")
    return {"url": SERIES_API, "series_id": sid, "start": START}


# ----------------------------------------------------------------- CPI (tariff proxy)
# CAMMESA region -> INDEC CPI region
CPI_REGION = {"GRAN BS.AS.": "GBA", "BUENOS AIRES": "Pampeana", "LITORAL": "Pampeana", "CENTRO": "Pampeana",
              "CUYO": "Cuyo", "NOROESTE": "Noroeste", "NORESTE": "Noreste", "COMAHUE": "Patagonia",
              "PATAGONICA": "Patagonia"}
CPI_KEYS = {"GBA": ["gba", "gran buenos aires"], "Pampeana": ["pampeana"], "Cuyo": ["cuyo"],
            "Noroeste": ["noroeste", "noa"], "Noreste": ["noreste", "nea"], "Patagonia": ["patagonia"]}


def _search(q: str, limit: int = 100) -> list[dict]:
    res = json.loads(get(SEARCH_API, dict(q=q, limit=limit)))["data"]
    return [{"id": r["field"]["id"], "title": r["field"].get("title", ""), "description": r["field"].get("description", ""),
             "frequency": r["field"].get("frequency", ""), "start": r["field"].get("time_index_start", ""),
             "end": r["field"].get("time_index_end", ""), "dataset": r["dataset"].get("title", "")} for r in res]


def _pick(cands: pd.DataFrame, region_keys: list[str], must: list[str]) -> str | None:
    """First monthly CPI (base Dec 2016) series whose text mentions the region and all the ``must`` words."""
    for _, c in cands.iterrows():
        text = f"{c['id']} {c['title']} {c['description']} {c['dataset']}".lower().replace("_", " ")
        if (c["frequency"] in ("", "R/P1M") and "2016" in text and any(k in text for k in region_keys)
                and all(m in text for m in must)):
            return c["id"]
    return None


def download_prices() -> dict:
    found = (pd.DataFrame(_search("ipc vivienda agua electricidad gas combustibles region")
                          + _search("ipc nivel general region") + _search("indice precios consumidor nivel general"))
             .drop_duplicates("id"))
    found.to_csv(OUT / "price_series_candidates.csv", index=False)
    ids = {}
    for reg, keys in CPI_KEYS.items():
        ids[(reg, "housing_utilities")] = _pick(found, keys, ["vivienda"])
        ids[(reg, "general")] = _pick(found, keys, ["general"])
    missing = [f"{r}/{k}" for (r, k), v in ids.items() if v is None]
    if missing:
        print("  could not match:", ", ".join(missing), "-> see price_series_candidates.csv and paste it to me")
    frames = []
    for (reg, kind), sid in ids.items():
        if sid is None:
            continue
        print(f"  CPI {reg:<9} {kind:<17} {sid}")
        raw = get(SERIES_API, dict(ids=sid, format="csv", limit=5000, start_date=START))
        OUT.joinpath("_cpi.csv").write_bytes(raw)
        d = pd.read_csv(OUT / "_cpi.csv")
        frames.append(pd.DataFrame({"cpi_region": reg, "kind": kind, "indice_tiempo": pd.to_datetime(d.iloc[:, 0]).dt.strftime("%Y-%m"),
                                    "value": d.iloc[:, 1]}))
        time.sleep(0.5)
    OUT.joinpath("_cpi.csv").unlink(missing_ok=True)
    if not frames:
        raise RuntimeError("No CPI series matched; paste data/external/price_series_candidates.csv to me.")
    cpi = pd.concat(frames).pivot_table(index=["cpi_region", "indice_tiempo"], columns="kind", values="value").reset_index()
    if {"housing_utilities", "general"} <= set(cpi.columns):
        cpi["relative_price"] = (cpi["housing_utilities"] / cpi["general"]).round(4)   # utilities price in real terms
    out = pd.DataFrame([{"region": r, "cpi_region": c} for r, c in CPI_REGION.items()]).merge(cpi, on="cpi_region")
    out.to_csv(OUT / "prices_monthly_regions.csv", index=False)
    print(f"  -> {len(out)} region-months, {out['indice_tiempo'].min()} to {out['indice_tiempo'].max()}")
    return {"url": SERIES_API, "series_ids": {f"{r}/{k}": v for (r, k), v in ids.items()}, "cpi_region_map": CPI_REGION}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["weather", "emae", "prices"], help="download just one source")
    ap.add_argument("--emae-id", help="override the EMAE series ID")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "sources.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if a.only in (None, "weather"):
        print("Weather (Open-Meteo historical API)")
        meta["weather"] = download_weather()
    if a.only in (None, "emae"):
        print("EMAE (datos.gob.ar time-series API)")
        meta["emae"] = download_emae(a.emae_id)
    if a.only in (None, "prices"):
        print("CPI by region (datos.gob.ar time-series API)")
        meta["prices"] = download_prices()
    meta["downloaded"] = date.today().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
