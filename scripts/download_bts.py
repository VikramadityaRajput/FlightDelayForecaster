import argparse #parse Command line arguments
from pathlib import Path #manipulate path and directories
import zipfile #allows u to work with zip files
import requests #allows HTTP requests for website or APi interaction
#HTTP Notes
#Hypertext Transfer protocol is a system for transfering files in the form of web pages from servers to clients across the internet.
#Transmission Control Protocol (TCP) manages the channels between your browser and the server
# implements communication between a client adn a server
# you command your browser to open a TCP to the server that responsds to that URL
# your browser is the client and the url ix the address that belong to that server.
#once the conection is established client sends HTTP get request to th server to retrieve the web page it should display. after server sends the response it closes  the TCP connection

URL_Template = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
RAW_DIR = Path("data/raw")

def download_month(year: int, month: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f"bts_{year}_{month}.csv"

    if csv_path.exists():
        return csv_path
    url = URL_Template.format(year=year, month=month) #inputs the year and month into the template to get the URL
    zip_path = RAW_DIR / f"bts_{year}_{month}.zip" #creates the path that the zip file will go into
    #downloads the zip from url into the zip path
    print("Downloading...")
    with requests.get(url, stream = True, timeout = 60) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size = 1024 * 1024):
                if chunk:
                    f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
        extracted = Path(zf.extract(csv_name, path=RAW_DIR))
        extracted.rename(csv_path)

    zip_path.unlink()  # delete the .zip now that we have the .csv
    return csv_path 

def main() -> None:
    parser = argparse.ArgumentParser(description="Download monthly BTS flight data.")
    parser.add_argument(
        "--months", nargs = "+", required = True, help = "Months in YYYY-M format"
    )
    args = parser.parse_args()
    for ym in args.months:
        y, m = ym.split("-")
        download_month(int(y), int(m))

if __name__ == "__main__":
    main()