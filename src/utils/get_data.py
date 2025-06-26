import requests

def download_fuelwatch_data():
    """
    https://www.fuelwatch.wa.gov.au/retail/historic
    """
    for year in range(2001, 2026):
        for month in range(1, 13):
            url = f"https://warsydprdstafuelwatch.blob.core.windows.net/historical-reports/FuelWatchRetail-{month:02d}-{year}.csv"
            response = requests.get(url, timeout=40)
            
            if response.status_code == 200:
                with open(f"../../data/raw/retail/{year}-{month:02d}-FuelWatchRetail.csv", "wb") as file:
                    file.write(response.content)
                print(f"Data downloaded successfully {year}-{month}.")
            else:
                print(f"Failed to download data. Status code: {response.status_code}")

def download_terminal_gate_prices():
    """
    https://www.fuelwatch.wa.gov.au/industry/historic-terminal-gate-prices
    """
    for year in range(2001, 2026):
        url = f"https://warsydprdstafuelwatch.blob.core.windows.net/historical-reports/FuelWatchWholesale-{year}.csv"
        response = requests.get(url, timeout=40)
        if response.status_code == 200:
            with open(f"../../data/raw/tgp/FuelWatchWholesale-{year}.csv", "wb") as file:
                file.write(response.content)
            print(f"Data downloaded successfully {year}.")
        else:
            print(f"Failed to download data. Status code: {response.status_code}")


    

if __name__ == "__main__":
    download_fuelwatch_data()
    download_terminal_gate_prices()
