import os
import time
import requests
import pandas as pd

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

DATA_PATH = "data/train.csv"
IMAGE_DIR = "data/images"

ZOOM = 18
IMG_SIZE = 224
STYLE = "satellite-v9"
SLEEP_TIME = 0.2

def build_url(lat, lon):
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/{STYLE}/static/"
        f"{lon},{lat},{ZOOM}/"
        f"{IMG_SIZE}x{IMG_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )


def fetch_images(limit=None):
    if MAPBOX_TOKEN is None:
        raise ValueError("MAPBOX_TOKEN not set")

    os.makedirs(IMAGE_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if limit:
        df = df.head(limit)

    for idx, row in df.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{idx}.png")

        if os.path.exists(img_path):
            continue

        url = build_url(row["lat"], row["long"])

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(response.content)
                print(f"[✓] Saved image {idx}")
            else:
                print(f"[!] Failed {idx} | Status {response.status_code}")

        except Exception as e:
            print(f"[X] Error {idx}: {e}")

        time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    fetch_images(limit=10)
