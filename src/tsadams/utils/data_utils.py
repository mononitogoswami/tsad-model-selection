import matplotlib.pyplot as plt
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm

def plot_nasa(telemetry, labels):
    """Convenience function to plot the nasa data

    Parameters
    ----------
    """
    # Plot the data 
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 6))
    ax[0].plot(telemetry, label='Telemetry', c='darkblue')
    ax[0].legend(loc='upper right')
    ax[1].plot(labels, label='Anomaly label', c='red')
    ax[1].legend(loc='upper right')
    plt.show()


def download_file(filename:str, directory: str, source_url: str, decompress: bool = False) -> None:
    """Download data from source_ulr inside directory.
    Parameters
    ----------
    filename: str
        Name of file
    directory: str, Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    decompress: bool
        Wheter decompress downloaded file. Default False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filepath = Path(f'{directory}/{filename}')

    # Streaming, so we can iterate over the response.
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(source_url, stream=True, headers=headers)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
            f.flush()
    t.close()

    size = filepath.stat().st_size

    if decompress:
        if '.zip' in filepath.suffix:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            from patoolib import extract_archive
            extract_archive(str(filepath), outdir=directory)
