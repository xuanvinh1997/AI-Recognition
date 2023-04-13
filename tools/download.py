import requests

def download_large_file(url, file_path, chunk_size=8192):
    """
    Download a large file from the given URL and save it to the specified file path.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)