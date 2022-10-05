import sys
import os
import subprocess
import tarfile
import sysconfig


def absolute_path(component):
    """
    Returns the absolute path to a file in the addon directory.

    Alternative to `os.abspath` that works the same on macOS and Windows.
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), component)

def install_and_import_requirements(requirements_txt=None):
    """
    Installs all modules in the 'requirements.txt' file.
    """
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "tqdm"])
    
    import requests
    from tqdm import tqdm

    environ_copy = dict(os.environ)
    environ_copy["PYTHONNOUSERSITE"] = "1"

    python_devel_tgz_path = absolute_path('python-devel.tgz')

    print("Downloading python include")
    url = f"https://www.python.org/ftp/python/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.tgz"
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(python_devel_tgz_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("ERROR, something went wrong during download")
    
    print("Installing python include")
    python_devel_tgz = tarfile.open(python_devel_tgz_path)
    python_include_dir = sysconfig.get_paths()['include'] # absolute_path("venv/Include/") # 
    def members(tf):
        prefix = f"Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Include/"
        l = len(prefix)
        for member in tf.getmembers():
            if member.path.startswith(prefix):
                member.path = member.path[l:]
                yield member
    python_devel_tgz.extractall(path=python_include_dir, members=members(python_devel_tgz))

    print("Installing dependencies")
    requirements_path = requirements_txt
    if requirements_path is None:
        if sys.platform == 'darwin': # Use MPS dependencies list on macOS
            requirements_path = 'requirements-mac.txt'
        else: # Use CUDA dependencies by default on Linux/Windows.
            # These are not the submodule dependencies from the `development` branch, but use the `main` branch deps for PyTorch 1.11.0.
            requirements_path = 'requirements-win.txt'
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", absolute_path(requirements_path), "--no-cache-dir"], check=True, env=environ_copy, cwd=absolute_path("stable_diffusion/"))

install_and_import_requirements()