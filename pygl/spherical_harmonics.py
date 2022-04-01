import json
import os
import zipfile
import urllib.request
import shutil

import numpy as np

PRIOR_URL = "https://gravis.dmi.unibas.ch/data/bip2017.zip"

def default():
    BASEL_MEAN = np.array([1.1011728048324585, 1.4209811687469482, 1.4512102603912354, 0.093048095703125, 0.33028173446655273, 0.31052374839782715, -0.18665151298046112, -0.5508308410644531, -0.5709071159362793, 0.014245382510125637, 0.035528603941202164, 0.04360370710492134, 0.004437612369656563, 0.028474748134613037, 0.03126871585845947, 0.04162950441241264, -0.1991603970527649, -0.11923013627529144, 0.16667290031909943, 0.3205763101577759, 0.3221690058708191, 0.021271375939249992, 0.006222990341484547, 0.006456124596297741, -0.03698839619755745, -0.17653866112232208, -0.15761275589466095], np.float32)
    return BASEL_MEAN.reshape((9, 3))

class BaselSphericalHarmonicsPrior(object):
    """Class containing the Basel illumination prior."""
    def __init__(self, directory=None):
        # If directory is not specified save it in the home directory
        if directory is None:
            directory = os.path.join(os.path.expanduser("~"), "basel_prior")

        self.__maybe_download(directory)

        self.root = directory
        with open(os.path.join(self.root, "basel_illumination_prior.json"), "r") as f:
            prior_data = json.load(f)
        self.mean = np.asanyarray(prior_data["mean"], dtype=np.float32)
        self.cov  = np.asanyarray(prior_data["cov"], dtype=np.float32)

        params_dir = os.path.join(self.root, "parameters")
        self._parameter_files = sorted([os.path.join(params_dir, f) for f in os.listdir(params_dir)])


    def __maybe_download(self, directory):
        """Downloads the prior if it is not already downloaded."""
        params_dir = os.path.join(directory, "parameters")

        if os.path.exists(params_dir):
            return

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Download the prior data
        if not os.path.exists(os.path.join(directory, "bip2017.zip")):
            print("Downloading Basel illumination prior...", end="", flush=True)
            urllib.request.urlretrieve(PRIOR_URL, os.path.join(directory, "bip2017.zip"))
            print("Done.")

        # Extract the prior data
        print("Extracting Basel illumination prior...", end="", flush=True)
        with zipfile.ZipFile(os.path.join(directory, "bip2017.zip"), "r") as zip_ref:
            zip_ref.extractall(directory)
        print("Done!")

        shutil.copytree(os.path.join(directory, "data", "parameters"),
                        os.path.join(directory, "parameters"))
        shutil.copytree(os.path.join(directory, "data", "sphere"),
                        os.path.join(directory, "sphere"))
        
        # Cleanup
        os.remove(os.path.join(directory, "bip2017.zip"))   
        os.remove(os.path.join(directory, "README.txt"))
        os.remove(os.path.join(directory, "BIP2017.jar"))
        shutil.rmtree(os.path.join(directory, "data"))

        print("Computing illumination distribution...", end="", flush=True)
        all_coeffs = []
        for parameter_file in os.listdir(params_dir):
            with open(os.path.join(params_dir, parameter_file), "r") as f:
                data = json.load(f)

            all_coeffs.append(np.asanyarray(data['environmentMap']['coefficients'], dtype=np.float32))
        all_coeffs = np.stack(all_coeffs, axis=0)
        all_coeffs = all_coeffs.reshape((-1, 9*3))

        basel_mean = np.mean(all_coeffs, axis=0)
        basel_cov = np.cov(all_coeffs.T)
        # Save the distribution
        with open(os.path.join(directory, "basel_illumination_prior.json"), "w") as f:
            json.dump({
                "mean": basel_mean.tolist(),
                "cov": basel_cov.tolist()
            }, f)
        print("Done!")

    def draw(self):
        """Draws a sample from the prior."""
        sample = np.random.multivariate_normal(self.mean, self.cov)
        sample = sample.reshape((9, 3)).astype(np.float32)
        return sample
    
    def __len__(self):
        return len(self._parameter_files)

    def __getitem__(self, i):
        with open(self._parameter_files[i], "r") as f:
            data = json.load(f)
        sample = np.asanyarray(data['environmentMap']['coefficients'], dtype=np.float32)
        sample = sample.reshape((9, 3)).astype(np.float32)
        return sample

    @property
    def probe(self):
        return os.path.join(self.root, "sphere", "sphere.ply")