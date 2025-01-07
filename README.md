# reflective-error

<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT HEADER -->
<br />
<div align="center">
  <a href="https://github.com/robert-edwin-rouse/reflective-error">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">project_title</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/robert-edwin-rouse/reflective-error"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/robert-edwin-rouse/reflective-error/issues">Report Bug</a>
    ·
    <a href="https://github.com/robert-edwin-rouse/reflective-error/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com|width=100)

Reflective Error: A Metric for Assessing Predictive Performance at Extremes

This repo contains the files necessary to reproduce the machine learning model for the paper entitled 'Reflective Error: A Metric for Assessing Predictive Performance at Extremes'; instructions, in terms of prerequisites and accessing the required data, are included below.  There are two notebooks: the first, Part I, goes through the fictitious experiments and highlights the motivation in a more straight forward context; the second, Part II, introduces the application of Reflective Error as a loss function for neural networks whilst also highlighting the error metric's usage in a real world setting.  Part I requires negligible storage whilst Part II requires no more than 2GB of data; this can be reduced by taking a subset of the ERA5 data stipulated in the notebook (we took data from across the UK as part of a broader hydrological research project).  The neural model used was trained on a 24GB M2 Macbook Air (8‑core CPU, 10‑core GPU, 16‑core Neural Engine) in less than a minute, so the computational demand is also low.

In order to download ERA5 data, an ECMWF CDS account will be required along with a CDS API key.  Note that the CDS servers were upgraded in the summer of 2024, so users of the system prior to the upgrade will require a new account.  Instructions on account creation and downloading data can be found [here](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

The following libraries/tools/etc. are required for this project.

* python3
  * pandas
  * numpy
  * scipy
  * matplotlib
* cdsapi
* xarray
* pytorch
* scikit-learn

We recommend using conda for package management, for which instructions on downloading and installing it can can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Installation

1. This repository can be installed from the command line via:
   ```sh
   git clone https://github.com/robert-edwin-rouse/reflective-error.git
   ```
2. The most up to date version of apollo can be installed via:
   ```sh
   git clone https://github.com/robert-edwin-rouse/apollo.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Usage

As discussed in the brief, the journey through the paper is documented and can be followed through the two Jupyter notebooks.  The first notebook covers the fictitious experiments using synthetic, simplified datasets, whilst the second covers the real world application to a hydrological problem.  Alternatively, provided the catchment database csv file has been populated with data for the target catchments, the initial combined meteorology and streamflow files can be produced by running the ```assembly.py``` script.  The artificial neural network can then be trained and the results on the test set, as partitioned in the paper, produced by running ```reflective_streamflow.py```.  We have also provided the results we obtained from a grid search through alpha and beta parameters for the reflective loss function and a script that generates the corresponding plots in ```reflective_streamflow.py```.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Roadmap
- [ ] Add script to create additional plots.

See the [open issues](https://github.com/robert-edwin-rouse/reflective-error/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contributing

Contributions are welcome.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contact

Project Link: [https://github.com/robert-edwin-rouse/reflective-error](https://github.com/robert-edwin-rouse/reflective-error)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Acknowledgments

* The reviewers who recommended substantial changes to the codebase in order to improve its accessibility and interpretability.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/robert-edwin-rouse/reflective-error.svg?style=for-the-badge
[contributors-url]: https://github.com/robert-edwin-rouse/reflective-error/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/robert-edwin-rouse/reflective-error.svg?style=for-the-badge
[forks-url]: https://github.com/robert-edwin-rouse/reflective-error/network/members
[stars-shield]: https://img.shields.io/github/stars/robert-edwin-rouse/reflective-error.svg?style=for-the-badge
[stars-url]: https://github.com/robert-edwin-rouse/reflective-error/stargazers
[issues-shield]: https://img.shields.io/github/issues/robert-edwin-rouse/reflective-error.svg?style=for-the-badge
[issues-url]: https://github.com/robert-edwin-rouse/reflective-error/issues
[license-shield]: https://img.shields.io/github/license/robert-edwin-rouse/reflective-error.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[product-screenshot]: figures/Normal.png