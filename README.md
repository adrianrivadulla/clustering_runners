# Clustering study

The scripts within this repo can replicate the analysis conducted for 

Rivadulla, A. R., Chen, X., Cazzola, D., Trewartha, G., & Preatoni, E. (2024). Clustering analysis across different speeds reveals two distinct running techniques with no differences in running economy. Sports Biomechanics, 1–24. https://doi.org/10.1080/14763141.2024.2372608

## Getting Started

These instructions will get a copy of the project up and running on your local machine for development and testing purposes.

### Pre-requisites

- Python >= 3.8

### Installing

- Clone this repository and enter it:

```Shell
   git clone https://github.com/adrianrivadulla/clustering_runners.git
```

or download and unzip this repository.

- Set up the environment. Using Anaconda is recommended:

Navigate to the clustering_runners directory and create a version environment with the environment.yml file provided:

 ```Shell
     cd /path/to/clustering_runners
     conda env create -f environment.yml
 ```

### Getting the data

Download the dataset from [waiting for Bath research data link]. Create a `data` directory by unzipping the dataset within the project root. Please refer to the dataset description in the link for further details on how the data and files are structured.


### Usage


- Activate the environment:

```Shell
    conda activate clustering_runners_env
```

- Just run the main.py script:

```Shell
    python main.py
```

Note that the script is interactive since this is how I designed it for myself when I was working on the project.
You will have to select the number of clusters you want for analysis based on the dendrogram and internal validity scores.
Similarly, since clustering was performed at three different speeds and the multispeed condition independently, you will have to
match up the colours via a GUI for the visualisations to make more sense. 

**To replicate the exact results and figures of the analysis, you will need to select 2 as the number of clusters
in every condition, since this was our choice based on the dendrogram and internal validity scores (prioritising Silhouette scores).**


## Limitations

The functions for statistical tests have been developed for the specific case of having 2 clusters and will need to be 
expanded if you want to compare more than 2 clusters.

Creating the SPM figures takes a few minutes. Please be patient. This may be solved with newer versions of the spm1d package itself or matplotlib.

Documentation for the different scripts and functions has been generated using Copilot and it can be brief in some cases. 

The code has not gone through linting or code style checks so it could definitely be improved in that regard.

Newer package managers like uv could be used for faster environment set up.

## License

Copyright (c) 2024 Adrian R Rivadulla

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program (see gpl.txt and lgpl.txt). If not, see <https://www.gnu.org/licenses/>.


# Citation
If you use FootNet or this code base in your work, please cite:

Rivadulla, A. R., Chen, X., Cazzola, D., Trewartha, G., & Preatoni, E. (2024). Clustering analysis across different speeds reveals two distinct running techniques with no differences in running economy. Sports Biomechanics, 1–24. https://doi.org/10.1080/14763141.2024.2372608


# Contact
For questions about our paper or code, please contact [Adrian R](mailto:arr43@bath.ac.uk). Since I currently do not have the time to make this repo adhere to the best practices in terms of code quality, style and documentation, improvements and suggestions are welcome.
