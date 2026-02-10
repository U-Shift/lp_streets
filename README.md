# Introduction

In Jones et al. ([2008](https://www.sciencedirect.com/science/article/pii/S0386111214601965)), streets are understood to serve two fundamental and sometimes competing roles: a **“link”** and a **“place”** function. 

The **link** function refers to a street’s role in facilitating movement through the network, allowing vehicles, cyclists, pedestrians, and public transport to travel efficiently from one location to another. Streets with a strong link function are typically designed to prioritize capacity, speed, and continuity of traffic flow. In sum, a good **link** street reduces the time users spend in it.

In contrast, the **place** function describes the street as a destination in itself—a setting for social, economic, and civic activities such as shopping, dining, meeting, or simply spending time. Streets with a strong place function are designed to support accessibility, comfort, safety, and the quality of the public realm. In this case, streets with a good **place** function encourage users to spend more time in them.

The authors argue that effective street design requires balancing these two roles, since prioritizing movement alone can undermine the street’s social and economic vitality, while emphasizing place without considering movement can reduce network efficiency. However, the assessment of these functions has traditionally been mostly microscopic (at the street-level) or qualitative. This framework attempts to solve this gap, by providing a methodology to calculate the **link** and **place** functions of streets in a systematic way, at the macroscopic level of the street network, and in a quantitative way, by using [OpenStreetMap](https://www.openstreetmap.org) data and open-source Python libraries.


# Development information

Currently, all of the work is being done in [lp_streets.ipynb](lp_streets.ipynb), in the context of the MSc dissertation, since it's easier for iteration purposes. The repository is meant to be updated with ``.py`` files and proper documentation in the future, in order for the methodology to be easily applicable, by only needing to call functions from the library. The following subsections provide a brief overview of the files in the repository, and their purpose:

### [lp_streets.ipynb](lp_streets.ipynb)

This is the main file, where all of the work is being done. It's a Jupyter Notebook, which allows for easy iteration and visualization of the results and outputs. The notebook is structured in a way that allows for a step-by-step execution of the methodology, from data collection and preprocessing, to the calculation of the link and place functions and street classifications, and finally to the visualization of the results in each of these steps. If you intend to use the methodology or explore the code, I recommend starting with this file, since it contains all of the code and explanations for each step of the process. The other files in the repository are meant to either support the work in this notebook (such as the ``legend_colormaps`` folder), or to provide a structure for future developments (such as the ``cities_user_inputs.csv`` file).


### [legend_colormaps](legend_colormaps)

This folder contains the legend colormaps used for the visualizations in the interactive report at the end of the notebook. These colormaps are used to represent the different classifications of streets based on their link and place functions, in a way that is visually intuitive and easy to interpret. You should not need to modify these colormaps, unless you want to change the visual representation of the classifications in the report. If you do want to modify them, be aware that you'll need to modify the code in the notebook accordingly, in order for the visualizations to work properly.


### [cities_user_inputs.csv](cities_user_inputs.csv)

This file corresponds to the structure that may be used in the future for user inputs, in order to apply the methodology to different cities, simultaneously. The current values correspond to examples that may be considered to run the methodology and they may need to be corrected. It's also important to consider that some of these examples correspond to large metropolitan areas in which the calculation of the "edge betweenness centrality" (one of the indicators) is computationally expensive (consider using a ``k`` parameter in that step when running [lp_streets.ipynb](lp_streets.ipynb)). It contains the following columns:
- **city**: the name of the city for which the methodology will be applied (just for reference, not the name that will be used to retrieve the data from OpenStreetMap).
- **study_area_Nominatim**: the name of the area to be retrieved from OpenStreetMap, using the names from the [Nominatim](https://nominatim.openstreetmap.org/) search engine. This column is a string that should correspond exactly to the name of the area in OpenStreetMap, in order for the data retrieval to work properly. For example, for the Lisbon municipality, the string should be "Município de Lisboa, Portugal". The bigger the detail in the name, the better, since it will help to retrieve the correct area.
- **local_CRS**: the local Coordinate Reference System (CRS) to be used for the city, in order to project the data and perform the calculations. This column is a string that should correspond to the EPSG code of the CRS, in the format ``epsg:XXXX``. For example, for Lisbon, the string should be ``epsg:3763``, which corresponds to the ETRS89 / Portugal TM06 CRS, commonly used in Portugal.
- **speed_limits_XXXX**: these columns correspond to the speed limits for each of the OSM ``highway`` levels, where ``XXXX`` should be replaced by the name of the OSM ``highway`` level. For example, for the ``primary`` level, the column should be named ``speed_limits_primary``, and for the ``residential`` level, the column should be named ``speed_limits_residential``. The values in these columns should correspond to the speed limits (in km/h) for each of the OSM ``highway`` levels, which will be used to calculate the link function of the streets. These values can be based on local regulations or standards, or can be estimated based on typical values for each type of street. Naturally, this is a simplification done locally in order to be able to apply the methodology at the macroscopic level, since in reality, speed limits can vary within the same street, and can also be different from the ones defined in the local regulations, due to various factors such as traffic conditions, road design, or enforcement practices. For better results, consider contributing directly in OpenStreetMap with the correct speed limits for each street, so that the data retrieval can be more accurate and the methodology can be applied without needing to define these values locally (you're also helping the OpenStreetMap community by contributing with more accurate data!).


# Acknowledgement

This work is being developed at [U-Shift](https://ushift.tecnico.ulisboa.pt/) urban mobility research group, part of the [CERIS](https://ceris.pt/) research center, at [Instituto Superior Técnico](https://tecnico.ulisboa.pt/en/), [University of Lisbon](https://www.ulisboa.pt/en), Portugal. Currently, it is being developed in the context of the MSc dissertation in Transportation Systems, by [Miguel Relvas Pires](https://github.com/miguelrelvaspires).

![logo_acknowledgement](readme_images/logo_acknowledgement.png)