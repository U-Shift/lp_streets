# Street_functions

In Jones et al. ([2008](https://www.sciencedirect.com/science/article/pii/S0386111214601965)), streets are understood to serve two fundamental and often competing roles: a **“link”** and a **“place”** function. 

The **link** function refers to a street’s role in facilitating movement through the network, allowing vehicles, cyclists, pedestrians, and public transport to travel efficiently from one location to another. Streets with a strong link function are typically designed to prioritize capacity, speed, and continuity of traffic flow. In sum, a good **link** street reduces the time users spend in it.

In contrast, the **place** function describes the street as a destination in itself—a setting for social, economic, and civic activities such as shopping, dining, meeting, or simply spending time. Streets with a strong place function are designed to support accessibility, comfort, safety, and the quality of the public realm. In this case, streets with a good **place** function encourage users to spend more time in them.

The authors argue that effective street design requires balancing these two roles, since prioritizing movement alone can undermine the street’s social and economic vitality, while emphasizing place without considering movement can reduce network efficiency. However, the assessment of these functions has traditionally been mostly microscopic (at the street-level) or qualitative. This framework attempts to solve this gap, by providing a methodology to calculate the **link** and **place** functions of streets in a systematic way, at the macroscopic level of the street network, and in a quantitative way, by using OpenStreetMap data and open-source Python libraries.


## Development information

Currently, all of the work is being done in [lp_streets.ipynb](lp_streets.ipynb), in the context of the MSc dissertation, since it's easier for iteration purposes. The repository is meant to be updated with ``.py`` files and proper documentation in the future, in order for the methodology to be easily applicable, by only needing to call functions from the library.


## Acknowledgement

This work is being developed at [U-Shift](https://ushift.tecnico.ulisboa.pt/) urban mobility research group, part of the [CERIS](https://ceris.pt/) research center, at [Instituto Superior Técnico](https://tecnico.ulisboa.pt/en/), [University of Lisbon](https://www.ulisboa.pt/en), Portugal. Currently, it is being developed in the context of the MSc dissertation in Transportation Systems, by [Miguel Relvas Pires](https://github.com/miguelrelvaspires).