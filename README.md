# Objective
The broad goal is to develop a model to automate the detection and sizing of transplanted corals 
from the photos Costal Impact takes of their artificial reefs. This is an AI model where:
### Input:
Images taken underwater during Coastal Impact dives. These are images of tiles which have been placed on our artificial reefs. 
Each tile consists of 4 corals in total, each in one corner of the tile.
Format will be JPEG, PNG. Dimensions of the image can change. 
Clarity and other parameters affecting how clearly the corals are visible in the image will change too. 
In all images, there is a white reference block in the middle of the tile.

### Output:
* Polygons as annotations, in a particular format (coco), drawn around corals
* Estimated size of each coral in an Excel sheet for each individual coral


# Why we are doing this?
We have multiple artificial reefs that we have placed around Grande Island. 
Each month, we go to each artificial reef (which consists of multiple tiles, where each tile might have 
4 corals or might be without corals) and click pictures of every tile. The goal is to analyse how corals are 
growing at different sites and under different conditions across our study period. To do this, we need to know the size of each coral (or coral fragment, will be interchangeably used) in each picture that we are
taking. Once we know the sizes, this data of individual corals with their sizes across time will go to the 
Coastal Impact's marine biologists who will combine this data with other information and start their study.

---
# Github Repo Structure
Loosely adapted from [here](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa)