# Swiss Hiking Routes
Data Analytics Project

# Setup local environment

## Prerequisites
Install the following software on your local machine:
- Python
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [MySQL Workbench](https://dev.mysql.com/downloads/workbench/)

# Run application
1. Clone the repository to your local machine.
2. Start Docker Desktop & ensure it's running.
3. Open a terminal and navigate to the project directory.
4. Execute the following command to build and start the Docker container: `docker-compose up`

# Connect to MySQL Database (requires Docker container running)
1. Open MySQL Workbench.
2. Create a new connection with the following details:
   - Connection Name `DA_Swiss_Hiking_Routes`
   - Connection Method: Standard (TCP/IP)
   - Hostname: `127.0.0.1`
   - Port: `3306`
   - Username: `root`
   - password: `password`
3. Test the connection and save it.
4. Connect to the database.

# Stop application
1. Open a terminal and navigate to the project directory.
2. Execute the following command to stop and remove the Docker container: `docker-compose down`


# Project Overview

## (1) Data collection using Web Scraping and/or a Web API.
## Data Collection
Data was collected through web scraping from the official Swiss hiking trails website (schweizer-wanderwege.ch), focusing on hiking routes in the Berner Oberland region. The scraping process captured comprehensive information about each hiking trail directly from the website.

## (2) Data preparation (e.g. remove missing values and duplicates, create new variables, enrich the data with open data).
## Data Structure & Preparation
The scraped dataset includes the following core information for each hiking trail:
- **URL**: Direct link to the detailed trail information page
- **Title**: Name of the hiking route
- **Location**: Geographic location and starting/ending points of the trail
- **Canton**: Swiss canton where the trail is located (extracted from location data)
- **Difficulty Level**: Trail difficulty classification (T1-T6 scale)
- **Duration**: Time required to complete the hike (converted to minutes)
- **Distance**: Trail length in kilometers
- **Ascent/Descent**: Elevation gain and loss in meters
- **Physical Demand**: Required fitness level 

## Data Enrichment
To enhance the analytical capabilities of the dataset, additional geographic information was integrated:
- **Latitude & Longitude**: Precise geographic coordinates obtained through the OpenStreetMap API (Nominatim service)

The geographic coordinates were added by querying the OpenStreetMap Nominatim service for each unique hiking location. This open data integration enables spatial analysis and visualization of hiking trails on interactive maps, allowing for better understanding of trail distribution and accessibility across the Berner Oberland region.


