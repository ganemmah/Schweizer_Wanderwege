# Schweizer_Wanderwege
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
   - Connection Name `DA_Schweizer_Wanderwege`
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