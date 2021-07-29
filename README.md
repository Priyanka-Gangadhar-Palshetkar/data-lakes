# Project: Data Lakes

### Goal:
To build an ETL pipeline for a data lake hosted on S3

### Project Context: 
A music streaming startup, **Sparkify**, has grown their user base and song database even more and want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

### Process:
- Load song_data and log_data from S3
- Process the data into analytics tables by building an ETL pipeline using Spark
- Write output datasets back into S3 as parquet files

### Input Datasets:

#### Song Dataset
The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song.

The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.
- song_data/A/B/C/TRABCEI128F424C983.json
- song_data/A/A/B/TRAABJL12903CDCF1A.json

#### Log Dataset
The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

The log files in the dataset you'll be working with are partitioned by year and month. For example, here are filepaths to two files in this dataset.
- log_data/2018/11/2018-11-12-events.json
- log_data/2018/11/2018-11-13-events.json


### Output Datasets:

Using the song and log datasets, you'll need to create a star schema optimized for queries on song play analysis. This includes the following tables.

#### Fact Table

- songplays - records in log data associated with song plays i.e. records with page NextSong
columns:- songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent


#### Dimension Tables

- users - users in the app
columns:- user_id, first_name, last_name, gender, level

- songs - songs in music database
columns:- song_id, title, artist_id, year, duration

- artists - artists in music database
columns:- artist_id, name, location, lattitude, longitude

- time - timestamps of records in songplays broken down into specific units
columns:- start_time, hour, day, week, month, year, weekday


### Transformation Logic

1. Input dataset **song_data** provides us with columns that make up the **songs** and **artists** table.
2. Input dataset **log_data** provides us with columns that make up the **time** and **users** table.
3. The **songplays** table is then built by joining the **songs** and **log_data** on song title column.

### Project Structure

- etl.py - The ETL to reads data from S3, processes that data using Spark, and writes them to a new S3
- dl.cfg - Configuration file that contains info about AWS credentials
- test.ipynb - Logic test file

### How to run

Execute the ETL pipeline script by running the command **python etl.py**

#### References

- https://sparkbyexamples.com/pyspark/pyspark-read-and-write-parquet-file/