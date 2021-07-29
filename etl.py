import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """Get or Create Spark Session"""
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Load data from song_data dataset from S3, extract columns for songs and artist tables and write the data into parquet files on S3.
    
    Parameters
    ----------
    spark: session
          This is the spark session that has been created
    input_data: path
           This is the path to the input data s3 bucket.
    output_data: path
            This is the path where the parquet files will be written.
    """
    
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    
    # read song data file
    df = spark.read.json(song_data).dropDuplicates()

    # extract columns to create songs table
    # columns - song_id, title, artist_id, year, duration, artist_name
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration", "artist_name").distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").mode("overwrite").parquet(os.path.join(output_data + 'songs/songs.parquet'))

    # extract columns to create artists table
    # columns - artist_id, name, location, latitude, longitude
    artists_table = df.selectExpr("artist_id", "artist_name as name", "artist_location as location", \
                                  "artist_latitude as latitude", "artist_longitude as longitude").dropDuplicates(["artist_id"])
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(os.path.join(output_data + 'artists/artists.parquet'))


def process_log_data(spark, input_data, output_data):
    """
    Load data from log_data dataset from S3, extract columns for users and time tables and write the data into parquet files on S3.
    Use songs table and log_data to extract columns for songplays table and write the output into parquet files on S3
    
    Parameters
    ----------
    spark: session
          This is the spark session that has been created
    input_data: path
           This is the path to the input data s3 bucket.
    output_data: path
            This is the path where the parquet files will be written.
    """
    
    # get filepath to log data file
    log_data = input_data + 'log_data/*.json'

    # read log data file
    df = spark.read.json(log_data).dropDuplicates()
    
    # filter by actions for song plays
    df = df.filter(col('page') == 'NextSong')

    # extract columns for users table 
    # columns - user_id, first_name, last_name, gender, level
    users_table = df.selectExpr("userId as user_id", "firstName as first_name", "lastName as last_name", \
                               "gender", "level").dropDuplicates(["user_id"])
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(os.path.join(output_data + 'users/users.parquet'))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(int(x)/1000),TimestampType())
    df = df.withColumn('timestamp', get_timestamp(col("ts")))
    
    # create datetime column from original timestamp column
    # get_datetime = udf(lambda x: x.cast(DateType()))
    df = df.withColumn('datetime', col("timestamp").cast('date'))
    
    # extract columns to create time table
    # columns - start_time, hour, day, week, month, year, weekday
    time_table = df.withColumn('start_time', df.timestamp) \
             .withColumn('day', dayofmonth(df.datetime)) \
             .withColumn('month', month(df.datetime)) \
             .withColumn('year', year(df.datetime)) \
             .withColumn('hour', hour(df.timestamp)) \
             .withColumn('week', weekofyear(df.datetime)) \
             .withColumn('weekday', dayofweek(df.datetime)) \
             .select('start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday')
                    
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").mode("overwrite").parquet(os.path.join(output_data + 'time/time.parquet'))

    # read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data + 'songs/songs.parquet'))

    # extract columns from joined song and log datasets to create songplays table 
    # columns - songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent, datetime, year, month
    songplays_table = df.join(song_df, ((df.song == song_df.title) & (df.artist == song_df.artist_name))) \
                      .selectExpr("timestamp as start_time", "userId as user_id", "level", "song_id", "artist_id", 
                                  "sessionId as session_id", "location", "userAgent as user_agent", "datetime") \
                      .withColumn("songplay_id", monotonically_increasing_id()) \
                      .withColumn("month", month(df.datetime)) \
                      .withColumn("year", year(df.datetime))
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").mode("overwrite").parquet(os.path.join(output_data + 'songplays/songplays.parquet'))


def main():
    """
    Create a spark session.
    Define input and output locations.
    Call process_song_data function to build songs and artists table.
    Call process_log_data function to build users, time and songplays table.
    """
   
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://priyankapalshetkar/udacity_output/"

    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
