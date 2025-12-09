from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag, sum as spark_sum
from pyspark.sql.window import Window

# Инициализация Spark
spark = SparkSession.builder \
    .appName("COVID19_DataFrame_Analysis") \
    .getOrCreate()

# Загрузка датасета
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("covid-data.csv") \
    .withColumn("date", to_date(col("date")))

# =====================================================
# Задание 1
# 15 стран с наибольшим процентом переболевших на 31.03.2021
# =====================================================
top_15_countries = (
    df.filter(col("date") == "2021-03-31")
      .select(
          col("iso_code"),
          col("location").alias("country"),
          col("total_cases"),
          col("population")
      )
      .withColumn(
          "recovered_percent",
          col("total_cases") / col("population") * 100
      )
      .orderBy(col("recovered_percent").desc())
      .select("iso_code", "country", "recovered_percent")
      .limit(15)
)

# =====================================================
# Задание 2
# Топ-10 стран по количеству новых случаев
# за последнюю неделю марта 2021
# =====================================================
top_10_countries = (
    df.filter((col("date") >= "2021-03-24") & (col("date") <= "2021-03-31"))
      .groupBy("location")
      .agg(spark_sum("new_cases").alias("new_cases_week"))
      .orderBy(col("new_cases_week").desc())
      .select(
          col("location").alias("country"),
          col("new_cases_week")
      )
      .limit(10)
)

# =====================================================
# Задание 3
# Изменение количества новых случаев в России
# относительно предыдущего дня
# =====================================================
window = Window.orderBy("date")

russia_daily_delta = (
    df.filter(
        (col("location") == "Russia") &
        (col("date") >= "2021-03-24") &
        (col("date") <= "2021-03-31")
    )
    .select("date", col("new_cases"))
    .withColumn("yesterday_cases", lag("new_cases").over(window))
    .withColumn("delta", col("new_cases") - col("yesterday_cases"))
    .select(
        col("date"),
        col("yesterday_cases"),
        col("new_cases"),
        col("delta")
    )
)

spark.stop()
