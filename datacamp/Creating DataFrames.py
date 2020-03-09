# Creating a DataFrame
import pandas as pd

#Creating a dataframe from list of dictionaries
# Create a list of dictionaries with new data
avocados_list = [
    {"date": "2019-11-03", "small_sold": 10376832, "large_sold": 7835071},
    {"date": "2019-11-10", "small_sold": 10717154, "large_sold": 8561348},
]

# Convert list into DataFrame
avocados_2019 = pd.DataFrame(avocados_list)
# Print the new DataFrame
print(avocados_2019)

# Create a dictionary of lists with new data
avocados_dict = {
  "date": ["2019-11-17", "2019-12-01"],
  "small_sold": [10859987, 9291631],
  "large_sold": [7674135, 6238096]
}

# Convert dictionary into DataFrame
avocados_2019_dict = pd.DataFrame(avocados_dict)

# Print the new DataFrame
print(avocados_2019_dict)

#CSV to DataFrame
airline_bumping = pd.read_csv("airline_bumping.csv")
print(airline_bumping.head())

# For each airline, select nb_bumped and total_passengers and sum
airline_totals = airline_bumping.groupby("airline")[["nb_bumped", "total_passengers"]].agg(sum)

# Create new col, bumps_per_10k: no. of bumps per 10k passengers for each airline
airline_totals["bumps_per_10k"] = airline_totals["nb_bumped"] / airline_totals["total_passengers"] * 10000

# Print airline_totals
print(airline_totals)

# Create airline_totals_sorted
airline_totals_sorted = airline_totals.sort_values(by = ["bumps_per_10k"], ascending = False)

# Print airline_totals_sorted
print(airline_totals_sorted)

# Save results
airline_totals_sorted.to_csv("airline_totals_sorted.csv")
avocados_2019.to_csv("avocados_2019.csv")
avocados_2019_dict.to_csv("avocados_2019_dict.csv")

