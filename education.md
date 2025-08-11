# Education
Moses Trundle-Strachan


```python
# For testing purposes
from matplotlib.patches import Rectangle
from pandas.testing import assert_series_equal

import pandas as pd
import seaborn as sns

sns.set_theme()
```

The [National Center for Education Statistics](https://nces.ed.gov/) is a U.S. federal government agency for collecting and analyzing data related to education. We have downloaded and cleaned one of their datasets *[Percentage of persons 25 to 29 years old with selected levels of educational attainment, by race/ethnicity and sex: Selected years, 1920 through 2018](https://nces.ed.gov/programs/digest/d18/tables/dt18_104.20.asp)* into the `nces-ed-attainment.csv` file.

The `nces_ed_attainment.csv` file has the columns `Year`, `Sex`, `Min degree`, and percentages for each subdivision in the specified year, sex, and min degree. The data is represented as a `pandas` `DataFrame` with the following `MultiIndex`:

- `Year` is the first level of the `MultiIndex` with values ranging from 1920 to 2018.
- `Sex` is the second level of the `MultiIndex` with values `F` for female, `M` for male, or `A` for all students.
- `Min degree` is the third level of the `MultiIndex` with values referring to the minimum degree of educational attainment: `high school`, `associate's`, `bachelor's`, or `master's`.

and columns:
- `Total` is the overall percentage of the given `Sex` population in the `Year` with at least the `Min degree` of educational attainment.
- `White`, `Black`, `Hispanic`, `Asian`, `Pacific Islander`, `American Indian/Alaska Native`, and `Two or more races` is the percentage of students of the specified racial category (and of the `Sex` in the `Year`) with at least the `Min degree` of educational attainment.

Missing data is denoted by `NaN` (not a number).


```python
data = pd.read_csv(
    "nces_ed_attainment.csv",
    na_values=["---"],
    index_col=["Year", "Sex", "Min degree"]
).sort_index(level="Year", sort_remaining=False)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>Total</th>
      <th>White</th>
      <th>Black</th>
      <th>Hispanic</th>
      <th>Asian</th>
      <th>Pacific Islander</th>
      <th>American Indian/Alaska Native</th>
      <th>Two or more races</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Sex</th>
      <th>Min degree</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1920</th>
      <th rowspan="2" valign="top">A</th>
      <th>high school</th>
      <td>NaN</td>
      <td>22.0</td>
      <td>6.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bachelor's</th>
      <td>NaN</td>
      <td>4.5</td>
      <td>1.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1940</th>
      <th rowspan="2" valign="top">A</th>
      <th>high school</th>
      <td>38.1</td>
      <td>41.2</td>
      <td>12.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bachelor's</th>
      <td>5.9</td>
      <td>6.4</td>
      <td>1.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1950</th>
      <th>A</th>
      <th>high school</th>
      <td>52.8</td>
      <td>56.3</td>
      <td>23.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2018</th>
      <th>M</th>
      <th>master's</th>
      <td>7.3</td>
      <td>7.7</td>
      <td>2.8</td>
      <td>3.1</td>
      <td>28.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">F</th>
      <th>high school</th>
      <td>94.0</td>
      <td>96.3</td>
      <td>93.2</td>
      <td>87.2</td>
      <td>97.4</td>
      <td>91.8</td>
      <td>95.1</td>
      <td>93.8</td>
    </tr>
    <tr>
      <th>associate's</th>
      <td>51.5</td>
      <td>59.6</td>
      <td>35.8</td>
      <td>34.2</td>
      <td>76.9</td>
      <td>23.6</td>
      <td>32.8</td>
      <td>48.2</td>
    </tr>
    <tr>
      <th>bachelor's</th>
      <td>40.8</td>
      <td>48.4</td>
      <td>26.2</td>
      <td>23.2</td>
      <td>71.5</td>
      <td>13.5</td>
      <td>22.5</td>
      <td>28.7</td>
    </tr>
    <tr>
      <th>master's</th>
      <td>10.7</td>
      <td>12.6</td>
      <td>6.2</td>
      <td>3.8</td>
      <td>29.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 8 columns</p>
</div>



The cell above reads `nces_ed_attainment.csv` and replaces all occurrences of the `str` `---` with `pandas` `NaN` to help with later data processing steps. By defining a `MultiIndex` on the columns `Year`, `Sex`, and `Min degree`, we can answer questions like "What is the overall percentage of those who have at least a high school degree in the year 2018?" with the following `df.loc[index, columns]` expression.


```python
data.loc[(1940, "A", "bachelor's"), "Total"]
```




    5.9




```python
def compare_bachelors_year(data, year):
    """
    Given a dataframe with multiIndex ("Year", "Sex", "Min degree") and a year, returns a
    a two-row Series that indicates the percentages of persons with listed sex "M" of "F"
    who achieved at least a bachelor's degrees in the year given. 
    """

    values = data.loc[(year, ["M", "F"], "bachelor's"), "Total"]
    return values

output = compare_bachelors_year(data, 1980)
display(output)
assert_series_equal(output, pd.Series([24., 21.], name="Total",
    index=pd.MultiIndex.from_product([[1980], ["M", "F"], ["bachelor's"]], names=data.index.names)
))

```


    Year  Sex  Min degree
    1980  M    bachelor's    24.0
          F    bachelor's    21.0
    Name: Total, dtype: float64



```python
def mean_min_degrees(data, start_year=None, end_year=None, category="Total"):
    """
    Takes a dataframe of educational attainment data, a start_year (default None), 
    an end_year (default None), a string category (default "Total") and returns a 
    Series indicating, for each of the Min degrees within the given years, the average 
    percentage of educational attainment for people of the given category between 
    the start_year and the end_year for the sex "A".
    """
    # for degree in min degrees
        #average percentage of education attainment 
        #for people of the given category\
        #between start_year and end_year for all sexes
        #when start or end year is none slice to end or beginning

    all_years = data.loc[(slice(start_year,end_year), "A", slice(None)), category]
    grouped = all_years.groupby(level = "Min degree").mean()

    return grouped


output = mean_min_degrees(data, start_year=2000, end_year=2009)
display(output)
assert_series_equal(output, pd.Series([38.366667, 29.55, 87.35, 6.466667], name="Total",
    index=pd.Index(["associate's", "bachelor's", "high school", "master's"], name="Min degree")
))
```


    Min degree
    associate's    38.366667
    bachelor's     29.550000
    high school    87.350000
    master's        6.466667
    Name: Total, dtype: float64



```python
mean_min_degrees(data, category="Pacific Islander")
```




    Min degree
    associate's    29.838462
    bachelor's     19.853846
    high school    93.450000
    master's             NaN
    Name: Pacific Islander, dtype: float64




```python
def line_plot_min_degree(data, min_degree):
    """
    Takes a dataframe of educational attainment data and a minimum degree, returns a
    line plot comparing the total percentages of sex "A" that attained at minimum 
    a bachelors degree through each year in the dataset.
    """

    all_years = data.loc[(slice(None), "A", min_degree), ["Total"]]
    all_years["Percentage"] = all_years["Total"]

    plot = sns.relplot(all_years, x="Year", y="Percentage", kind='line')
    plot.ax.set_title("Min degree for all " + min_degree)
    return plot


ax = line_plot_min_degree(data, "bachelor's").facet_axis(0, 0)
assert [tuple(xy) for xy in ax.get_lines()[0].get_xydata()] == [
    (1940,  5.9), (1950,  7.7), (1960, 11.0), (1970, 16.4), (1980, 22.5), (1990, 23.2),
    (1995, 24.7), (2000, 29.1), (2005, 28.8), (2006, 28.4), (2007, 29.6), (2008, 30.8),
    (2009, 30.6), (2010, 31.7), (2011, 32.2), (2012, 33.5), (2013, 33.6), (2014, 34.0),
    (2015, 35.6), (2016, 36.1), (2017, 35.7), (2018, 37.0),
], "data does not match expected"
assert all(line.get_xydata().size == 0 for line in ax.get_lines()[1:]), "plot has more than 1 line"
assert ax.get_title() == "Min degree for all bachelor's", "title does not match expected"
assert ax.get_xlabel() == "Year", "x-label does not match expected"
assert ax.get_ylabel() == "Percentage", "y-label does not match expected"
```


    
![png](output_9_0.png)
    



```python
def bar_plot_high_school_compare_sex(data, year):
    """
    Takes a dataframe of educational attainment data and a year, returns a bar plot
    comparing the total percentages of Sex (A, M, F) that attained at minimum a 
    high school degree in the given year. 
    """
    df = data.loc[(year, slice(None), "high school"), ["Total"]]
    df["Percentage"] = df["Total"]

    plot = sns.catplot(df, x="Sex", y="Percentage", kind="bar")
    plot.set_axis_labels("Sex", "Percentage")
    plot.ax.set_title("High school completion in " + str(year))

    return plot

ax = bar_plot_high_school_compare_sex(data, 2009).facet_axis(0, 0)
assert sorted(rectangle.get_height() for rectangle in ax.findobj(Rectangle)[:3]) == [
    87.5, 88.6, 89.8,
], "data does not match expected"
assert len(ax.findobj(Rectangle)) == 4, "too many rectangles drawn" # ignore background Rectangle
assert ax.get_title() == "High school completion in 2009", "title does not match expected"
assert ax.get_xlabel() == "Sex", "x-label does not match expected"
assert ax.get_ylabel() == "Percentage", "y-label does not match expected"
```


    
![png](output_10_0.png)
    


I prefer the plot from Data Visualization section 1.6: Problems of honesty and good judgment. My reasoning for this is pretty simple, it just feels easier to read. The vertical bars look nice, but they shift my gaze to the top of the graph and I find myself constantly looking arounf to see the different axis labels, its exhuasting. Having the bars horizontal however lets my gaze go up and down in a consistent pattern, way easier to digest!

<img alt="Scatter plot for High school completion in 2009 (for comparison)" src="data:image/webp;base64,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" />


```python
def plot_race_compare_min_degree(data, category):
    """
    """
    df = data.loc[(slice(None), "A", slice(None)), [category]]
    df["Percentage"] = df[category]

    plot = sns.relplot(df, x="Year", y="Percentage", hue="Min degree", kind="line")
    plot.set_axis_labels("Year", "Percentage")
    plot.ax.set_title("Min degree for " + category)
    return plot

ax = plot_race_compare_min_degree(data, "Hispanic").facet_axis(0, 0)
assert sorted([tuple(xy) for xy in line.get_xydata()] for line in ax.get_lines()[:4]) == [
    [(1980,  7.7), (1990,  8.1), (1995,  8.9), (2000,  9.7), (2005, 11.2), (2006,  9.5),
     (2007, 11.6), (2008, 12.4), (2009, 12.2), (2010, 13.5), (2011, 12.8), (2012, 14.8),
     (2013, 15.7), (2014, 15.1), (2015, 16.4), (2016, 18.7), (2017, 18.5), (2018, 20.7)],
    [(1980, 58.0), (1990, 58.2), (1995, 57.1), (2000, 62.8), (2005, 63.3), (2006, 63.2),
     (2007, 65.0), (2008, 68.3), (2009, 68.9), (2010, 69.4), (2011, 71.5), (2012, 75.0),
     (2013, 75.8), (2014, 74.7), (2015, 77.1), (2016, 80.6), (2017, 82.7), (2018, 85.2)],
    [                            (1995,  1.6), (2000,  2.1), (2005,  2.1), (2006,  1.5),
     (2007,  1.5), (2008,  2.0), (2009,  1.9), (2010,  2.5), (2011,  2.7), (2012,  2.7),
     (2013,  3.0), (2014,  2.9), (2015,  3.2), (2016,  4.1), (2017,  3.9), (2018,  3.4)],
    [                            (1995, 13.0), (2000, 15.4), (2005, 17.3), (2006, 16.1),
     (2007, 18.1), (2008, 18.7), (2009, 18.4), (2010, 20.5), (2011, 20.6), (2012, 22.7),
     (2013, 23.1), (2014, 23.4), (2015, 25.7), (2016, 27.0), (2017, 27.7), (2018, 30.5)],
], "data does not match expected"
assert all(line.get_xydata().size == 0 for line in ax.get_lines()[4:]), "plot has more than 4 lines"
assert ax.get_title() == "Min degree for Hispanic", "title does not match expected"
assert ax.get_xlabel() == "Year", "x-label does not match expected"
assert ax.get_ylabel() == "Percentage", "y-label does not match expected"
```


    
![png](output_13_0.png)
    



```python
def line_plot_compare_race(data, min_degree):
    """
    Takes a dataframe of educational attainment data and a category of Race and 
    plots a line chart comparing the percentage of educational attainment over the years
    for a given racial category (sex 'A') across four minimum degree options.
    """
    
    df = data.loc[(slice(2009, None), "A", min_degree)]
    df = df.droplevel(["Sex", "Min degree"])
    
    df = df.loc[:, df.columns.difference(["Total", "Two or more races"])]
    
    plot = sns.relplot(data=df, kind="line")
    plot.set_axis_labels("Year", "Percentage")
    plot.ax.set_title(f"Attainment by race for all {min_degree}")
    return plot


ax = line_plot_compare_race(data, "associate's").facet_axis(0, 0)
assert sorted([tuple(xy) for xy in line.get_xydata()] for line in ax.get_lines()[:6]) == [
    [(2009, 18.4), (2010, 20.5), (2011, 20.6), (2012, 22.7), (2013, 23.1),
     (2014, 23.4), (2015, 25.7), (2016, 27.0), (2017, 27.7), (2018, 30.5)],
    [(2009, 20.8), (2010, 28.9), (2011, 25.0), (2012, 23.6), (2013, 26.3),
     (2014, 18.2), (2015, 22.3), (2016, 16.5), (2017, 27.1), (2018, 24.4)],
    [(2009, 20.9), (2010, 22.0), (2011, 39.7), (2012, 32.4), (2013, 37.3),
                   (2015, 24.9), (2016, 28.6), (2017, 35.8), (2018, 22.6)],
    [(2009, 27.8), (2010, 29.4), (2011, 29.8), (2012, 31.6), (2013, 29.5),
     (2014, 32.0), (2015, 31.1), (2016, 31.7), (2017, 32.7), (2018, 32.6)],
    [(2009, 47.1), (2010, 48.9), (2011, 50.1), (2012, 49.9), (2013, 51.0),
     (2014, 51.9), (2015, 54.0), (2016, 54.3), (2017, 53.5), (2018, 53.6)],
    [(2009, 66.7), (2010, 63.4), (2011, 64.6), (2012, 68.3), (2013, 67.2),
     (2014, 70.3), (2015, 71.7), (2016, 71.5), (2017, 69.9), (2018, 75.5)],
], "data does not match expected"
assert all(line.get_xydata().size == 0 for line in ax.get_lines()[6:]), "plot has more than 6 lines"
assert ax.get_title() == "Attainment by race for all associate's", "title does not match expected"
assert ax.get_xlabel() == "Year", "x-label does not match expected"
assert ax.get_ylabel() == "Percentage", "y-label does not match expected"
```


    
![png](output_14_0.png)
    



```python
line_plot_compare_race(data, "associate's").set(title="Asian associate's attainment reaches new heights")
```




    <seaborn.axisgrid.FacetGrid at 0x7d4a499d9250>




    
![png](output_15_1.png)
    


To learn more about the design of the data dashboard, read the write-up by [Darkhorse Analytics](https://www.darkhorseanalytics.com/blog/lumina-foundations-stronger-nation-gets-a-powerful-new-data-experience) and compare it the write-up by the previous team, [Periscopic](https://periscopic.com/#!/impacts/americas-educational-attainment).
