# Rust KNN

This is a small implementation of a K nearest neighbors algorithm written in
Rust. It provides the ability to read in a specified CSV file and then perform
classification predictions on the specified labels and columns.

## Commands

The application provides the ability to directly predict a single datapoint
using the `predict` command. It also provides the ability to search for an
optimal set of parameters using the `search` command.

You are able to specify `k` values as a single value, a range of values, and a
range of values with a specified step.

```
-k 3 # single value
-k 3-6 # range of values
-k 2-8,2 # range of values with step
```

You can also choose the distance function to use which is `euclidean` or
`manhattan` currently.

```
--algo euclidean
--algo manhattan
```

When specifying a column you can give either the name of the column if the CSV
has a header or by specifying the index (0 based index). When used for the
`search` command this will tell the application which columns to train against.
Note: if the column name happens to be a value that can be parsed as an index
then it will prioritize that over a column name.

```
-c width
--column 3
```

When running the `predict` command, the arugment for supplying a datapoint to
estimate its label for can be specified as a comma delimited list of numbers
that are in the same order as the columns specified.

```
-c width
-c height
--datapoint 5.6,7.3 # width,height
```

When running the `search` command you can specify how much to split the data
between training and testing by specifying a percentage value between 0 and 1.

```
--test 0.375 # 37.5% to use for testing and 62.5% to use for training
```

Some example commands of how to run the application.

This will try to find an optimal `k` value between `3-6` using `euclidean`
distance function for the label `species` with the specified columns
`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` and `body_mass_g` for the
file `penguins.csv`.

```bash
$ knn -f penguins.csv \
    search \
    -k 3-6 \
    --algo euclidean \
    -c bill_length_mm -c bill_depth_mm -c flipper_length_mm -c body_mass_g \
    --label species
```

This is a similar set of parameters but for predicting a single datapoint.

```bash
$ knn -f ./penguins.csv \
    predict \
    -k 3-6 \
    --algo euclidean \
    -c bill_length_mm -c bill_depth_mm -c flipper_length_mm -c body_mass_g \
    --label species \
    --datapoint 34.8,18.7,200,4000
```

## Requirements

The Rust version used during development is `1.86.0`. Other versions of Rust
may work but would need to be formally tested.

Currently all columns that are needed for calculation must be parsable as a 64
bit floating point value. The label column can be any value as it is purely used
as a key for classification. The CSV must be encoded as UTF-8 strings as all
logic is built around parsing valid UTF-8.

## Code

The application uses some libraries to assist with parsing commands and csv
files. All other code is custom made for this application using only the
standard library. The code includes documentation and comments through out to
illustrate or provide context to what is happening.
