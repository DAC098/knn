use std::collections::HashMap;
use std::convert::Infallible;
use std::fs::OpenOptions;
use std::io::{BufReader, ErrorKind};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, bail, Error};
use clap::{Parser, ValueEnum};
use csv::{ReaderBuilder, StringRecord};

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    if args.columns.is_empty() {
        bail!("no columns specified to pull numeric data from");
    }

    let result = OpenOptions::new().read(true).open(&args.file);

    let file = match result {
        Ok(f) => f,
        Err(err) => match err.kind() {
            ErrorKind::NotFound => bail!("the requested csv file was not found"),
            _ => return Err(Error::new(err).context("failed to load csv file")),
        },
    };

    // store a reference to the distance algorithm
    let algo = match args.algo {
        AlgoType::Euclidean => euclidean_distance,
        AlgoType::Manhattan => manhattan_distance,
    };

    let mut reader = ReaderBuilder::new()
        .has_headers(!args.no_header)
        .from_reader(BufReader::new(file));

    // retrieve the label and datapoint columns from the csv reader
    let (label, columns) = get_columns_and_label(&mut reader, &args.label, &args.columns)?;
    // parse the provided datapoint to estimate. will expect a similar amount
    // of numbers as the provided number of columns
    let datapoint = parse_datapoint(&args.datapoint, columns.len())?;
    // map the csv records iterator into a list of knn records to use later
    let iter = reader.records()
        .enumerate()
        .map(|(index, maybe)| match maybe {
            Ok(record) => map_record(label, &columns, index, record),
            Err(err) => Err(anyhow::Error::new(err).context(format!("failed to parse csv record. row: {index}")))
        });

    // collect all the records since we are offering the ability to run k over
    // a range vs a single iteration
    let records: Vec<KnnRecord> = {
        let mut rtn = Vec::new();

        for maybe in iter {
            rtn.push(maybe?);
        }

        rtn
    };

    // k will be the min of the specified high value or the total number of
    // records
    for k in args.k.get_range(records.len()) {
        // collect the datapoints with the calculated euclidean distance from
        // the provided datapoint
        let mut collected: Vec<(f64, &KnnRecord)> = Vec::new();

        for record in &records {
            collected.push((algo(&datapoint, &record.data), record));
        }

        // sort the collected records by the euclidean distance. since floats
        // dont directly implement the std::cmp::Ord trait we will sort by
        // f64::total_cmp
        collected.sort_by(|(a, _), (b, _)| a.total_cmp(b));

        // collect the label groups and count how many are encountered
        let mut groups = HashMap::with_capacity(k);

        for index in 0..k {
            groups.entry(collected[index].1.label.as_str())
                // increment if the group was previously added
                .and_modify(|counter| *counter += 1)
                // insert if not already existing
                .or_insert(1);
        }

        println!("k value: {k}");

        for (key, count) in groups {
            // print the calculated percentage for each group found
            println!("  {key}: {count} {:.2}", (count as f64) / (k as f64));
        }
    }

    Ok(())
}

/// a simple k nearest neighbors (knn) calculator that loads a csv file
/// containing records to use for estimating a given datapoint.
#[derive(Debug, Parser)]
struct CliArgs {
    /// the number of neighbors to lookup
    #[arg(short, default_value = "3")]
    k: KValue,

    /// specifies the algorithm to use when calculating distances
    #[arg(long, default_value = "euclidean")]
    algo: AlgoType,

    /// the list of columns to use as datapoints
    #[arg(short, long = "col")]
    columns: Vec<ColumnType>,

    /// indicates that the csv contains no header row
    #[arg(long)]
    no_header: bool,

    /// the column to use as the label
    #[arg(long)]
    label: ColumnType,

    /// a comma delimitered list of numbers to estimate its group for
    #[arg(long)]
    datapoint: String,

    /// path to the csv file to load
    #[arg(short, long)]
    file: PathBuf,
}

/// represents the k value to use for calculations
#[derive(Debug, Clone)]
pub struct KValue((usize, usize, usize));

impl KValue {
    fn parse_range(given: &str) -> Result<Option<(usize, usize)>, &'static str> {
        if let Some((low,high)) = given.split_once('-') {
            let Ok(low) = usize::from_str(low) else {
                return Err("failed to parse low value for k range");
            };

            let Ok(high) = usize::from_str(high) else {
                return Err("failed to parse high value for k range");
            };

            if low == 0 {
                return Err("low value for k range cannot be 0");
            }

            if low > high {
                return Err("low value for k range cannot be greater than the high value");
            }

            // add one to high so that we can treat it as inclusive
            Ok(Some((low, high + 1)))
        } else {
            Ok(None)
        }
    }

    fn get_range(&self, total: usize) -> std::iter::StepBy<std::ops::Range<usize>> {
        // figure out if the minimum value is either the k or the number of records
        // collected
        let len = std::cmp::min(total, self.0.1);

        ((self.0.0)..(len)).step_by(self.0.2)
    }
}

impl FromStr for KValue {
    type Err = &'static str;

    fn from_str(given: &str) -> Result<Self, Self::Err> {
        if let Some((range, step)) = given.split_once(',') {
            let Ok(step) = usize::from_str(step) else {
                return Err("failed to parse step size for k value");
            };

            if step == 0 {
                return Err("step size must be larger than 0");
            }

            if let Some((low, high)) = Self::parse_range(range)? {
                Ok(Self((low, high, step)))
            } else {
                Err("you must specify a range when using a k range")
            }
        } else if let Some((low,high)) = Self::parse_range(given)? {
            Ok(Self((low, high, 1)))
        } else if let Ok(value) = usize::from_str(given) {
            if value == 0 {
                Err("k value cannot be 0")
            } else {
                Ok(Self((value, value + 1, 1)))
            }
        } else {
            Err("invalid k value specified")
        }
    }
}

/// represents the algorithm to use when calculating distances
#[derive(Debug, Clone, ValueEnum)]
pub enum AlgoType {
    Euclidean,
    Manhattan,
}

/// represents the column type specified in the command line arguments
#[derive(Debug, Clone)]
pub enum ColumnType {
    /// a column name to attempt to lookup
    Name(String),

    /// a defined zero based index number in the csv
    Index(usize),
}

impl FromStr for ColumnType {
    type Err = Infallible;

    fn from_str(given: &str) -> Result<Self, Self::Err> {
        if let Ok(index) = usize::from_str(given) {
            Ok(Self::Index(index))
        } else {
            Ok(Self::Name(given.into()))
        }
    }
}

/// represents the data collected from the csv for the knn
#[derive(Debug)]
pub struct KnnRecord {
    data: Vec<f64>,
    label: String,
}

/// attempts to retrieve the desired data columns and label from the csv file
fn get_columns_and_label<R>(
    reader: &mut csv::Reader<R>,
    label: &ColumnType,
    retrieve: &[ColumnType],
) -> anyhow::Result<(usize, Vec<usize>)>
where
    R: std::io::Read,
{
    let mut columns = Vec::with_capacity(retrieve.len());

    let found = if reader.has_headers() {
        let headers = reader
            .headers()
            .context("failed to retrieve csv headers")?
            .into_iter()
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect::<HashMap<&str, usize>>();

        for to_get in retrieve {
            match to_get {
                ColumnType::Name(name) => {
                    let Some(index) = headers.get(name.as_str()) else {
                        bail!("unknown column header specified. column: {name}");
                    };

                    columns.push(*index);
                }
                ColumnType::Index(index) => {
                    if *index >= headers.len() {
                        bail!("index is out of range for known headers. column index: {index}");
                    }

                    columns.push(*index);
                }
            }
        }

        match label {
            ColumnType::Name(name) => {
                let Some(index) = headers.get(name.as_str()) else {
                    bail!("unknown label column header specified. column: {name}");
                };

                *index
            }
            ColumnType::Index(index) => {
                if *index >= headers.len() {
                    bail!("label index is out of range for known headers. column index: {index}");
                }

                *index
            }
        }
    } else {
        for to_get in retrieve {
            match to_get {
                ColumnType::Name(name) => {
                    bail!(
                        "no headers were specified in the csv but given a named column. column: {name}"
                    )
                }
                ColumnType::Index(index) => columns.push(*index),
            }
        }

        match label {
            ColumnType::Name(name) => {
                bail!(
                    "no headers were specified in the csv but given a named label column. column: {name}"
                );
            }
            ColumnType::Index(index) => *index,
        }
    };

    Ok((found, columns))
}

/// maps at csv record into a [`KnnRecord`] with the expected columns and label
fn map_record(label: usize, columns: &[usize], index: usize, record: StringRecord) -> anyhow::Result<KnnRecord> {
    let mut rtn = Vec::with_capacity(columns.len());

    for col in columns {
        if let Some(value) = record.get(*col) {
            let Ok(v) = f64::from_str(&value) else {
                bail!("failed to parse column data. row: {} column index: {}", index + 1, col + 1);
            };

            rtn.push(v);
        } else {
            bail!("column data not found. column index: {}", col + 1);
        }
    }

    let Some(found) = record.get(label) else {
        bail!("failed to find label. label index: {index}");
    };

    Ok(KnnRecord {
        data: rtn,
        label: found.to_owned(),
    })
}

/// attempts to parse the requested datapoint
fn parse_datapoint(given: &str, num_cols: usize) -> anyhow::Result<Vec<f64>> {
    let mut rtn = Vec::with_capacity(num_cols);

    for result in given.split(',').map(f64::from_str) {
        rtn.push(result.context("failed to parse datapoint")?);
    }

    if rtn.len() != num_cols {
        bail!("number of datapoints does not match number of columns");
    }

    Ok(rtn)
}

/// calculates the euclidean distance between 2 sets of datapoints
fn euclidean_distance(a_data: &[f64], b_data: &[f64]) -> f64 {
    // we will expect the total datapoints from a and b to be the same and just
    // zip them together for the iterator chain
    a_data.iter()
        .zip(b_data)
        .map(|(a, b)| (a - b).powf(2.0))
        .sum::<f64>()
        .sqrt()
}

/// calculates the manhattan distance between 2 sets of datapoints
fn manhattan_distance(a_data: &[f64], b_data: &[f64]) -> f64 {
    // similar to the euclidean distance expectation
    a_data.iter()
        .zip(b_data)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
}
