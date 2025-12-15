use std::collections::HashMap;
use std::str::FromStr;

use anyhow::bail;
use clap::Args;

use crate::cli::{AlgoType, ColumnType, Datapoint, KValue};
use crate::csv::{Reader, StringRecord, get_columns_and_label};
use crate::distance;

#[derive(Debug, Args)]
pub struct PredictArgs {
    /// the number of neighbors to lookup
    #[arg(short, default_value = "3")]
    k: KValue,

    /// specifies the algorithm to use when calculating distances
    #[arg(long, default_value = "euclidean")]
    algo: AlgoType,

    /// the list of columns to use as datapoints
    #[arg(short, long = "col")]
    columns: Vec<ColumnType>,

    /// the column to use as the label
    #[arg(long)]
    label: ColumnType,

    /// a comma delimitered list of numbers to estimate its group for
    #[arg(long)]
    datapoint: Datapoint,
}

pub fn knn_predict<R>(mut reader: Reader<R>, arg: PredictArgs) -> anyhow::Result<()>
where
    R: std::io::Read,
{
    if arg.columns.is_empty() {
        bail!("no columns specified to pull numeric data from");
    }

    // store a reference to the distance algorithm
    let algo = match arg.algo {
        AlgoType::Euclidean => distance::euclidean,
        AlgoType::Manhattan => distance::manhattan,
    };

    // retrieve the label and datapoint columns from the csv reader
    let (label, columns) = get_columns_and_label(&mut reader, &arg.label, &arg.columns)?;
    // parse the provided datapoint to estimate. will expect a similar amount of
    // numbers as the provided number of columns
    let datapoint = arg.datapoint.into_inner();

    if datapoint.len() != columns.len() {
        bail!("number of datapoints does not match number of columns");
    }

    let records = collect_records(reader, label, &columns)?;

    // k will be the min of the specified high value or the total number of
    // records
    for k in arg.k.get_range(records.len()) {
        let (min, groups) = classify_datapoint(k, &records, algo, &datapoint);

        print!("k value: {k} |");

        for v in &datapoint {
            print!(" {v}");
        }

        println!();

        for (key, count) in groups {
            // print the calculated percentage for each group found
            println!("  {key}: {count} {:.2}", (count as f64) / (min as f64));
        }
    }

    Ok(())
}

/// represents the data collected from the csv for the knn
#[derive(Debug)]
pub struct KnnRecord {
    data: Vec<f64>,
    label: String,
}

/// maps at csv record into a [`KnnRecord`] with the expected columns and label
fn map_record(
    label: usize,
    columns: &[usize],
    index: usize,
    record: StringRecord,
) -> anyhow::Result<KnnRecord> {
    let mut rtn = Vec::with_capacity(columns.len());

    for col in columns {
        if let Some(value) = record.get(*col) {
            let Ok(v) = f64::from_str(&value) else {
                bail!(
                    "failed to parse column data. row: {} column index: {}",
                    index + 1,
                    col + 1
                );
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

/// determines the percentage classification for a given datapoint
fn classify_datapoint<'a, F>(
    k: usize,
    records: &'a [KnnRecord],
    algo: F,
    datapoint: &[f64],
) -> (usize, HashMap<&'a str, u32>)
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    // collect the datapoints with the calculated distance function from
    // the provided datapoint
    let mut collected: Vec<(f64, &KnnRecord)> = Vec::new();

    for record in records {
        collected.push((algo(&datapoint, &record.data), record));
    }

    // sort the collected records by the distance function. since floats
    // dont directly implement the std::cmp::Ord trait we will sort by
    // f64::total_cmp
    collected.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    // collect the label groups and count how many are encountered
    let mut groups = HashMap::with_capacity(k);

    let min = std::cmp::min(k, collected.len());

    for index in 0..min {
        groups
            .entry(collected[index].1.label.as_str())
            // increment if the group was previously added
            .and_modify(|counter| *counter += 1)
            // insert if not already existing
            .or_insert(1);
    }

    (min, groups)
}

fn collect_records<R>(
    mut reader: Reader<R>,
    label: usize,
    columns: &[usize],
) -> anyhow::Result<Vec<KnnRecord>>
where
    R: std::io::Read,
{
    // map the csv records iterator into a list of knn records to use later
    let iter = reader
        .records()
        .enumerate()
        .map(|(index, maybe)| match maybe {
            Ok(record) => map_record(label, &columns, index, record),
            Err(err) => Err(anyhow::Error::new(err)
                .context(format!("failed to parse csv record. row: {index}"))),
        });

    // collect all the records since we are offering the ability to run k over
    // a range vs a single iteration
    let mut rtn = Vec::new();

    for maybe in iter {
        rtn.push(maybe?);
    }

    Ok(rtn)
}
