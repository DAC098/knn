use anyhow::bail;
use clap::Args;

use crate::classify::classify_datapoint_owned;
use crate::cli::{AlgoType, ColumnType, Datapoint, KValue};
use crate::csv::{Reader, collect_records, get_columns_and_label};
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
        let iter = records
            .iter()
            .map(|record| (&record.data, record.label.as_str()));

        let (min, groups) = classify_datapoint_owned(k, iter, algo, &datapoint);

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
