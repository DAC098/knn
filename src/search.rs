use std::collections::HashMap;
use std::str::FromStr;

use anyhow::bail;
use clap::Args;

use crate::cli::{AlgoType, ColumnType, KValue};
use crate::csv::{Reader, StringRecord, get_columns_and_label};
use crate::distance;

#[derive(Debug, Args)]
pub struct SearchArgs {
    /// the number of neighbors to lookup
    #[arg(short, default_value = "3-10")]
    k: KValue,

    /// specifies the algorithm to use when calculating distances
    #[arg(long, default_value = "euclidean")]
    algo: AlgoType,

    /// the list of columns to use when searching
    #[arg(short, long = "col")]
    columns: Vec<ColumnType>,

    /// the cross validation value to use
    #[arg(long, default_value = "3")]
    cv: usize,

    /// the percent of data to test against
    #[arg(long, default_value = "0.25")]
    test: f64,

    /// the colume to use as the label
    #[arg(long)]
    label: ColumnType,
}

pub fn knn_search<R>(mut reader: Reader<R>, arg: SearchArgs) -> anyhow::Result<()>
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
    let records = collect_records(reader, label, &columns)?;

    let (train, test) = split_dataset(&records, arg.test);

    let selected: Vec<usize> = Vec::new();
    let avail = columns.to_owned();

    while !avail.is_empty() {
        let best = None::<usize>;
    }

    Ok(())
}

/// represents the data collected from the csv for the knn
#[derive(Debug)]
pub struct KnnRecord {
    cols: Vec<usize>,
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
        cols: columns.to_owned(),
        data: rtn,
        label: found.to_owned(),
    })
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

fn split_dataset<'a>(
    records: &'a [KnnRecord],
    split: f64,
) -> (Vec<&'a KnnRecord>, Vec<&'a KnnRecord>) {
    let mut groups: HashMap<&'a str, Vec<&KnnRecord>> = HashMap::new();

    for record in records {
        groups
            .entry(record.label.as_str())
            // increment if the group was previously added
            .and_modify(|list| list.push(record))
            // insert if not already existing
            .or_insert(vec![record]);
    }

    let mut train = Vec::new();
    let mut test = Vec::new();

    for (_, mut records) in groups {
        let amount = (records.len() as f64 * split).floor() as usize;

        train.extend(records.split_off(amount));
        test.extend(records);
    }

    (train, test)
}
