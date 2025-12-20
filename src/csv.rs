use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{Context, bail};
pub use csv::{Reader, ReaderBuilder, StringRecord};

use crate::cli::ColumnType;

/// represents the data collected from the csv for the knn
#[derive(Debug)]
pub struct KnnRecord {
    pub data: Vec<f64>,
    pub label: String,
}

/// attempts to retrieve the desired data columns and label from the csv file
pub fn get_columns_and_label<R>(
    reader: &mut csv::Reader<R>,
    label: &ColumnType,
    retrieve: &[ColumnType],
) -> anyhow::Result<(usize, Vec<usize>)>
where
    R: std::io::Read,
{
    let mut columns = Vec::with_capacity(retrieve.len());

    let found = if reader.has_headers() {
        let known_headers = reader.headers().context("failed to retrieve csv headers")?;

        let headers = known_headers
            .iter()
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect::<HashMap<&str, usize>>();

        for to_get in retrieve {
            match to_get {
                ColumnType::Name(name) => {
                    let Some(index) = headers.get(name.as_str()) else {
                        bail!(
                            "unknown column header specified. column: {name}\navail: {:#?}",
                            headers
                        );
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
                    bail!(
                        "unknown label column header specified. column: {name}\navail: {:#?}",
                        headers
                    );
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
pub fn map_record(
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

pub fn collect_records<R>(
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
