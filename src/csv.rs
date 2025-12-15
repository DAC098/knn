use std::collections::HashMap;

use anyhow::{Context, bail};
pub use csv::{Reader, ReaderBuilder, StringRecord};

use crate::cli::ColumnType;

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
