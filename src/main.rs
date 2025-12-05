use std::collections::HashMap;
use std::convert::Infallible;
use std::fs::OpenOptions;
use std::io::{BufReader, ErrorKind};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, bail};
use clap::Parser;

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    if args.columns.is_empty() {
        bail!("no columns specified to pull numeric data from");
    }

    let result = OpenOptions::new().read(true).open(&args.filename);

    let file = match result {
        Ok(f) => f,
        Err(err) => match err.kind() {
            ErrorKind::NotFound => bail!("the requested csv file was not found"),
            _ => bail!(err),
        },
    };
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(!args.no_header)
        .from_reader(BufReader::new(file));

    let (label, columns) = get_columns_and_label(&mut reader, &args.label, &args.columns)?;

    Ok(())
}

#[derive(Debug, Parser)]
struct CliArgs {
    #[arg(short, default_value = "3")]
    k: u32,

    #[arg(short, long = "col")]
    columns: Vec<ColumnType>,

    #[arg(long)]
    no_header: bool,

    label: ColumnType,

    filename: PathBuf,
}

#[derive(Debug, Clone)]
pub enum ColumnType {
    Name(String),
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
