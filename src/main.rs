use std::fs::OpenOptions;
use std::io::{BufReader, ErrorKind};
use std::path::PathBuf;

use anyhow::{Error, bail};
use clap::{Parser, Subcommand};

mod classify;
mod cli;
mod csv;
mod distance;
mod predict;
mod search;

use csv::ReaderBuilder;

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();

    let result = OpenOptions::new().read(true).open(&args.file);

    let file = match result {
        Ok(f) => f,
        Err(err) => match err.kind() {
            ErrorKind::NotFound => bail!("the requested csv file was not found"),
            _ => return Err(Error::new(err).context("failed to load csv file")),
        },
    };

    let reader = ReaderBuilder::new()
        .has_headers(!args.no_header)
        .from_reader(BufReader::new(file));

    match args.cmd {
        KnnCmd::Predict(arg) => predict::knn_predict(reader, arg),
        KnnCmd::Search(arg) => search::knn_search(reader, arg),
    }
}

/// a simple k nearest neighbors (knn) calculator that loads a csv file
/// containing records to use for estimating a given datapoint.
#[derive(Debug, Parser)]
struct CliArgs {
    /// indicates that the csv contains no header row
    #[arg(long)]
    no_header: bool,

    /// path to the csv file to load
    #[arg(short, long)]
    file: PathBuf,

    #[command(subcommand)]
    cmd: KnnCmd,
}

#[derive(Debug, Subcommand)]
pub enum KnnCmd {
    /// attempts to predict a specific datapoint with the specified dataset
    Predict(predict::PredictArgs),
    /// searches for an optimal set of arguments to predict values with
    Search(search::SearchArgs),
}
