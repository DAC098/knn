use std::convert::Infallible;
use std::str::FromStr;

use clap::ValueEnum;

/// represents the k value to use for calculations
#[derive(Debug, Clone)]
pub struct KValue((usize, usize, usize));

impl KValue {
    fn parse_range(given: &str) -> Result<Option<(usize, usize)>, &'static str> {
        if let Some((low, high)) = given.split_once('-') {
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

    pub fn get_range(&self, total: usize) -> std::iter::StepBy<std::ops::Range<usize>> {
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
        } else if let Some((low, high)) = Self::parse_range(given)? {
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

#[derive(Debug, Clone)]
pub struct Datapoint(Vec<f64>);

impl Datapoint {
    pub fn into_inner(self) -> Vec<f64> {
        self.0
    }
}

impl FromStr for Datapoint {
    type Err = &'static str;

    fn from_str(given: &str) -> Result<Self, Self::Err> {
        let mut rtn = Vec::new();
        let iter = given.split(',').map(f64::from_str);

        for result in iter {
            rtn.push(result.map_err(|_| "failed to parse datapoint")?);
        }

        Ok(Self(rtn))
    }
}
