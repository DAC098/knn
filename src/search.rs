use std::collections::HashMap;

use anyhow::bail;
use clap::Args;

use crate::cli::{AlgoType, ColumnType, KValue};
use crate::csv::{Reader, get_columns_and_label, KnnRecord, collect_records};
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

    /// the percent of data to test against
    #[arg(long, default_value = "0.25")]
    test: f64,

    /// the colume to use as the label
    #[arg(long)]
    label: ColumnType,
}

struct SearchResult {
    k: usize,
    percent: f64,
    cols: Vec<usize>,
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

    // we are going to keep this pre-allocated since it is being reused multiple
    // times so we will just clear it when needed vs constaint memory
    // allocations
    let mut collected: Vec<(f64,  &KnnRecord)> = Vec::with_capacity(train.len());
    let mut results = Vec::new();

    println!("train size: {} test size: {}", train.len(), test.len());

    // we are using the train dataset and manually iterating through
    // the test dataset for datapoints to use for testing
    for k in arg.k.get_range(train.len()) {
        let mut selected: Vec<(usize, usize)> = Vec::new();
        let mut avail: Vec<(usize, usize)> = columns
            .iter()
            .enumerate()
            .map(|(index, col)| (index, *col))
            .collect();
        // can also pre allocate this since it will be reused for
        // multiple test records
        let mut groups = HashMap::with_capacity(k);

        println!("k: {k}");

        while !avail.is_empty() {
            let mut best = None::<(usize, f64, (usize, usize))>;
            let mut a_buf = Vec::with_capacity(selected.len() + 1);
            let mut b_buf = Vec::with_capacity(selected.len() + 1);

            for (avail_index, (index, col)) in avail.iter().enumerate() {
                let mut passed = 0;
                let mut failed = 0;
                let mut unknown = 0;

                for test_record in &test {
                    collected.clear();
                    groups.clear();

                    collect_data(test_record, &mut a_buf, &selected, *index);

                    // iterate through test records to collect and get distance for
                    // current test record
                    for train_record in &train {
                        collect_data(train_record, &mut b_buf, &selected, *index);

                        collected.push((algo(&a_buf, &b_buf), train_record));
                    }

                    // follow a similar process as the knn predict
                    collected.sort_by(|(a, _), (b, _)| a.total_cmp(b));

                    let min = std::cmp::min(k, collected.len());

                    for index in 0..min {
                        groups
                            .entry(collected[index].1.label.as_str())
                            .and_modify(|counter| *counter += 1)
                            .or_insert(1);
                    }

                    let mut largest = None::<(f64, &str)>;

                    for (key, count) in &groups {
                        let prob = (*count as f64) / (min as f64);

                        // find the largest percent value from the collected
                        // labels and store that.
                        largest = if let Some((percent, label)) = largest {
                            if prob > percent {
                                Some((prob, key))
                            } else {
                                Some((percent, label))
                            }
                        } else {
                            Some((prob, key))
                        };
                    }

                    // check to see if the largest value found is valid.
                    // increment values accordingly
                    if let Some((_, label)) = largest {
                        if label == test_record.label {
                            passed += 1;
                        } else {
                            failed += 1;
                        }
                    } else {
                        unknown += 1;
                    }
                }

                // this is not RMSE or similar and instead just calculating the
                // percentage of records correct. the largest percentage will
                // be included in the `selected` list. output the results for
                // this iteration

                let p_correct = (passed as f64) / (test.len() as f64);

                print!("       ");

                for (_, sel_col) in &selected {
                    print!(" {sel_col}");
                }

                println!(" {col} | passed: {passed} {p_correct:.2} failed: {failed} unknown: {unknown}");

                best = if let Some((best_index, best_p, (index_ref, best_col))) = best {
                    if best_p > p_correct {
                        Some((best_index, best_p, (index_ref, best_col)))
                    } else {
                        Some((avail_index, p_correct, (*index, *col)))
                    }
                } else {
                    Some((avail_index, p_correct, (*index, *col)))
                };
            }

            let Some((best_index, best_p, (index, col))) = best else {
                break;
            };

            // updated the selected columns and remove from available so we
            // make progress and don't repeat columns
            selected.push((index, col));
            avail.remove(best_index);

            let mut cols = Vec::new();

            for (_, col) in &selected {
                cols.push(*col);
            }

            // store the results to be output later
            results.push(SearchResult {
                k,
                percent: best_p * 100.0,
                cols,
            });
        }
    }

    for record in results {
        print!("k {} % {:.2} cols:", record.k, record.percent);

        for col in record.cols {
            print!(" {col}");
        }

        println!();
    }

    Ok(())
}

/// split the specified list of records based on the label provided
///
/// ordering is preserved from the original list
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
        // split the record groups based on the split specified.
        let amount = (records.len() as f64 * split).floor() as usize;

        train.extend(records.split_off(amount));
        test.extend(records);
    }

    (train, test)
}

fn collect_data(
    record: &KnnRecord,
    buf: &mut Vec<f64>,
    selected: &Vec<(usize, usize)>,
    checking: usize,
) {
    buf.clear();

    // collect the datapoints from test record
    for (index, _) in selected {
        buf.push(record.data[*index]);
    }

    buf.push(record.data[checking]);
}
