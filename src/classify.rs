//! knn algorithm for classifying a given datapoint from the provided records
//!
//! this provides the implementation for the KNN algorithm.
//!
//! [`classify_datapoint`] is a convienience function that will allocate memory
//! for the calculated groups and collected records.
//!
//! [`classify_datapoint_buffered`] performs the actual calculation based on
//! the records provided to it.
use std::collections::HashMap;
use std::iter::Iterator;

/// convienience function that will allocate memory for the calculated groups
/// and collected records.
///
/// refer to [`classify_datapoint`]
pub fn classify_datapoint_owned<'a, 'b, F, R, D>(
    k: usize,
    records: R,
    algo: F,
    datapoint: &[f64],
) -> (usize, HashMap<&'a str, u32>)
where
    D: AsRef<[f64]>,
    R: Iterator<Item = (D, &'a str)>,
    F: Fn(&[f64], &[f64]) -> f64,
{
    let (_, max_size) = records.size_hint();

    let mut groups = HashMap::with_capacity(k);
    // collect the datapoints with the calculated distance function from
    // the provided datapoint
    let mut collected = if let Some(max_size) = max_size {
        Vec::with_capacity(max_size)
    } else {
        Vec::new()
    };

    let min = classify_datapoint(k, records, algo, datapoint, &mut collected, &mut groups);

    (min, groups)
}

/// performs the KNN algorithm on the provided records
///
/// this will calculate a single floating point value based on the result from
/// the provided algorithm that must accept a pair of [`&[f64]`]'s to compare
/// against. from that calculation the values will be sorted according to
/// [`f64::total_cmp`] for comparison and [`slice::sort_by`] for arranging the
/// values in assending order. once they have been sorted the first `k` values
/// will be inserted into the `groups` argument.
pub fn classify_datapoint<'a, F, R, D>(
    k: usize,
    records: R,
    algo: F,
    datapoint: &[f64],
    collected: &mut Vec<(f64, &'a str)>,
    groups: &mut HashMap<&'a str, u32>,
) -> usize
where
    // accepting any generic that can return a reference to a slice of f64's
    D: AsRef<[f64]>,
    // accepting any generic that is an iterator that returns a tuple of
    // D and the label associated with it
    R: Iterator<Item = (D, &'a str)>,
    F: Fn(&[f64], &[f64]) -> f64,
{
    // note for future improvement. this could be given as a replacement for the
    // current iterator and just require that the iterator yields a tuple of
    // an f64 and the label associated with it. could help solve some current
    // issues with the search code and the predict would require not too much
    // modification since it only runs this once.
    for (data, label) in records {
        collected.push((algo(&datapoint, data.as_ref()), label));
    }

    // sort the collected records by the distance function. since floats
    // dont directly implement the std::cmp::Ord trait we will sort by
    // f64::total_cmp
    collected.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    let min = std::cmp::min(k, collected.len());

    // collect the label groups and count how many are encountered
    for index in 0..min {
        groups
            .entry(collected[index].1)
            // increment if the group was previously added
            .and_modify(|counter| *counter += 1)
            // insert if not already existing
            .or_insert(1);
    }

    min
}

#[cfg(test)]
mod test {
    //! these are a set of tests to verify that the knn algorithm is properly
    //! implemented (or at least passes the required tests).
    //!
    //! datapoints used for classification and testing can be visually
    //! represented in the `test_datapoints_visual.png` at the root of the
    //! repository.
    use std::collections::HashMap;

    use crate::distance;

    use super::*;

    const T1: [f64; 2] = [1.5, 1.0];
    const T2: [f64; 2] = [1.5, 1.5];

    // (x, y) datapoints on a small graph
    const RECORDS: [([f64; 2], &'static str); 8] = [
        ([1.0, 1.0], "a"),
        ([2.0, 2.0], "b"),
        ([1.5, 2.5], "a"),
        ([1.0, 3.0], "b"),
        ([2.0, 1.0], "a"),
        ([1.0, 2.0], "b"),
        ([3.0, 1.0], "a"),
        ([2.5, 1.5], "b"),
    ];

    fn records_iter() -> impl std::iter::Iterator<Item = (&'static [f64], &'static str)> {
        RECORDS
            .iter()
            .map(|(data, label)| (data.as_slice(), *label))
    }

    #[test]
    fn classify_datapoint_k2_euclidean_t1() {
        let (_min, groups) = classify_datapoint_owned(2, records_iter(), distance::euclidean, &T1);

        let expected = HashMap::from([("a", 2)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k2_manhattan_t1() {
        let (_min, groups) = classify_datapoint_owned(2, records_iter(), distance::manhattan, &T1);

        let expected = HashMap::from([("a", 2)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k2_euclidean_t2() {
        let (_min, groups) = classify_datapoint_owned(2, records_iter(), distance::euclidean, &T2);

        // 4 datapoints should be equidistant from the desired one so it will
        // depend more on ordering of floating point values when we sort the
        // values which should be a stable sort. refer to f64::total_cmp and
        // slice::sort_by
        let expected = HashMap::from([("a", 1), ("b", 1)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k2_manhattan_t2() {
        let (_min, groups) = classify_datapoint_owned(2, records_iter(), distance::euclidean, &T2);

        // similar to the euclidean, we should expect 4 equidistant datapoints
        // and sort by the specification in slice and f64
        let expected = HashMap::from([("a", 1), ("b", 1)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k3_euclidean_t1() {
        let (_min, groups) = classify_datapoint_owned(3, records_iter(), distance::euclidean, &T1);

        // there will be ambiguity between which b datapoint is selected but
        // it should still just be 1
        let expected = HashMap::from([("a", 2), ("b", 1)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k3_manhattan_t1() {
        let (_min, groups) = classify_datapoint_owned(3, records_iter(), distance::manhattan, &T1);

        // should be similar to the euclidean but still result in the same
        // groups
        let expected = HashMap::from([("a", 2), ("b", 1)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k3_euclidean_t2() {
        let (_min, groups) = classify_datapoint_owned(3, records_iter(), distance::euclidean, &T2);

        let expected = HashMap::from([("a", 2), ("b", 1)]);

        assert_eq!(groups, expected);
    }

    #[test]
    fn classify_datapoint_k3_manhattan_t2() {
        let (_min, groups) = classify_datapoint_owned(3, records_iter(), distance::manhattan, &T2);

        let expected = HashMap::from([("a", 2), ("b", 1)]);

        assert_eq!(groups, expected);
    }
}
