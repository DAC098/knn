/// calculates the euclidean distance between 2 sets of datapoints
pub fn euclidean(a_data: &[f64], b_data: &[f64]) -> f64 {
    // we will expect the total datapoints from a and b to be the same and just
    // zip them together for the iterator chain
    a_data
        .iter()
        .zip(b_data)
        .map(|(a, b)| (a - b).powf(2.0))
        .sum::<f64>()
        .sqrt()
}

/// calculates the manhattan distance between 2 sets of datapoints
pub fn manhattan(a_data: &[f64], b_data: &[f64]) -> f64 {
    // similar to the euclidean distance expectation
    a_data
        .iter()
        .zip(b_data)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
}
