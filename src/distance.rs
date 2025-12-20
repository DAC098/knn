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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check_euclidean() {
        let a_data = [3.0, 3.0];
        let b_data = [1.0, 1.0];

        let calc = euclidean(&a_data, &b_data);

        assert_eq!(
            calc,
            ((a_data[0] - b_data[0]).powf(2.0) + (a_data[0] - b_data[0]).powf(2.0)).sqrt()
        );
        assert_eq!(calc, 8.0f64.sqrt());
    }

    #[test]
    fn check_manhattan() {
        let a_data = [4.0, 4.0];
        let b_data = [2.0, 2.0];

        let calc = manhattan(&a_data, &b_data);

        assert_eq!(
            calc,
            (a_data[0] - b_data[0]).abs() + (a_data[1] - b_data[1]).abs()
        );
        assert_eq!(calc, 4.0);
    }
}
