
extern crate stats;

use stats::{
    mean,
    stddev
};

/// Model of patches as used in that paper.
/// Need a way of namespacing these, as this
/// is a hopelessly vague name.
struct PatchModel {
    hists: [PointHist; 64]
}

impl PatchModel {

    /// Add a patch of normalised pixel intensities to the
    /// per-pixel-location histograms.
    pub fn add_sample(&mut self, sample: &[u8; 64]) {
        for i in 0..64 {
            self.hists[i][bin(sample[i]) as usize] += 1;
        }
    }

    /// Convert the 64 5-bin intensity histograms into
    /// 5 64-bit ints where the ith bit of the jth output int
    /// is set to 1 if fewer than 5% of the values in the ith
    /// histogram lie in bin j.
    pub fn quantise(&self) -> PatchDescriptor {
        let mut descriptor = [0u64; 5];

        for h in 0..64 {
            let hist = self.hists[h];
            let sum = hist.iter().fold(0, |x, &y| x + y);
            for i in 0..5 {
                let fraction = hist[i] as f32 / sum as f32;
                if fraction < 0.05 {
                    descriptor[i] = set_bit(descriptor[i], h as u8);
                }
            }
        }

        descriptor
    }
}

fn set_bit(n: u64, pos: u8) -> u64 {
    n | (1 << pos)
}

fn bin(value: u8) -> u8 {
    value / 52
}

/// Counts of samples seen at a given location which
/// fall into each intensity range.
pub type PointHist = [u32; 5];

/// The result of quantising the 5-bin histograms
/// of a patch model. Element i of the wrapped array
/// contains the quantised entries from the ith bin
/// in each of the 64 location bins.
pub type PatchDescriptor = [u64; 5];

/// Returns the number of positions where the sampled pixel lies
/// in a bin which has value 1 in the model, i.e. in a bin containing
/// few training samples.
pub fn discrepancy(patch: &PatchDescriptor, model: &PatchDescriptor)
    -> u16 {
    let mut count = 0u16;
    for i in 0..5 {
        let intersect = patch[i] & model[i];
        count += count_bits(intersect) as u16;
    }
    count
}

/// Normalise a range of values to have mean 0
/// and variance 1.
fn normalise(patch: &[u8; 64]) -> [u8; 64] {
    let mean = mean(patch.iter().map(|x| *x));
    let stddev = stddev(patch.iter().map(|x| *x));
    let mut normalised = [0u8; 64];
    for i in 0..64 {
        let v = (patch[i] as f64 - mean) / stddev;
        normalised[i] = v as u8;
    }
    normalised
}

/// Count bits set in a u64.
// TODO: Use the intrinsic
fn count_bits(x: u64) -> u8 {
    let mut y = x;
    let mut count = 0u8;
    loop {
        if y == 0 {
            break;
        }
        count += 1;
        y = y & y - 1;
    }
    count
}

#[cfg(test)]
mod test {

    use super::{
        bin,
        count_bits,
        set_bit
    };

    #[test]
    fn test_count_bits() {
        assert_eq!(count_bits(0), 0);
        assert_eq!(count_bits(1), 1);
        assert_eq!(count_bits(2), 1);
        assert_eq!(count_bits(3), 2);
        assert_eq!(count_bits(4), 1);
        assert_eq!(count_bits(5), 2);
    }

    #[test]
    fn test_bin() {
        assert_eq!(bin(0), 0);
        assert_eq!(bin(51), 0);
        assert_eq!(bin(52), 1);
        assert_eq!(bin(255), 4);
    }

    #[test]
    fn test_set_bit() {
        assert_eq!(set_bit(0, 0), 1);
        assert_eq!(set_bit(0, 1), 2);
        assert_eq!(set_bit(0, 2), 4);
        assert_eq!(set_bit(1, 1), 3);
    }
}
