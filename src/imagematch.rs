//! An implementation of the algorithm described in http://www.edwardrosten.com/work/taylor_2009_robust.pdf.

use image::{GenericImage,Luma,Pixel};
use stats::{mean,stddev};

/// Counts of samples seen at a given location which
/// fall into each intensity range.
pub type PointHist = [u32; 5];

/// A point histogram of observed pixel intensities at
/// each location in an 8x8 template.
struct PatchModel {
    hists: [PointHist; 64]
}

/// The result of quantising the 5-bin histograms
/// of a patch model. Element i of the wrapped array
/// contains the quantised entries from the ith bin
/// in each of the 64 location bins.
pub type PatchDescriptor = [u64; 5];

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

/// Returns the number of positions where the sampled pixel lies
/// in a bin which has value 1 in the model, i.e. in a bin containing
/// few training samples.
pub fn discrepancy(patch: &PatchDescriptor, model: &PatchDescriptor) -> u16 {
    let mut count = 0u16;
    for i in 0..5 {
        let intersect = patch[i] & model[i];
        count += intersect.count_ones() as u16;
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

/// Samples an 8x8 patch of every-other-pixel around a given point.
/// Return None if the pixel is too near an image boundary
pub fn sample_patch<I>(image: &I, x: u32, y: u32) -> Option<[u8; 64]>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let (width, height) = image.dimensions();
    if x < 7 || y < 7 || x + 7 >= width || y + 7 >= height {
        return None;
    }

    // +/- 1, 3, 5, 7
    let offsets = (0..8).map(|x| 2 * x - 7).collect::<Vec<_>>();

    let mut count = 0;
    let mut sample = [0u8; 64];

    for dy in offsets.iter() {
        for dx in offsets.iter() {
            let p = image.get_pixel(x + dx, y + dy)[0];
            sample[count] = p;
            count += 1;
        }
    }

    Some(sample)
}

#[cfg(test)]
mod test {

    use super::{
        bin,
        count_bits,
        set_bit
    };

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
