//! Noise
use knyst_macro::impl_gen;

use super::random::next_randomness_seed;
use super::GenState;
use crate as knyst;
use crate::Sample;

/// Whate noise (fastrand RNG based on wyrand)
pub struct WhiteNoise {
    rng: fastrand::Rng,
}
#[impl_gen(range = normal)]
impl WhiteNoise {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        let mut rng = fastrand::Rng::new();
        rng.seed(next_randomness_seed());
        Self { rng }
    }
    #[allow(missing_docs)]
    pub fn process(&mut self, output: &mut [Sample]) -> GenState {
        for out in output.iter_mut() {
            *out = self.rng.f32() as Sample * 2.0 - 1.0;
        }
        GenState::Continue
    }
}
const PINK_NOISE_OCTAVES: u32 = 9;
/// Pink noise
///
/// Usually outputs in the +-0.75 range and cannot surpass +-1.0
///
/// Computed using the Voss-MacCartney method of stacking white noise in lower and lower octaves.
/// Algorithms from: https://www.firstpr.com.au/dsp/pink-noise/#Filtering
pub struct PinkNoise {
    white_noises: [Sample; PINK_NOISE_OCTAVES as usize],
    always_on_white_noise: Sample,
    counter: u32,
    mask: u32,
    pink: Sample,
    rng: fastrand::Rng,
}

#[impl_gen(range = normal)]
impl PinkNoise {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        let mut rng = fastrand::Rng::new();
        rng.seed(next_randomness_seed());
        Self {
            white_noises: [0.; PINK_NOISE_OCTAVES as usize],
            always_on_white_noise: 0.,
            counter: 1,
            mask: 2_u32.pow(PINK_NOISE_OCTAVES - 1),
            pink: 0.,
            rng,
        }
    }
    fn noise_index(&self) -> u32 {
        assert!(self.counter > 0);
        assert!(self.counter <= self.mask);

        self.counter.trailing_zeros()
    }

    fn increment_counter(&mut self) {
        assert!(self.counter > 0);
        assert!(self.counter <= self.mask);

        self.counter = self.counter & (self.mask - 1);
        self.counter = self.counter + 1;
    }

    #[allow(missing_docs)]
    pub fn process_sample(&mut self) -> Sample {
        let index = self.noise_index() as usize;
        assert!(index < PINK_NOISE_OCTAVES as usize);

        self.pink = self.pink - self.white_noises[index];
        self.white_noises[index] = self.rng.f32() as Sample * 2.0 - 1.0;
        self.pink = self.pink + self.white_noises[index];

        self.pink = self.pink - self.always_on_white_noise;
        self.always_on_white_noise = self.rng.f32() as Sample * 2.0 - 1.0;
        self.pink = self.pink + self.always_on_white_noise;

        self.increment_counter();

        self.pink / (PINK_NOISE_OCTAVES as Sample + 1.)
    }

    #[allow(missing_docs)]
    pub fn process(&mut self, output: &mut [Sample]) -> GenState {
        for out in output.iter_mut() {
            *out = self.process_sample();
        }
        GenState::Continue
    }
}
