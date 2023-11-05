//! # Delay
//! This module contains some basic delay Gens

use crate as knyst;
use crate::gen::Gen;
use crate::gen::GenContext;
use crate::gen::GenState;
use crate::SampleRate;
use knyst_macro::impl_gen;

use crate::time::Superseconds;
use crate::Resources;
use crate::Sample;

/// Delay by an integer number of samples, no interpolation. This is good for e.g. triggers.
///
/// *inputs*
/// 0. "signal": input signal, the signal to be delayed
/// 1. "delay_time": the delay time in seconds (will be truncated to the nearest sample)
/// *outputs*
/// 0. "signal": the delayed signal
pub struct SampleDelay {
    buffer: Vec<Sample>,
    write_position: usize,
    max_delay_length: Superseconds,
}
impl SampleDelay {}

#[impl_gen]
impl SampleDelay {
    #[new]
    /// Create a new SampleDelay with a maximum delay time.
    pub fn new(max_delay_length: Superseconds) -> Self {
        Self {
            buffer: vec![0.0; 0],
            max_delay_length,
            write_position: 0,
        }
    }
    #[process]
    fn process(
        &mut self,
        signal: &[Sample],
        delay_time: &[Sample],
        output: &mut [Sample],
        sample_rate: SampleRate,
    ) -> GenState {
        let sig_buf = signal;
        let time_buf = delay_time;
        let out_buf = output;
        for ((&input, &time), o) in sig_buf.iter().zip(time_buf).zip(out_buf.iter_mut()) {
            self.buffer[self.write_position] = input;
            let delay_samples = (time * *sample_rate) as usize;
            *o = self.buffer
                [(self.write_position + self.buffer.len() - delay_samples) % self.buffer.len()];
            self.write_position = (self.write_position + 1) % self.buffer.len();
        }
        GenState::Continue
    }

    #[init]
    fn init(&mut self, sample_rate: SampleRate) {
        self.buffer =
            vec![0.0; (self.max_delay_length.to_seconds_f64() * sample_rate.to_f64()) as usize];
        self.write_position = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AllpassInterpolator {
    coeff: f64,
    prev_input: f64,
    prev_output: f64,
}

impl AllpassInterpolator {
    pub fn new() -> Self {
        Self {
            coeff: 0.0,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
    /// Reset any state to 0
    pub fn clear(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
    pub fn set_delta(&mut self, delta: f64) {
        self.coeff = (1.0 - delta) / (1.0 + delta);
    }
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.coeff * (input - self.prev_output) + self.prev_input;
        self.prev_output = output;
        self.prev_input = input;
        output
    }
}

#[derive(Clone, Debug)]
pub struct AllpassDelay {
    buffer: Vec<f64>,
    buffer_size: usize,
    frame: usize,
    num_frames: usize,
    allpass: AllpassInterpolator,
}

impl AllpassDelay {
    pub fn new(buffer_size: usize) -> Self {
        let buffer = vec![0.0; buffer_size];
        Self {
            buffer,
            buffer_size,
            frame: 0,
            num_frames: 1,
            allpass: AllpassInterpolator::new(),
        }
    }
    /// Read the current frame from the delay and allpass interpolate. Read before `write_and_advance` for the correct sample.
    pub fn read(&mut self) -> f64 {
        let index = self.frame % self.buffer.len();
        self.allpass.process_sample(self.buffer[index])
    }
    pub fn set_delay_in_frames(&mut self, num_frames: f64) {
        self.num_frames = num_frames.floor() as usize;
        self.allpass
            .set_delta((num_frames - self.num_frames as f64) as f64);
    }
    pub fn clear(&mut self) {
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
        self.allpass.clear();
    }
    /// Reset the delay with a new length in frames
    pub fn set_delay_in_frames_and_clear(&mut self, num_frames: f64) {
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
        self.set_delay_in_frames(num_frames);
        // println!(
        //     "num_frames: {}, delta: {}",
        //     self.num_frames,
        //     (num_frames - self.num_frames as f64)
        // );
    }
    /// Write a new value into the delay after incrementing the sample pointer.
    pub fn write_and_advance(&mut self, input: f64) {
        self.frame += 1;
        let index = (self.frame + self.num_frames) % self.buffer_size;
        self.buffer[index] = input;
    }
}

#[derive(Clone, Debug)]
pub struct AllpassFeedbackDelay {
    pub feedback: f64,
    allpass_delay: AllpassDelay,
}
impl AllpassFeedbackDelay {
    pub fn new(max_delay_samples: usize) -> Self {
        let allpass_delay = AllpassDelay::new(max_delay_samples);
        let s = Self {
            feedback: 0.,
            allpass_delay,
        };
        s
    }
    pub fn set_delay_in_frames(&mut self, delay_length: f64) {
        self.allpass_delay.set_delay_in_frames(delay_length);
    }
    /// Clear any values in the delay
    pub fn clear(&mut self) {
        self.allpass_delay.clear();
    }
    // fn calculate_values(&mut self) {
    //     self.feedback = (0.001 as Sample).powf(self.delay_time / self.decay_time.abs())
    //         * self.decay_time.signum();
    //     let delay_samples = self.delay_time * self.sample_rate;
    //     self.allpass_delay.set_num_frames(delay_samples as f64);
    // }
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let delayed_sig = self.allpass_delay.read();
        let delay_write = delayed_sig * self.feedback + input;
        self.allpass_delay.write_and_advance(delay_write);

        delayed_sig - self.feedback * delay_write
    }
}
