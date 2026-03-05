//! Transport-aware `Gen`s backed by the graph transport clock.

use crate::{
    gen::{Gen, GenContext, GenState},
    Resources, Sample,
};

/// Outputs the current transport position in beats at audio rate.
#[derive(Debug, Default, Clone, Copy)]
pub struct TransportBeatsGen;

impl Gen for TransportBeatsGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let Some(transport) = ctx.transport else {
            if let Some(output) = ctx.outputs.iter_mut().next() {
                output.fill(0.0);
            }
            return GenState::Continue;
        };
        if let Some(output) = ctx.outputs.iter_mut().next() {
            for (frame, sample) in output.iter_mut().enumerate() {
                *sample = transport
                    .beats_at(frame)
                    .map_or(0.0, |beats| beats.as_beats_f32());
            }
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "TransportBeatsGen"
    }
}

/// Outputs the current transport position in seconds at audio rate.
#[derive(Debug, Default, Clone, Copy)]
pub struct TransportSecondsGen;

impl Gen for TransportSecondsGen {
    fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
        let Some(transport) = ctx.transport else {
            if let Some(output) = ctx.outputs.iter_mut().next() {
                output.fill(0.0);
            }
            return GenState::Continue;
        };
        if let Some(output) = ctx.outputs.iter_mut().next() {
            for (frame, sample) in output.iter_mut().enumerate() {
                *sample = transport.seconds_at(frame).to_seconds_f64() as Sample;
            }
        }
        GenState::Continue
    }

    fn num_inputs(&self) -> usize {
        0
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "TransportSecondsGen"
    }
}
