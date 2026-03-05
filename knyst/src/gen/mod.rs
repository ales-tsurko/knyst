//! A [`Gen`] is anything that implements the [`Gen`] trait and is the code of Knyst. Gen is short for Generator, often called unit generators in other contexts, a term that dates back to [Max Mathews' program MUSIC from 1957](https://en.wikipedia.org/wiki/MUSIC-N)
//!
//! The best way of implementing [`Gen`] on your own type is using the [`impl_gen`] macro.
mod basic_gens;
pub mod dynamics;
pub mod noise;
pub use noise::*;
pub mod random;
pub use basic_gens::*;
mod smoothing;
pub use smoothing::*;
mod osc;
pub mod transport;
use crate::{
    graph::{NodeId, TransportState},
    node_buffer::NodeBufferRef,
    resources::Resources,
    scheduling::MusicalTimeMap,
    time::{Beats, Seconds},
    Sample,
};
pub use osc::*;
pub use transport::*;
pub mod delay;
pub mod filter;

#[allow(unused)]
use crate::graph::{Connection, Graph};
#[allow(unused)]
use knyst_macro::impl_gen;

/// If it implements Gen, it can be a `Node` in a [`Graph`].
pub trait Gen {
    /// The input and output buffers are both indexed using \[in/out_index\]\[sample_index\].
    ///
    /// - *inputs*: The inputs to the Gen filled with the relevant values. May be any size the same or larger
    ///   than the number of inputs to this particular Gen.
    ///
    /// - *outputs*: The buffer to place the result of the Gen inside. This buffer may contain any data and
    ///   will not be zeroed. If the output should be zero, the Gen needs to write zeroes into the output
    ///   buffer. This buffer will be correctly sized to hold the number of outputs that the Gen requires.
    fn process(&mut self, ctx: GenContext, resources: &mut Resources) -> GenState;
    /// The number of inputs this `Gen` takes. Determines how big the input buffer is.
    fn num_inputs(&self) -> usize;
    /// The number of outputs this `Gen` produces. Determines how big the output buffer is.
    fn num_outputs(&self) -> usize;
    /// Initialize buffers etc.
    /// Default: noop
    #[allow(unused)]
    fn init(&mut self, block_size: usize, sample_rate: Sample, node_id: NodeId) {}
    /// Return a label for a given input channel index. This sets the label in the [`Connection`] API.
    #[allow(unused)]
    fn input_desc(&self, input: usize) -> &'static str {
        ""
    }
    /// Return a label for a given output channel index. This sets the label in the [`Connection`] API.
    #[allow(unused)]
    fn output_desc(&self, output: usize) -> &'static str {
        ""
    }
    /// A name identifying this `Gen`.
    fn name(&self) -> &'static str {
        "no_name"
    }
}

/// Gives access to the inputs and outputs buffers of a node for processing.
#[derive(Clone, Copy)]
pub struct TransportContext<'a> {
    state: TransportState,
    block_start_samples: u64,
    sample_rate: Sample,
    musical_time_map: Option<&'a MusicalTimeMap>,
}

impl<'a> TransportContext<'a> {
    pub(crate) fn new(
        state: TransportState,
        block_start_samples: u64,
        sample_rate: Sample,
        musical_time_map: Option<&'a MusicalTimeMap>,
    ) -> Self {
        Self {
            state,
            block_start_samples,
            sample_rate,
            musical_time_map,
        }
    }

    /// Returns the current transport state.
    pub fn state(&self) -> TransportState {
        self.state
    }

    /// Returns the graph sample rate used by this transport context.
    pub fn sample_rate(&self) -> Sample {
        self.sample_rate
    }

    /// Returns the transport sample position at the start of the current block.
    pub fn block_start_samples(&self) -> u64 {
        self.block_start_samples
    }

    /// Returns the transport sample position for one frame offset in the current block.
    pub fn samples_at(&self, frame_offset: usize) -> u64 {
        match self.state {
            TransportState::Playing => self.block_start_samples.saturating_add(frame_offset as u64),
            TransportState::Paused => self.block_start_samples,
        }
    }

    /// Returns the transport seconds position for one frame offset in the current block.
    pub fn seconds_at(&self, frame_offset: usize) -> Seconds {
        Seconds::from_samples(self.samples_at(frame_offset), self.sample_rate as u64)
    }

    /// Returns the transport beats position for one frame offset in the current block.
    pub fn beats_at(&self, frame_offset: usize) -> Option<Beats> {
        self.musical_time_map
            .map(|map| map.seconds_to_beats(self.seconds_at(frame_offset)))
    }

    pub(crate) fn advanced(self, frame_offset: usize) -> Self {
        Self {
            block_start_samples: self.samples_at(frame_offset),
            ..self
        }
    }

    pub(crate) fn resampled(self, sample_rate: Sample) -> Self {
        if self.sample_rate == sample_rate {
            return self;
        }
        let scaled_samples = ((self.block_start_samples as f64) * (sample_rate as f64)
            / (self.sample_rate as f64)) as u64;
        Self {
            block_start_samples: scaled_samples,
            sample_rate,
            ..self
        }
    }
}

/// Gives access to the inputs and outputs buffers of a node for processing.
pub struct GenContext<'a, 'b, 'c> {
    /// Input buffers to the Gen.
    pub inputs: &'a NodeBufferRef,
    /// Output buffers the Gen is supposed to fill.
    pub outputs: &'b mut NodeBufferRef,
    /// The sample rate of the [`Graph`] that the current Gen is in.
    pub sample_rate: Sample,
    /// The current transport context for this block, if transport is available.
    pub transport: Option<TransportContext<'c>>,
}
impl<'a, 'b, 'c> GenContext<'a, 'b, 'c> {
    /// Returns the current block size
    pub fn block_size(&self) -> usize {
        self.outputs.block_size()
    }
}

/// The Gen should return Continue unless it needs to free itself or the Graph it is in.
///
/// No promise is made as to when the node or the Graph will be freed so the Node needs to do the right thing
/// if run again the next block. E.g. a node returning `FreeSelfMendConnections` is expected to act as a
/// connection bridge from its non constant inputs to its outputs as if it weren't there. Only inputs with a
/// corresponding output should be passed through, e.g. in\[0\] -> out\[0\], in\[1\] -> out\[1\], in\[2..5\] go nowhere.
///
/// The FreeGraph and FreeGraphMendConnections values also return the relative
/// sample in the current block after which the graph should return 0 or connect
/// its non constant inputs to its outputs.
#[derive(Debug, Clone, Copy)]
pub enum GenState {
    /// Continue running
    Continue,
    /// Free the node containing the Gen
    FreeSelf,
    /// Free the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeSelfMendConnections,
    /// Free the graph containing the node containing the Gen.
    FreeGraph(usize),
    /// Free the graph containing the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeGraphMendConnections(usize),
}

/// Specify what happens when a [`Gen`] is done with its processing. This translates to a [`GenState`] being returned from the [`Gen`], but without additional parameters.
#[derive(Debug, Clone, Copy)]
pub enum StopAction {
    /// Continue running
    Continue,
    /// Free the node containing the Gen
    FreeSelf,
    /// Free the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeSelfMendConnections,
    /// Free the graph containing the node containing the Gen.
    FreeGraph,
    /// Free the graph containing the node containing the Gen, bridging its input node(s) to its output node(s).
    FreeGraphMendConnections,
}
impl StopAction {
    /// Convert the [`StopAction`] into a [`GenState`].
    ///
    /// `stop_sample` is only used for the `FreeGraph` and
    /// `FreeGraphMendConnections` variants to communicate from what sample time
    /// the graph outputs should be 0.
    #[must_use]
    pub fn to_gen_state(&self, stop_sample: usize) -> GenState {
        match self {
            StopAction::Continue => GenState::Continue,
            StopAction::FreeSelf => GenState::FreeSelf,
            StopAction::FreeSelfMendConnections => GenState::FreeSelfMendConnections,
            StopAction::FreeGraph => GenState::FreeGraph(stop_sample),
            StopAction::FreeGraphMendConnections => GenState::FreeGraphMendConnections(stop_sample),
        }
    }
}
