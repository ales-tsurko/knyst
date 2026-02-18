//! A [`KnystSphere`] contains one instance of Knyst running on a backend. You can
//! create multiple [`KnystSphere`]s in one program and keep an explicit
//! [`crate::modal_interface::KnystContext`] for each one.

use crate::controller::Controller;
#[allow(unused)]
use crate::controller::KnystCommands;
use crate::KnystError;
use crate::{resources::ResourcesSettings, Resources, Sample};
use std::time::Duration;

use crate::{
    graph::{Graph, GraphSettings, RunGraphSettings},
    modal_interface::{KnystContext, SphereError},
    prelude::{AudioBackend, MultiThreadedKnystCommands},
};

/// One instance of Knyst, responsible for its overall context.
pub struct KnystSphere {
    #[allow(unused)]
    name: String,
    context: KnystContext,
}

impl KnystSphere {
    /// Create a graph matching the settings and the backend and start processing with a helper thread.
    pub fn start<B: AudioBackend>(
        backend: &mut B,
        settings: SphereSettings,
        error_handler: impl FnMut(KnystError) + Send + 'static,
    ) -> Result<Self, SphereError> {
        let resources = Resources::new(settings.resources_settings)?;
        let graph_settings = GraphSettings {
            name: settings.name.clone(),
            num_inputs: backend
                .native_input_channels()
                .unwrap_or(settings.num_inputs),
            num_outputs: backend
                .native_output_channels()
                .unwrap_or(settings.num_outputs),
            block_size: backend.block_size().unwrap_or(64),
            sample_rate: backend.sample_rate() as Sample,
            ..Default::default()
        };
        let graph: Graph = Graph::new(graph_settings);
        let k = backend.start_processing(
            graph,
            resources,
            RunGraphSettings {
                scheduling_latency: settings.scheduling_latency,
            },
            Box::new(error_handler),
        )?;
        let s = Self {
            name: settings.name,
            context: KnystContext::new(k),
        };
        s.context.activate();
        Ok(s)
    }
    /// Start the sphere and upload it, but return the Controller and make it the user's responsibility.
    /// Unless you are implementing a custom backend solution, you probably want the [`KnystSphere::start`] method.
    pub fn start_return_controller<B: AudioBackend>(
        backend: &mut B,
        settings: SphereSettings,
        error_handler: impl FnMut(KnystError) + Send + 'static,
    ) -> Result<(Self, Controller), SphereError> {
        let resources = Resources::new(settings.resources_settings)?;
        let graph_settings = GraphSettings {
            name: settings.name.clone(),
            num_inputs: backend
                .native_input_channels()
                .unwrap_or(settings.num_inputs),
            num_outputs: backend
                .native_output_channels()
                .unwrap_or(settings.num_outputs),
            block_size: backend.block_size().unwrap_or(64),
            sample_rate: backend.sample_rate() as Sample,
            ring_buffer_size: settings.scheduling_ring_buffer_capacity,
            ..Default::default()
        };
        let graph: Graph = Graph::new(graph_settings);
        let controller = backend.start_processing_return_controller(
            graph,
            resources,
            RunGraphSettings {
                scheduling_latency: settings.scheduling_latency,
            },
            Box::new(error_handler),
        )?;
        let s = Self {
            name: settings.name,
            context: KnystContext::new(controller.get_knyst_commands()),
        };
        s.context.activate();
        Ok((s, controller))
    }
    /// Return an object implementing [`KnystCommands`]
    #[must_use]
    pub fn commands(&self) -> MultiThreadedKnystCommands {
        self.context.commands()
    }

    /// Return the explicit context for this sphere.
    #[must_use]
    pub fn context(&self) -> KnystContext {
        self.context.clone()
    }
}

/// Settings pertaining to a sphere
#[derive(Debug, Clone)]
pub struct SphereSettings {
    /// The name of the sphere
    pub name: String,
    /// Settings for initialising the internal [`Resources`]
    pub resources_settings: ResourcesSettings,
    /// num inputs if determinable by the user (e.g. in the JACK backend). If the [`AudioBackend`]
    /// provides a native number of inputs that number is chosen instead.
    pub num_inputs: usize,
    /// num outputs if determinable by the user (e.g. in the JACK backend)
    /// If the [`AudioBackend`]
    /// provides a native number of outputs that number is chosen instead.
    pub num_outputs: usize,
    /// The latency added for time scheduled changes to the audio thread to allow enough time for events to take place.
    pub scheduling_latency: Duration,
    /// The capacity of the ring buffer transferring changes to constant inputs to the audio thread.
    pub scheduling_ring_buffer_capacity: usize,
}

impl Default for SphereSettings {
    fn default() -> Self {
        Self {
            name: Default::default(),
            resources_settings: Default::default(),
            scheduling_latency: Duration::from_millis(100),
            num_inputs: 2,
            num_outputs: 2,
            scheduling_ring_buffer_capacity: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KnystSphere, SphereSettings};
    use crate::{
        audio_backend::{AudioBackend, AudioBackendError},
        graph::{Graph, RunGraphSettings},
        modal_interface::SphereError,
        resources::{Resources, ResourcesError, ResourcesSettings},
        KnystError,
    };

    struct DummyBackend;

    impl AudioBackend for DummyBackend {
        fn start_processing_return_controller(
            &mut self,
            _graph: Graph,
            _resources: Resources,
            _run_graph_settings: RunGraphSettings,
            _error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
        ) -> Result<crate::controller::Controller, AudioBackendError> {
            panic!("backend should not start when resources initialization fails");
        }

        fn stop(&mut self) -> Result<(), AudioBackendError> {
            Ok(())
        }

        fn sample_rate(&self) -> usize {
            44_100
        }

        fn block_size(&self) -> Option<usize> {
            Some(64)
        }

        fn native_output_channels(&self) -> Option<usize> {
            Some(2)
        }

        fn native_input_channels(&self) -> Option<usize> {
            Some(0)
        }
    }

    fn failing_resources_settings() -> ResourcesSettings {
        ResourcesSettings {
            max_wavetables: usize::MAX,
            max_buffers: 1,
            max_user_data: 0,
        }
    }

    #[test]
    fn sphere_start_propagates_resources_error() {
        let mut backend = DummyBackend;
        let result = KnystSphere::start(
            &mut backend,
            SphereSettings {
                resources_settings: failing_resources_settings(),
                ..Default::default()
            },
            |_| {},
        );

        match result {
            Err(SphereError::ResourcesError(ResourcesError::InvalidWavetableCapacity(
                usize::MAX,
            ))) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("expected start to fail"),
        }
    }

    #[test]
    fn sphere_start_return_controller_propagates_resources_error() {
        let mut backend = DummyBackend;
        let result = KnystSphere::start_return_controller(
            &mut backend,
            SphereSettings {
                resources_settings: failing_resources_settings(),
                ..Default::default()
            },
            |_| {},
        );

        match result {
            Err(SphereError::ResourcesError(ResourcesError::InvalidWavetableCapacity(
                usize::MAX,
            ))) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("expected start_return_controller to fail"),
        }
    }
}
