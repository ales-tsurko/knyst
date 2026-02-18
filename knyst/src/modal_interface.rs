//! Explicit context API for interacting with Knyst.
//!
//! `KnystContext` is a first-class handle to one running Knyst instance. It can
//! be passed around explicitly by host code and used without any global state.
//!
//! A thread-local active context helper is still provided for APIs that rely on
//! `knyst_commands()` internally (for example many handle operations).

use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::audio_backend::AudioBackendError;
use crate::buffer::Buffer;
use crate::controller::{
    CallbackHandle, KnystCommands, MultiThreadedKnystCommands, StartBeat, UploadGraphError,
};
use crate::graph::connection::{ConnectionBundle, InputBundle};
use crate::graph::{
    Connection, GenOrGraph, GraphId, GraphSettings, NodeId, ObservabilitySnapshot, ParameterChange,
    SimultaneousChanges, Time, TransportSnapshot,
};
use crate::handles::{GraphHandle, Handle};
use crate::inspection::GraphInspection;
use crate::resources::{BufferId, ResourcesError, WavetableId};
use crate::scheduling::MusicalTimeMap;
use crate::time::{Beats, Seconds};
use crate::wavetable_aa::Wavetable;

/// Shared command handle that preserves mutable command state across clones.
#[derive(Clone)]
pub struct SharedKnystCommands {
    commands: Arc<Mutex<MultiThreadedKnystCommands>>,
}

impl SharedKnystCommands {
    /// Create a shared command handle from raw multithreaded commands.
    #[must_use]
    pub fn new(commands: MultiThreadedKnystCommands) -> Self {
        Self {
            commands: Arc::new(Mutex::new(commands)),
        }
    }

    /// Returns a snapshot clone of the underlying raw command handle.
    #[must_use]
    pub fn snapshot(&self) -> MultiThreadedKnystCommands {
        self.lock().clone()
    }

    fn lock(&self) -> MutexGuard<'_, MultiThreadedKnystCommands> {
        self.commands.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl KnystCommands for SharedKnystCommands {
    fn push_without_inputs(&mut self, gen_or_graph: impl GenOrGraph) -> NodeId {
        self.lock().push_without_inputs(gen_or_graph)
    }

    fn push(&mut self, gen_or_graph: impl GenOrGraph, inputs: impl Into<InputBundle>) -> NodeId {
        self.lock().push(gen_or_graph, inputs)
    }

    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeId {
        self.lock()
            .push_to_graph_without_inputs(gen_or_graph, graph_id)
    }

    fn push_to_graph(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
        inputs: impl Into<InputBundle>,
    ) -> NodeId {
        self.lock().push_to_graph(gen_or_graph, graph_id, inputs)
    }

    fn connect(&mut self, connection: Connection) {
        self.lock().connect(connection);
    }

    fn connect_bundle(&mut self, bundle: impl Into<ConnectionBundle>) {
        self.lock().connect_bundle(bundle);
    }

    fn schedule_beat_callback(
        &mut self,
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle {
        self.lock().schedule_beat_callback(callback, start_time)
    }

    fn disconnect(&mut self, connection: Connection) {
        self.lock().disconnect(connection);
    }

    fn set_mortality(&mut self, node: NodeId, is_mortal: bool) {
        self.lock().set_mortality(node, is_mortal);
    }

    fn free_disconnected_nodes(&mut self) {
        self.lock().free_disconnected_nodes();
    }

    fn free_node_mend_connections(&mut self, node: NodeId) {
        self.lock().free_node_mend_connections(node);
    }

    fn free_node(&mut self, node: NodeId) {
        self.lock().free_node(node);
    }

    fn schedule_change(&mut self, change: ParameterChange) {
        self.lock().schedule_change(change);
    }

    fn schedule_changes(&mut self, changes: SimultaneousChanges) {
        self.lock().schedule_changes(changes);
    }

    fn insert_buffer(&mut self, buffer: Buffer) -> BufferId {
        self.lock().insert_buffer(buffer)
    }

    fn remove_buffer(&mut self, buffer_id: BufferId) {
        self.lock().remove_buffer(buffer_id);
    }

    fn replace_buffer(&mut self, buffer_id: BufferId, buffer: Buffer) {
        self.lock().replace_buffer(buffer_id, buffer);
    }

    fn insert_wavetable(&mut self, wavetable: Wavetable) -> WavetableId {
        self.lock().insert_wavetable(wavetable)
    }

    fn remove_wavetable(&mut self, wavetable_id: WavetableId) {
        self.lock().remove_wavetable(wavetable_id);
    }

    fn replace_wavetable(&mut self, id: WavetableId, wavetable: Wavetable) {
        self.lock().replace_wavetable(id, wavetable);
    }

    fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap) + Send + 'static,
    ) {
        self.lock().change_musical_time_map(change_fn);
    }

    fn request_inspection(&mut self) -> std::sync::mpsc::Receiver<GraphInspection> {
        self.lock().request_inspection()
    }

    fn transport_play(&mut self) {
        self.lock().transport_play();
    }

    fn transport_pause(&mut self) {
        self.lock().transport_pause();
    }

    fn transport_seek_to_seconds(&mut self, position: Seconds) {
        self.lock().transport_seek_to_seconds(position);
    }

    fn transport_seek_to_beats(&mut self, position: Beats) {
        self.lock().transport_seek_to_beats(position);
    }

    fn request_transport_snapshot(
        &mut self,
    ) -> std::sync::mpsc::Receiver<Option<TransportSnapshot>> {
        self.lock().request_transport_snapshot()
    }

    fn request_observability_snapshot(
        &mut self,
    ) -> std::sync::mpsc::Receiver<Option<ObservabilitySnapshot>> {
        self.lock().request_observability_snapshot()
    }

    fn default_graph_settings(&self) -> GraphSettings {
        self.lock().default_graph_settings()
    }

    fn to_graph(&mut self, graph_id: GraphId) {
        self.lock().to_graph(graph_id);
    }

    fn to_top_level_graph(&mut self) {
        self.lock().to_top_level_graph();
    }

    fn current_graph(&self) -> GraphId {
        self.lock().current_graph()
    }

    fn init_local_graph(&mut self, settings: GraphSettings) -> GraphId {
        self.lock().init_local_graph(settings)
    }

    fn upload_local_graph(&mut self) -> Option<Handle<GraphHandle>> {
        self.lock().upload_local_graph()
    }

    fn start_scheduling_bundle(&mut self, time: Time) {
        self.lock().start_scheduling_bundle(time);
    }

    fn upload_scheduling_bundle(&mut self) {
        self.lock().upload_scheduling_bundle();
    }
}

/// First-class context handle for one running Knyst instance.
#[derive(Clone)]
pub struct KnystContext {
    commands: SharedKnystCommands,
}

impl KnystContext {
    /// Create a new context from command handle(s) returned by a running controller/backend.
    #[must_use]
    pub fn new(commands: MultiThreadedKnystCommands) -> Self {
        Self {
            commands: SharedKnystCommands::new(commands),
        }
    }

    /// Return a snapshot clone of the command handle for this context.
    ///
    /// To preserve mutable command state across clones, use [`knyst_commands`]
    /// or call [`KnystContext::shared_commands`].
    #[must_use]
    pub fn commands(&self) -> MultiThreadedKnystCommands {
        self.commands.snapshot()
    }

    /// Return a shared command handle for this context.
    #[must_use]
    pub fn shared_commands(&self) -> SharedKnystCommands {
        self.commands.clone()
    }

    /// Set this context as active for the current thread.
    pub fn activate(&self) {
        set_active_context(self.clone());
    }

    /// Set this context as active for the current thread and return a guard
    /// that restores the previously active context on drop.
    #[must_use]
    pub fn activate_scoped(&self) -> KnystContextActivationGuard {
        let previous = ACTIVE_KNYST_CONTEXT.with(|active| active.borrow().clone());
        set_active_context(self.clone());
        KnystContextActivationGuard { previous }
    }

    /// Temporarily activate this context for the current thread while running `f`.
    pub fn with_activation<R>(&self, f: impl FnOnce() -> R) -> R {
        let _guard = self.activate_scoped();
        f()
    }

    /// Create a new local graph, run `init`, and upload it to this context.
    pub fn upload_graph(
        &self,
        settings: GraphSettings,
        init: impl FnOnce(),
    ) -> Result<Handle<GraphHandle>, UploadGraphError> {
        let mut commands = self.shared_commands();
        commands.init_local_graph(settings);
        init();
        commands
            .upload_local_graph()
            .ok_or(UploadGraphError::LocalGraphMissing)
    }

    /// Run `c` while collecting scheduled changes, then upload as one bundle.
    pub fn schedule_bundle(&self, time: Time, c: impl FnOnce()) {
        let mut commands = self.shared_commands();
        commands.start_scheduling_bundle(time);
        c();
        commands.upload_scheduling_bundle();
    }
}

thread_local! {
    static ACTIVE_KNYST_CONTEXT: RefCell<Option<KnystContext>> = const { RefCell::new(None) };
}

/// RAII guard that restores the previous active context when dropped.
pub struct KnystContextActivationGuard {
    previous: Option<KnystContext>,
}

impl Drop for KnystContextActivationGuard {
    fn drop(&mut self) {
        ACTIVE_KNYST_CONTEXT.with(|active| {
            *active.borrow_mut() = self.previous.clone();
        });
    }
}

/// Set the active context for the current thread.
///
/// Prefer [`KnystContext::activate_scoped`] or [`KnystContext::with_activation`]
/// to avoid leaking thread-local state outside an intended scope.
pub fn set_active_context(context: KnystContext) {
    ACTIVE_KNYST_CONTEXT.with(|active| {
        *active.borrow_mut() = Some(context);
    });
}

/// Clear the active context for the current thread.
pub fn clear_active_context() {
    ACTIVE_KNYST_CONTEXT.with(|active| {
        *active.borrow_mut() = None;
    });
}

/// Return the active commands for this thread, if any.
#[must_use]
pub fn try_knyst_commands() -> Option<SharedKnystCommands> {
    ACTIVE_KNYST_CONTEXT.with(|active| active.borrow().as_ref().map(KnystContext::shared_commands))
}

/// Returns active commands for this thread.
///
/// Panics if no context has been activated on this thread.
/// Prefer using [`KnystContext::with_activation`] to ensure calls happen under
/// an explicit scoped activation.
pub fn knyst_commands() -> SharedKnystCommands {
    try_knyst_commands().expect(
        "No active KnystContext on this thread. Use KnystContext::activate() or pass commands explicitly.",
    )
}

/// Error type for higher-level sphere startup failures.
#[derive(thiserror::Error, Debug)]
pub enum SphereError {
    /// There was an error in the audio backend
    #[error("Audio backend error: {0}")]
    AudioBackendError(#[from] AudioBackendError),
    /// There was an error initialising resources
    #[error("Resources error: {0}")]
    ResourcesError(#[from] ResourcesError),
}

#[cfg(test)]
mod tests {
    use core::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use super::{clear_active_context, knyst_commands, try_knyst_commands};
    use crate::audio_backend::{AudioBackend, AudioBackendError};
    use crate::controller::{print_error_handler, Controller, KnystCommands};
    use crate::graph::{Graph, RunGraph, RunGraphSettings};
    use crate::prelude::{KnystSphere, SphereSettings};
    use crate::{KnystError, Resources};

    struct TestBackend {
        sample_rate: usize,
        block_size: usize,
        num_outputs: usize,
        num_inputs: usize,
        run_graph: Option<RunGraph>,
    }

    impl AudioBackend for TestBackend {
        fn start_processing_return_controller(
            &mut self,
            mut graph: Graph,
            resources: Resources,
            run_graph_settings: RunGraphSettings,
            error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
        ) -> Result<Controller, AudioBackendError> {
            if self.run_graph.is_some() {
                return Err(AudioBackendError::BackendAlreadyRunning);
            }
            let (run_graph, resources_command_sender, resources_command_receiver) =
                RunGraph::new(&mut graph, resources, run_graph_settings)?;
            self.run_graph = Some(run_graph);
            Ok(Controller::new(
                graph,
                error_handler,
                resources_command_sender,
                resources_command_receiver,
            ))
        }

        fn stop(&mut self) -> Result<(), AudioBackendError> {
            if self.run_graph.take().is_some() {
                Ok(())
            } else {
                Err(AudioBackendError::BackendNotRunning)
            }
        }

        fn sample_rate(&self) -> usize {
            self.sample_rate
        }

        fn block_size(&self) -> Option<usize> {
            Some(self.block_size)
        }

        fn native_output_channels(&self) -> Option<usize> {
            Some(self.num_outputs)
        }

        fn native_input_channels(&self) -> Option<usize> {
            Some(self.num_inputs)
        }
    }

    #[test]
    fn active_context_switches_command_target() {
        let mut backend_a = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };
        let mut backend_b = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };

        let sphere_a = KnystSphere::start(
            &mut backend_a,
            SphereSettings::default(),
            print_error_handler,
        )
        .expect("sphere A should start");
        let graph_a = sphere_a.commands().current_graph();

        let sphere_b = KnystSphere::start(
            &mut backend_b,
            SphereSettings::default(),
            print_error_handler,
        )
        .expect("sphere B should start");
        let graph_b = sphere_b.commands().current_graph();
        assert_ne!(graph_a, graph_b);

        sphere_a.context().activate();
        assert_eq!(knyst_commands().current_graph(), graph_a);

        sphere_b.context().activate();
        assert_eq!(knyst_commands().current_graph(), graph_b);
    }

    #[test]
    fn with_activation_restores_previous_context() {
        let mut backend_a = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };
        let mut backend_b = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };

        let sphere_a = KnystSphere::start(
            &mut backend_a,
            SphereSettings::default(),
            print_error_handler,
        )
        .expect("sphere A should start");
        let graph_a = sphere_a.commands().current_graph();

        let sphere_b = KnystSphere::start(
            &mut backend_b,
            SphereSettings::default(),
            print_error_handler,
        )
        .expect("sphere B should start");
        let graph_b = sphere_b.commands().current_graph();

        sphere_a.context().activate();
        assert_eq!(knyst_commands().current_graph(), graph_a);

        sphere_b.context().with_activation(|| {
            assert_eq!(knyst_commands().current_graph(), graph_b);
        });

        assert_eq!(knyst_commands().current_graph(), graph_a);
        clear_active_context();
    }

    #[test]
    fn with_activation_restores_previous_context_after_panic() {
        let mut backend_a = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };
        let mut backend_b = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };

        let sphere_a = KnystSphere::start(
            &mut backend_a,
            SphereSettings::default(),
            print_error_handler,
        )
        .expect("sphere A should start");
        let graph_a = sphere_a.commands().current_graph();

        let sphere_b = KnystSphere::start(
            &mut backend_b,
            SphereSettings::default(),
            print_error_handler,
        )
        .expect("sphere B should start");

        sphere_a.context().activate();
        assert_eq!(knyst_commands().current_graph(), graph_a);

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            sphere_b.context().with_activation(|| {
                panic!("expected panic in test");
            });
        }));
        assert!(panic_result.is_err());
        assert_eq!(knyst_commands().current_graph(), graph_a);
        clear_active_context();
    }

    #[test]
    fn active_context_is_thread_local() {
        let mut backend = TestBackend {
            sample_rate: 44_100,
            block_size: 64,
            num_outputs: 2,
            num_inputs: 0,
            run_graph: None,
        };

        let sphere =
            KnystSphere::start(&mut backend, SphereSettings::default(), print_error_handler)
                .expect("sphere should start");
        let graph = sphere.commands().current_graph();
        let context = sphere.context();
        context.activate();
        assert_eq!(knyst_commands().current_graph(), graph);

        let worker_context = context.clone();
        let worker = std::thread::spawn(move || {
            assert!(
                try_knyst_commands().is_none(),
                "worker thread should not inherit active context"
            );
            worker_context.with_activation(|| {
                assert_eq!(knyst_commands().current_graph(), graph);
            });
            assert!(
                try_knyst_commands().is_none(),
                "worker thread should restore previous (none) context"
            );
        });
        worker.join().expect("worker thread should complete");

        assert_eq!(knyst_commands().current_graph(), graph);
        clear_active_context();
    }
}
