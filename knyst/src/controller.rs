//! API for interacting with a running top level [`Graph`] from any number of
//! threads without having to manually keep track of running [`Graph::update`]
//! regularly.
//!
//! [`KnystCommands`] gives you a convenient API for sending messages to the
//! [`Controller`]. The API is similar to calling methods on [`Graph`] directly,
//! but also includes modifying [`Resources`].

#[allow(unused)]
use crate::resources::Resources;
use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, AtomicU64},
        Arc,
    },
    time::{Duration, Instant},
};

use crate::{
    buffer::Buffer,
    graph::{
        EventChange, NodeChanges, ObservabilitySnapshot, ScheduleError, Time, TransportSnapshot,
    },
    inspection::GraphInspection,
    knyst_commands,
    resources::{BufferId, ResourcesCommand, ResourcesResponse, WavetableId},
    time::Seconds,
    wavetable_aa::Wavetable,
};
use crate::{
    graph::{
        connection::{ConnectionBundle, ConnectionError, InputBundle},
        Connection, FreeError, GenOrGraph, GenOrGraphEnum, Graph, GraphId, GraphSettings, NodeId,
        ParameterChange, SharedTransportSnapshotState, SimultaneousChanges,
    },
    handles::{GraphHandle, Handle},
    inputs,
    scheduling::MusicalTimeMap,
    time::Beats,
    KnystError,
};
use audio_thread_priority::promote_current_thread_to_real_time;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

const CONTROLLER_ACTIVE_SLEEP: Duration = Duration::from_micros(100);
const CONTROLLER_IDLE_SLEEP: Duration = Duration::from_millis(1);

/// Encodes commands sent from a [`KnystCommands`]
enum Command {
    Push {
        gen_or_graph: GenOrGraphEnum,
        inputs: InputBundle,
        node_address: NodeId,
        graph_id: GraphId,
        start_time: Time,
    },
    Connect(Connection),
    Disconnect(Connection),
    SetMortality {
        node: NodeId,
        is_mortal: bool,
    },
    FreeNode(NodeId),
    FreeNodeMendConnections(NodeId),
    ScheduleChange(ParameterChange),
    ScheduleEvent(EventChange),
    ScheduleChanges(SimultaneousChanges),
    ClearScheduledChanges,
    FreeDisconnectedNodes,
    ResourcesCommand(ResourcesCommand),
    ChangeMusicalTimeMap(Box<dyn FnOnce(&mut MusicalTimeMap) + Send>),
    ScheduleBeatCallback(BeatCallback, StartBeat),
    RequestInspection(Sender<GraphInspection>),
    RequestGraphSettled(Sender<()>),
    RequestTransportSettled(Sender<()>),
    RequestTransportSnapshot(Sender<Option<TransportSnapshot>>),
    RequestObservabilitySnapshot(Sender<Option<ObservabilitySnapshot>>),
    TransportPlay,
    TransportPause,
    TransportSeekSeconds(Seconds),
    TransportSeekBeats(Beats),
    ReportDropouts(u64),
    ReportError(KnystError),
}
impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Push {
                gen_or_graph,
                inputs: _,
                node_address,
                graph_id,
                start_time,
            } => f
                .debug_struct("Push")
                .field("gen_or_graph", gen_or_graph)
                .field("has_inputs", &true)
                .field("node_address", node_address)
                .field("graph_id", graph_id)
                .field("start_time", start_time)
                .finish(),
            Self::Connect(arg0) => f.debug_tuple("Connect").field(arg0).finish(),
            Self::Disconnect(arg0) => f.debug_tuple("Disconnect").field(arg0).finish(),
            Self::FreeNode(arg0) => f.debug_tuple("FreeNode").field(arg0).finish(),
            Self::FreeNodeMendConnections(arg0) => f
                .debug_tuple("FreeNodeMendConnections")
                .field(arg0)
                .finish(),
            Self::ScheduleChange(arg0) => f.debug_tuple("ScheduleChange").field(arg0).finish(),
            Self::ScheduleEvent(arg0) => f.debug_tuple("ScheduleEvent").field(arg0).finish(),
            Self::ScheduleChanges(arg0) => f.debug_tuple("ScheduleChanges").field(arg0).finish(),
            Self::ClearScheduledChanges => write!(f, "ClearScheduledChanges"),
            Self::FreeDisconnectedNodes => write!(f, "FreeDisconnectedNodes"),
            Self::ResourcesCommand(_arg0) => f.debug_tuple("ResourcesCommand").finish(),
            Self::ChangeMusicalTimeMap(_arg0) => f.debug_tuple("ChangeMusicalTimeMap").finish(),
            Self::ScheduleBeatCallback(_arg0, _arg1) => {
                f.debug_tuple("ScheduleBeatCallback").finish()
            }
            Self::RequestInspection(arg0) => {
                f.debug_tuple("RequestInspection").field(arg0).finish()
            }
            Self::RequestGraphSettled(arg0) => {
                f.debug_tuple("RequestGraphSettled").field(arg0).finish()
            }
            Self::RequestTransportSettled(arg0) => f
                .debug_tuple("RequestTransportSettled")
                .field(arg0)
                .finish(),
            Self::RequestTransportSnapshot(arg0) => f
                .debug_tuple("RequestTransportSnapshot")
                .field(arg0)
                .finish(),
            Self::RequestObservabilitySnapshot(arg0) => f
                .debug_tuple("RequestObservabilitySnapshot")
                .field(arg0)
                .finish(),
            Self::TransportPlay => write!(f, "TransportPlay"),
            Self::TransportPause => write!(f, "TransportPause"),
            Self::TransportSeekSeconds(arg0) => {
                f.debug_tuple("TransportSeekSeconds").field(arg0).finish()
            }
            Self::TransportSeekBeats(arg0) => {
                f.debug_tuple("TransportSeekBeats").field(arg0).finish()
            }
            Self::ReportDropouts(arg0) => f.debug_tuple("ReportDropouts").field(arg0).finish(),
            Self::ReportError(arg0) => f.debug_tuple("ReportError").field(arg0).finish(),
            Command::SetMortality { node, is_mortal } => f
                .debug_tuple("SetMortality")
                .field(node)
                .field(is_mortal)
                .finish(),
        }
    }
}

/// Error from sending controller-related commands.
#[derive(thiserror::Error, Debug)]
pub enum ControllerError {
    /// Sending a command to the controller failed because the channel is closed.
    #[error("Failed to send command to controller because the command channel is closed.")]
    CommandChannelClosed,
    /// Sending a graph inspection response failed because the receiver dropped.
    #[error("Failed to send graph inspection response because the receiver was dropped.")]
    InspectionResponseChannelClosed,
    /// Sending a graph-settled response failed because the receiver dropped.
    #[error("Failed to send graph-settled response because the receiver was dropped.")]
    GraphSettledResponseChannelClosed,
    /// Sending a transport snapshot response failed because the receiver dropped.
    #[error("Failed to send transport snapshot response because the receiver was dropped.")]
    TransportSnapshotResponseChannelClosed,
    /// Sending a transport-settled response failed because the receiver dropped.
    #[error("Failed to send transport-settled response because the receiver was dropped.")]
    TransportSettledResponseChannelClosed,
    /// Sending an observability snapshot response failed because the receiver dropped.
    #[error("Failed to send observability snapshot response because the receiver was dropped.")]
    ObservabilitySnapshotResponseChannelClosed,
}

/// [`KnystCommands`] sends commands to the [`Controller`] which should hold the
/// top level [`Graph`]. The API is as close as possible to that of an owned
/// [`Graph`].
///
/// This can safely be cloned and sent to a different thread for use.
///
// TODO: What's the best way of referring to a graph? GraphId is unique, but not
// always the handiest. It would be nice to be able to choose to refer to Graphs
// by an identifier e.g. name. In Bevy holding on to GraphIds is easy.
pub trait KnystCommands {
    /// Push a Gen or Graph to the top level Graph without specifying any inputs.
    fn push_without_inputs(&mut self, gen_or_graph: impl GenOrGraph) -> NodeId;
    /// Push a Gen or Graph to the default Graph.
    fn push(&mut self, gen_or_graph: impl GenOrGraph, inputs: impl Into<InputBundle>) -> NodeId;
    /// Push a Gen or Graph to the Graph with the specified id without specifying inputs.
    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeId;
    /// Push a Gen or Graph to the Graph with the specified id.
    fn push_to_graph(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
        inputs: impl Into<InputBundle>,
    ) -> NodeId;
    /// Create a new connections
    fn connect(&mut self, connection: Connection);
    /// Make several connections at once using any of the ConnectionBundle
    /// notations
    fn connect_bundle(&mut self, bundle: impl Into<ConnectionBundle>);
    /// Add a new beat callback. See [`BeatCallback`] for documentation.
    fn schedule_beat_callback(
        &mut self,
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle;
    /// Disconnect (undo) a [`Connection`]
    fn disconnect(&mut self, connection: Connection);
    /// Sets the mortality of a node to mortal (true) or immortal (false). An immortal node cannot be freed.
    fn set_mortality(&mut self, node: NodeId, is_mortal: bool);
    /// Free any nodes that are not currently connected to the graph's outputs
    /// via any chain of connections.
    fn free_disconnected_nodes(&mut self);
    /// Free a node and try to mend connections between the inputs and the
    /// outputs of the node.
    fn free_node_mend_connections(&mut self, node: NodeId);
    /// Free a node.
    fn free_node(&mut self, node: NodeId);
    /// Schedule a change to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_change(&mut self, change: ParameterChange);
    /// Schedule a block-local event to be delivered to a node.
    fn schedule_event(&mut self, event: EventChange);
    /// Schedule multiple changes to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_changes(&mut self, changes: SimultaneousChanges);
    /// Clear any pending scheduled changes that have not yet been sent to the audio thread.
    fn clear_scheduled_changes(&mut self);
    /// Inserts a new buffer in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_buffer(&mut self, buffer: Buffer) -> BufferId;
    /// Remove a buffer from the [`Resources`]
    fn remove_buffer(&mut self, buffer_id: BufferId);
    /// Replace a buffer in the [`Resources`]
    fn replace_buffer(&mut self, buffer_id: BufferId, buffer: Buffer);
    /// Inserts a new wavetable in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_wavetable(&mut self, wavetable: Wavetable) -> WavetableId;
    /// Remove a wavetable from the [`Resources`]
    fn remove_wavetable(&mut self, wavetable_id: WavetableId);
    /// Replace a wavetable in the [`Resources`]
    fn replace_wavetable(&mut self, id: WavetableId, wavetable: Wavetable);
    /// Make a change to the shared [`MusicalTimeMap`]
    fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap) + Send + 'static,
    );
    /// Request a [`GraphInspection`] of the top level graph which will be sent back in the returned channel
    fn request_inspection(&mut self) -> Receiver<GraphInspection>;
    /// Request a notification when graph topology/task updates submitted before this
    /// call have been applied on the audio thread.
    fn request_graph_settled(&mut self) -> Receiver<()>;
    /// Request a notification when transport updates submitted before this call
    /// have been applied on the audio thread.
    fn request_transport_settled(&mut self) -> Receiver<()>;
    /// Start transport playback.
    fn transport_play(&mut self);
    /// Pause transport playback.
    fn transport_pause(&mut self);
    /// Seek transport to an absolute seconds position.
    fn transport_seek_to_seconds(&mut self, position: Seconds);
    /// Seek transport to an absolute beats position.
    fn transport_seek_to_beats(&mut self, position: Beats);
    /// Request current transport state and position.
    fn request_transport_snapshot(&mut self) -> Receiver<Option<TransportSnapshot>>;
    /// Return the current transport snapshot without a controller roundtrip.
    fn current_transport_snapshot(&self) -> Option<TransportSnapshot>;
    /// Request current runtime observability metrics.
    fn request_observability_snapshot(&mut self) -> Receiver<Option<ObservabilitySnapshot>>;

    /// Return the [`GraphSettings`] of the top level graph. This means you
    /// don't have to manually keep track of matching sample rate and block size
    /// for example.
    fn default_graph_settings(&self) -> GraphSettings;
    /// Set knyst commands on the current thread to use the selected GraphId by default
    fn to_graph(&mut self, graph_id: GraphId);
    /// Set knyst commands on the current thread to use the top level GraphId by default
    fn to_top_level_graph(&mut self);
    /// Get the id of the currently active graph
    fn current_graph(&self) -> GraphId;
    /// Creates a new local graph and sets it as the default graph
    fn init_local_graph(&mut self, settings: GraphSettings) -> GraphId;
    /// Upload the local graph to the previously default graph and restore the default graph to that previous default graph.
    fn upload_local_graph(&mut self)
        -> Option<crate::handles::Handle<crate::handles::GraphHandle>>;
    /// Start a scheduling bundle, meaning any change scheduled will not be applied until [`KnystCommands::upload_scheduling_bundle`] is called. Prefer using [`schedule_bundle`] as it is more difficult to misuse.
    fn start_scheduling_bundle(&mut self, time: Time);
    /// Uploads scheduled changes to the graph and schedules them for the time specified in [`KnystCommands::start_scheduling_bundle`]. Prefer [`schedule_bundle`] to help reinforce scoping and potential thread switches.
    fn upload_scheduling_bundle(&mut self);
}

/// Error from using [`upload_graph`].
#[derive(thiserror::Error, Debug)]
pub enum UploadGraphError {
    /// No local graph was available to upload.
    #[error("No local graph was available to upload.")]
    LocalGraphMissing,
}

/// Create a new local graph, runs the init function to let you build it, and then uploads it to the active Sphere.
pub fn upload_graph(
    settings: GraphSettings,
    init: impl FnOnce(),
) -> Result<crate::handles::Handle<crate::handles::GraphHandle>, UploadGraphError> {
    knyst_commands().init_local_graph(settings);
    init();
    knyst_commands()
        .upload_local_graph()
        .ok_or(UploadGraphError::LocalGraphMissing)
}

/// Schedules any changes made in the closure at the given time. Currently limited to changes of constant values and spawning new nodes, not new connections.
pub fn schedule_bundle(time: Time, c: impl FnOnce()) {
    knyst_commands().start_scheduling_bundle(time);
    c();
    knyst_commands().upload_scheduling_bundle();
}

#[derive(Clone)]
/// Multi threaded implementation on KnystCommands, default
pub struct MultiThreadedKnystCommands {
    /// Sends Commands to the Controller.
    sender: crossbeam_channel::Sender<Command>,
    /// As pushing to the top level Graph is the default we store the GraphId to that Graph.
    top_level_graph_id: GraphId,
    /// Make the top level graph settings available so that creating a matching sub graph is easy.
    top_level_graph_settings: GraphSettings,
    /// The default graph to push new nodes to
    selected_graph_remote_graph: GraphId,
    /// If changes should be bundled
    bundle_changes: bool,
    /// The vec holding changes to be later scheduled as a bundle
    changes_bundle: Vec<NodeChanges>,
    changes_bundle_time: Time,
    transport_snapshot_state: Arc<SharedTransportSnapshotState>,
}

impl MultiThreadedKnystCommands {
    fn send_command(&self, command: Command) -> Result<(), ControllerError> {
        self.sender
            .send(command)
            .map_err(|_| ControllerError::CommandChannelClosed)
    }

    /// Best-effort error reporting to the controller error handler.
    pub(crate) fn report_error(&self, error: impl Into<KnystError>) {
        let _ = self.send_command(Command::ReportError(error.into()));
    }

    /// Best-effort reporting of backend dropouts/xruns.
    pub(crate) fn report_dropouts(&self, count: u64) {
        let _ = self.send_command(Command::ReportDropouts(count));
    }

    pub(crate) fn shared_transport_snapshot_state(&self) -> Arc<SharedTransportSnapshotState> {
        self.transport_snapshot_state.clone()
    }
}

impl KnystCommands for MultiThreadedKnystCommands {
    /// Push a Gen or Graph to the top level Graph without specifying any inputs.
    fn push_without_inputs(&mut self, gen_or_graph: impl GenOrGraph) -> NodeId {
        self.push(gen_or_graph, inputs![])
    }
    /// Push a Gen or Graph to the default Graph.
    fn push(&mut self, gen_or_graph: impl GenOrGraph, inputs: impl Into<InputBundle>) -> NodeId {
        let inputs: InputBundle = inputs.into();
        let local_graph_id = LOCAL_GRAPH.with_borrow(|g| g.last().map(|g| g.id()));
        if let Some(local_graph_id) = local_graph_id {
            let node_id = LOCAL_GRAPH.with_borrow_mut(|g| {
                let g = g.last_mut().expect("local graph should still exist");
                let mut node_id = NodeId::new(local_graph_id);
                g.push_with_existing_address_at_time(
                    gen_or_graph,
                    &mut node_id,
                    self.changes_bundle_time,
                );
                node_id
            });
            self.connect_bundle(inputs.to(node_id));
            node_id
        } else {
            let new_node_address = NodeId::new(self.selected_graph_remote_graph);
            let command = Command::Push {
                gen_or_graph: gen_or_graph.into_gen_or_graph_enum(),
                inputs,
                node_address: new_node_address,
                graph_id: self.selected_graph_remote_graph,
                start_time: self.changes_bundle_time,
            };
            if let Err(error) = self.send_command(command) {
                self.report_error(error);
            }
            new_node_address
        }
    }
    /// Push a Gen or Graph to the Graph with the specified id without specifying inputs.
    fn push_to_graph_without_inputs(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
    ) -> NodeId {
        let gen_or_graph = gen_or_graph.into_gen_or_graph_enum();
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                if g.id() == graph_id {
                    let mut node_id = NodeId::new(graph_id);
                    if let Err(e) =
                        g.push_with_existing_address_to_graph(gen_or_graph, &mut node_id, g.id())
                    {
                        self.report_error(e);
                    }
                    Ok(node_id)
                } else {
                    Err(gen_or_graph)
                }
            } else {
                // There is no local graph
                Err(gen_or_graph)
            }
        });
        match found_in_local {
            Ok(node_id) => node_id,
            Err(gen_or_graph) => {
                let new_node_address = NodeId::new(graph_id);
                let command = Command::Push {
                    gen_or_graph,
                    inputs: inputs![],
                    node_address: new_node_address,
                    graph_id,
                    start_time: self.changes_bundle_time,
                };
                if let Err(error) = self.send_command(command) {
                    self.report_error(error);
                }
                new_node_address
            }
        }
    }
    /// Push a Gen or Graph to the Graph with the specified id.
    fn push_to_graph(
        &mut self,
        gen_or_graph: impl GenOrGraph,
        graph_id: GraphId,
        inputs: impl Into<InputBundle>,
    ) -> NodeId {
        let inputs: InputBundle = inputs.into();
        let local_graph_matches =
            LOCAL_GRAPH.with_borrow(|g| g.last().is_some_and(|g| g.id() == graph_id));
        if local_graph_matches {
            let gen_or_graph = gen_or_graph.into_gen_or_graph_enum();
            let node_id = LOCAL_GRAPH.with_borrow_mut(|g| {
                let g = g.last_mut().expect("local graph should still exist");
                let mut node_id = NodeId::new(graph_id);
                if let Err(e) =
                    g.push_with_existing_address_to_graph(gen_or_graph, &mut node_id, g.id())
                {
                    self.report_error(e);
                }
                node_id
            });
            self.connect_bundle(inputs.to(node_id));
            node_id
        } else {
            let new_node_address = NodeId::new(graph_id);
            let command = Command::Push {
                gen_or_graph: gen_or_graph.into_gen_or_graph_enum(),
                inputs,
                node_address: new_node_address,
                graph_id,
                start_time: self.changes_bundle_time,
            };
            if let Err(error) = self.send_command(command) {
                self.report_error(error);
            }
            new_node_address
        }
    }
    /// Create a new connections
    fn connect(&mut self, connection: Connection) {
        // The connection may be in our local graph or remotely. Check local first.
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                match g.connect(connection.clone()) {
                    Ok(()) => true,
                    Err(e) => match e {
                        ConnectionError::GraphNotFound(_) => false,
                        _ => {
                            self.report_error(e);
                            // We found the correct graph, but there was a different error
                            true
                        }
                    },
                }
            } else {
                false
            }
        });
        if !found_in_local {
            if let Err(error) = self.send_command(Command::Connect(connection)) {
                self.report_error(error);
            }
        }
    }
    /// Make several connections at once using any of the ConnectionBundle
    /// notations
    fn connect_bundle(&mut self, bundle: impl Into<ConnectionBundle>) {
        let bundle = bundle.into();
        for c in bundle.as_connections() {
            self.connect(c);
        }
    }
    /// Add a new beat callback. See [`BeatCallback`] for documentation.
    fn schedule_beat_callback(
        &mut self,
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: StartBeat,
    ) -> CallbackHandle {
        let c = BeatCallback::new(callback, Beats::ZERO);
        let handle = c.handle();
        let command = Command::ScheduleBeatCallback(c, start_time);
        if let Err(error) = self.send_command(command) {
            self.report_error(error);
        }
        handle
    }
    /// Disconnect (undo) a [`Connection`]
    fn disconnect(&mut self, connection: Connection) {
        // The connection may be in our local graph or remotely. Check local first.
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                match g.disconnect(connection.clone()) {
                    Ok(()) => true,
                    Err(e) => match e {
                        ConnectionError::GraphNotFound(_) => false,
                        _ => {
                            self.report_error(e);
                            // We found the correct graph, but there was a different error
                            true
                        }
                    },
                }
            } else {
                false
            }
        });
        if !found_in_local {
            if let Err(error) = self.send_command(Command::Disconnect(connection)) {
                self.report_error(error);
            }
        }
    }
    /// Free any nodes that are not currently connected to the graph's outputs
    /// via any chain of connections.
    fn free_disconnected_nodes(&mut self) {
        if let Err(error) = self.send_command(Command::FreeDisconnectedNodes) {
            self.report_error(error);
        }
    }
    /// Free a node and try to mend connections between the inputs and the
    /// outputs of the node.
    fn free_node_mend_connections(&mut self, node: NodeId) {
        if let Err(error) = self.send_command(Command::FreeNodeMendConnections(node)) {
            self.report_error(error);
        }
    }
    /// Free a node.
    fn free_node(&mut self, node: NodeId) {
        if let Err(error) = self.send_command(Command::FreeNode(node)) {
            self.report_error(error);
        }
    }
    /// Schedule a change to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_change(&mut self, change: ParameterChange) {
        if self.bundle_changes {
            let change = NodeChanges {
                node: change.input.node,
                parameters: vec![(change.input.channel, change.value)],
                offset: None,
            };
            self.changes_bundle.push(change);
        } else {
            LOCAL_GRAPH.with_borrow_mut(|g| {
                if let Some(g) = g.last_mut() {
                    if let Err(e) = g.schedule_change(change) {
                        self.report_error(e);
                    }
                } else {
                    // There is no local graph
                    if let Err(error) = self.send_command(Command::ScheduleChange(change)) {
                        self.report_error(error);
                    }
                }
            });
        }
    }
    fn schedule_event(&mut self, event: EventChange) {
        LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                if let Err(e) = g.schedule_event(event.clone()) {
                    self.report_error(e);
                }
            } else if let Err(error) = self.send_command(Command::ScheduleEvent(event)) {
                self.report_error(error);
            }
        });
    }
    /// Schedule multiple changes to be made.
    ///
    /// NB: Changes are buffered and the scheduler needs to be regularly updated
    /// for them to be sent to the audio thread. If you are getting your
    /// [`KnystCommands`] through `AudioBackend::start_processing` this is taken
    /// care of automatically.
    fn schedule_changes(&mut self, changes: SimultaneousChanges) {
        if self.bundle_changes {
            self.changes_bundle.extend(changes.changes);
        } else {
            let mut all_node_graphs = vec![];
            let time = changes.time;
            for c in &changes.changes {
                if !all_node_graphs.contains(&c.node.graph_id()) {
                    all_node_graphs.push(c.node.graph_id());
                }
            }
            let change_bundles_per_graph = if all_node_graphs.len() < 2 {
                vec![changes.changes]
            } else {
                let mut per_graph = vec![vec![]; all_node_graphs.len()];
                for change in changes.changes {
                    let i = all_node_graphs
                        .iter()
                        .position(|graph| *graph == change.node.graph_id())
                        .unwrap();
                    per_graph[i].push(change);
                }
                per_graph
            };
            for changes in change_bundles_per_graph {
                LOCAL_GRAPH.with_borrow_mut(|g| {
                    if let Some(g) = g.last_mut() {
                        if let Err(e) = g.schedule_changes(changes, time) {
                            self.report_error(e);
                        }
                    } else {
                        // There is no local graph
                        if let Err(error) =
                            self.send_command(Command::ScheduleChanges(SimultaneousChanges {
                                time,
                                changes,
                            }))
                        {
                            self.report_error(error);
                        }
                    }
                });
            }
        }
    }
    fn clear_scheduled_changes(&mut self) {
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                if let Err(error) = g.clear_scheduled_changes() {
                    self.report_error(error);
                }
                true
            } else {
                false
            }
        });
        if !found_in_local {
            if let Err(error) = self.send_command(Command::ClearScheduledChanges) {
                self.report_error(error);
            }
        }
    }
    /// Inserts a new buffer in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_buffer(&mut self, buffer: Buffer) -> BufferId {
        let id = BufferId::new(&buffer);
        if let Err(error) =
            self.send_command(Command::ResourcesCommand(ResourcesCommand::InsertBuffer {
                id,
                buffer,
            }))
        {
            self.report_error(error);
        }
        id
    }
    /// Remove a buffer from the [`Resources`]
    fn remove_buffer(&mut self, buffer_id: BufferId) {
        if let Err(error) =
            self.send_command(Command::ResourcesCommand(ResourcesCommand::RemoveBuffer {
                id: buffer_id,
            }))
        {
            self.report_error(error);
        }
    }
    /// Replace a buffer in the [`Resources`]
    fn replace_buffer(&mut self, buffer_id: BufferId, buffer: Buffer) {
        if let Err(error) =
            self.send_command(Command::ResourcesCommand(ResourcesCommand::ReplaceBuffer {
                id: buffer_id,
                buffer,
            }))
        {
            self.report_error(error);
        }
    }
    /// Inserts a new wavetable in the [`Resources`] and returns an id which can be
    /// converted to a key on the audio thread with access to a [`Resources`].
    fn insert_wavetable(&mut self, wavetable: Wavetable) -> WavetableId {
        let id = WavetableId::new();
        if let Err(error) = self.send_command(Command::ResourcesCommand(
            ResourcesCommand::InsertWavetable { id, wavetable },
        )) {
            self.report_error(error);
        }
        id
    }
    /// Remove a wavetable from the [`Resources`]
    fn remove_wavetable(&mut self, wavetable_id: WavetableId) {
        if let Err(error) = self.send_command(Command::ResourcesCommand(
            ResourcesCommand::RemoveWavetable { id: wavetable_id },
        )) {
            self.report_error(error);
        }
    }
    /// Replace a wavetable in the [`Resources`]
    fn replace_wavetable(&mut self, id: WavetableId, wavetable: Wavetable) {
        if let Err(error) = self.send_command(Command::ResourcesCommand(
            ResourcesCommand::ReplaceWavetable { id, wavetable },
        )) {
            self.report_error(error);
        }
    }
    /// Make a change to the shared [`MusicalTimeMap`]
    fn change_musical_time_map(
        &mut self,
        change_fn: impl FnOnce(&mut MusicalTimeMap) + Send + 'static,
    ) {
        if let Err(error) = self.send_command(Command::ChangeMusicalTimeMap(Box::new(change_fn))) {
            self.report_error(error);
        }
    }
    /// Return the [`GraphSettings`] of the top level graph. This means you
    /// don't have to manually keep track of matching sample rate and block size
    /// for example.
    fn default_graph_settings(&self) -> GraphSettings {
        self.top_level_graph_settings.clone()
    }

    fn init_local_graph(&mut self, settings: GraphSettings) -> GraphId {
        let graph = Graph::new(settings);
        let graph_id = graph.id();
        LOCAL_GRAPH.with_borrow_mut(|g| g.push(graph));
        graph_id
    }

    fn upload_local_graph(&mut self) -> Option<Handle<GraphHandle>> {
        let graph_to_upload = LOCAL_GRAPH.with_borrow_mut(|g| g.pop());
        if let Some(g) = graph_to_upload {
            let num_inputs = g.num_inputs();
            let num_outputs = g.num_outputs();
            let graph_id = g.id();

            let id = self.push_without_inputs(g);
            Some(Handle::new(GraphHandle::new(
                id,
                graph_id,
                num_inputs,
                num_outputs,
            )))
        } else {
            None
        }
    }

    fn request_inspection(&mut self) -> Receiver<GraphInspection> {
        let (sender, receiver) = bounded(1);
        if let Err(error) = self.send_command(Command::RequestInspection(sender)) {
            self.report_error(error);
        }
        receiver
    }

    fn request_graph_settled(&mut self) -> Receiver<()> {
        let (sender, receiver) = bounded(1);
        if let Err(error) = self.send_command(Command::RequestGraphSettled(sender)) {
            self.report_error(error);
        }
        receiver
    }

    fn request_transport_settled(&mut self) -> Receiver<()> {
        let (sender, receiver) = bounded(1);
        if let Err(error) = self.send_command(Command::RequestTransportSettled(sender)) {
            self.report_error(error);
        }
        receiver
    }

    fn transport_play(&mut self) {
        if let Err(error) = self.send_command(Command::TransportPlay) {
            self.report_error(error);
        }
    }

    fn transport_pause(&mut self) {
        if let Err(error) = self.send_command(Command::TransportPause) {
            self.report_error(error);
        }
    }

    fn transport_seek_to_seconds(&mut self, position: Seconds) {
        if let Err(error) = self.send_command(Command::TransportSeekSeconds(position)) {
            self.report_error(error);
        }
    }

    fn transport_seek_to_beats(&mut self, position: Beats) {
        if let Err(error) = self.send_command(Command::TransportSeekBeats(position)) {
            self.report_error(error);
        }
    }

    fn request_transport_snapshot(&mut self) -> Receiver<Option<TransportSnapshot>> {
        let (sender, receiver) = bounded(1);
        if let Err(error) = self.send_command(Command::RequestTransportSnapshot(sender)) {
            self.report_error(error);
        }
        receiver
    }

    fn current_transport_snapshot(&self) -> Option<TransportSnapshot> {
        self.transport_snapshot_state.snapshot()
    }

    fn request_observability_snapshot(&mut self) -> Receiver<Option<ObservabilitySnapshot>> {
        let (sender, receiver) = bounded(1);
        if let Err(error) = self.send_command(Command::RequestObservabilitySnapshot(sender)) {
            self.report_error(error);
        }
        receiver
    }

    fn to_graph(&mut self, graph_id: GraphId) {
        self.selected_graph_remote_graph = graph_id;
    }

    fn to_top_level_graph(&mut self) {
        self.selected_graph_remote_graph = self.top_level_graph_id;
    }

    fn start_scheduling_bundle(&mut self, time: Time) {
        self.bundle_changes = true;
        self.changes_bundle_time = time;
        if !self.changes_bundle.is_empty() {
            eprintln!(
                "Warning: Starting a new scheduling bundle before the previous one was scheduled."
            )
        }
    }

    fn upload_scheduling_bundle(&mut self) {
        self.bundle_changes = false;
        let changes = SimultaneousChanges {
            time: self.changes_bundle_time,
            changes: self.changes_bundle.clone(),
        };
        self.schedule_changes(changes);
        self.changes_bundle.clear();
        self.changes_bundle_time = Time::Immediately;
    }

    fn current_graph(&self) -> GraphId {
        LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                g.id()
            } else {
                self.selected_graph_remote_graph
            }
        })
    }

    fn set_mortality(&mut self, node: NodeId, is_mortal: bool) {
        // The node may be in our local graph or remotely. Check local first.
        let found_in_local = LOCAL_GRAPH.with_borrow_mut(|g| {
            if let Some(g) = g.last_mut() {
                match g.set_node_mortality(node, is_mortal) {
                    Ok(()) => true,
                    Err(e) => match e {
                        ScheduleError::GraphNotFound(_) => false,
                        _ => {
                            self.report_error(e);
                            // We found the correct graph, but there was a different error
                            true
                        }
                    },
                }
            } else {
                false
            }
        });
        if !found_in_local {
            if let Err(error) = self.send_command(Command::SetMortality { node, is_mortal }) {
                self.report_error(error);
            }
        }
    }
    // /// Create a new Self which pushes to the selected GraphId by default
    // fn to_graph(&self, graph_id: GraphId) -> Self {
    //     let mut k = self.clone();
    //     k.default_graph_id = graph_id;
    //     k
    // }
    // /// Create a new Self which pushes to the top level GraphId by default
    // fn to_top_level_graph(&self) -> Self {
    //     let mut k = self.clone();
    //     k.default_graph_id = self.top_level_graph_id;
    //     k
    // }
}

thread_local! {
    static LOCAL_GRAPH: RefCell<Vec<Graph>> = RefCell::new(Vec::with_capacity(1));
}

/// Handle to modify a running/scheduled callback
pub struct CallbackHandle {
    free_flag: Arc<AtomicBool>,
}

impl CallbackHandle {
    /// Free/delete the callback this handle refers to.
    pub fn free(self) {
        self.free_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The beat on which a callback should start, either an absolute beat value or the next multiple of some number of beats.
pub enum StartBeat {
    /// An absolute time in beat
    Absolute(Beats),
    /// The next multiple of this number of beats
    Multiple(Beats),
}

type BeatCallbackFn = dyn FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send;

/// Callback that is scheduled in [`Beats`]. The closure inside the
/// callback should only schedule changes in Beats time guided by the value
/// to start scheduling that is passed to the function.
///
/// The closure takes two parameters: the time to start the next scheduling in
/// Beats time and a `&mut KnystCommands` for scheduling the changes. The
/// timestamp in the first parameter is the start time of the callback plus all
/// the returned beat intervals to wait until the next callback. The callback
/// can return the time to wait until it gets called again or `None` to remove
/// the callback.
pub struct BeatCallback {
    callback: Box<BeatCallbackFn>,
    next_timestamp: Beats,
    free_flag: Arc<AtomicBool>,
}
impl BeatCallback {
    /// Create a new [`BeatCallback`] with a given start time
    fn new(
        callback: impl FnMut(Beats, &mut MultiThreadedKnystCommands) -> Option<Beats> + Send + 'static,
        start_time: Beats,
    ) -> Self {
        let free_flag = Arc::new(AtomicBool::new(false));
        Self {
            callback: Box::new(callback),
            next_timestamp: start_time,
            free_flag,
        }
    }
    fn handle(&self) -> CallbackHandle {
        CallbackHandle {
            free_flag: self.free_flag.clone(),
        }
    }
    /// Called by the Controller when it is time to run the callback to schedule
    /// changes in the future.
    fn run_callback(&mut self, k: &mut MultiThreadedKnystCommands) -> CallbackResult {
        if self.free_flag.load(std::sync::atomic::Ordering::SeqCst) {
            CallbackResult::Delete
        } else {
            match (self.callback)(self.next_timestamp, k) {
                Some(time_to_next) => {
                    self.next_timestamp += time_to_next;
                    CallbackResult::Continue
                }
                None => CallbackResult::Delete,
            }
        }
    }
}

enum CallbackResult {
    Continue,
    Delete,
}

/// Receives commands from one or several [`KnystCommands`] that may be on
/// different threads, and applies those to a top level [`Graph`].
pub struct Controller {
    top_level_graph: Graph,
    command_receiver: Receiver<Command>,
    // TODO: Maybe we don't need to store the sender since it can be produced by cloning a ToKnyst
    command_sender: Sender<Command>,
    resources_sender: rtrb::Producer<ResourcesCommand>,
    resources_receiver: rtrb::Consumer<ResourcesResponse>,
    // The queue is for commands that couldn't be applied yet e.g. because a
    // NodeAddress couldn't be resolved because the node had not yet been
    // pushed.
    command_queue: Vec<(Instant, Command)>,
    graph_settled_waiters: Vec<(Arc<AtomicU64>, u64, Sender<()>)>,
    transport_settled_waiters: Vec<(Arc<AtomicU64>, u64, Sender<()>)>,
    error_handler: Box<dyn FnMut(KnystError) + Send>,
    beat_callbacks: Vec<BeatCallback>,
}
impl Controller {
    /// Creates a new [`Controller`] taking the top level [`Graph`] to which
    /// commands will be applied and an error handler. You almost never want to
    /// call this in program code; the AudioBackend will create one for you.
    pub fn new(
        top_level_graph: Graph,
        error_handler: impl FnMut(KnystError) + Send + 'static,
        resources_sender: rtrb::Producer<ResourcesCommand>,
        resources_receiver: rtrb::Consumer<ResourcesResponse>,
    ) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            top_level_graph,
            command_receiver: receiver,
            command_sender: sender,
            command_queue: vec![],
            graph_settled_waiters: vec![],
            transport_settled_waiters: vec![],
            error_handler: Box::new(error_handler),
            resources_receiver,
            resources_sender,
            beat_callbacks: vec![],
        }
    }

    fn apply_command(&mut self, command: Command) {
        let result: Result<(), crate::KnystError> = match command {
            Command::Push {
                gen_or_graph,
                inputs,
                mut node_address,
                graph_id,
                start_time,
            } => {
                if let Err(e) = self
                    .top_level_graph
                    .push_with_existing_address_to_graph_at_time(
                        gen_or_graph,
                        &mut node_address,
                        graph_id,
                        start_time,
                    )
                {
                    Err(From::from(e))
                } else {
                    self.top_level_graph
                        .apply_inputs_to_new_node(node_address, inputs)
                        .map_err(From::from)
                }
            }
            Command::Connect(connection) => {
                match self.top_level_graph.connect(connection.clone()) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        ConnectionError::SourceNodeNotPushed
                        | ConnectionError::SinkNodeNotPushed => {
                            self.command_queue
                                .push((Instant::now(), Command::Connect(connection)));
                            Ok(())
                        }
                        _ => Err(From::from(e)),
                    },
                }
            }
            Command::Disconnect(connection) => {
                match self.top_level_graph.disconnect(connection.clone()) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        ConnectionError::SourceNodeNotPushed
                        | ConnectionError::SinkNodeNotPushed => {
                            self.command_queue
                                .push((Instant::now(), Command::Disconnect(connection)));
                            Ok(())
                        }
                        _ => Err(From::from(e)),
                    },
                }
            }
            Command::FreeNode(node) => match self.top_level_graph.free_node(node) {
                Err(e) => {
                    if let FreeError::NodeNotFound = e {
                        self.command_queue
                            .push((Instant::now(), Command::FreeNode(node)));
                        Ok(())
                    } else {
                        Err(KnystError::from(e))
                    }
                }
                _ => Ok(()),
            },
            Command::FreeNodeMendConnections(node) => {
                match self.top_level_graph.free_node_mend_connections(node) {
                    Err(e) => {
                        if let FreeError::NodeNotFound = e {
                            self.command_queue
                                .push((Instant::now(), Command::FreeNodeMendConnections(node)));
                            Ok(())
                        } else {
                            Err(KnystError::from(e))
                        }
                    }
                    _ => Ok(()),
                }
            }
            Command::ScheduleChange(change) => self
                .top_level_graph
                .schedule_change(change)
                .map_err(From::from),
            Command::ScheduleEvent(event) => self
                .top_level_graph
                .schedule_event(event)
                .map_err(From::from),
            Command::FreeDisconnectedNodes => self
                .top_level_graph
                .free_disconnected_nodes()
                .map_err(From::from),
            Command::ResourcesCommand(resources_command) => {
                // Try sending it to Resources. If it fails, store it in the queue.
                match self.resources_sender.push(resources_command) {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        rtrb::PushError::Full(resources_command) => {
                            self.command_queue.push((
                                Instant::now(),
                                Command::ResourcesCommand(resources_command),
                            ));
                            Ok(())
                        }
                    },
                }
            }
            Command::ChangeMusicalTimeMap(change_fn) => self
                .top_level_graph
                .change_musical_time_map(change_fn)
                .map_err(From::from),
            Command::ScheduleChanges(changes) => {
                let changes_clone = changes.clone();
                match self
                    .top_level_graph
                    .schedule_changes(changes.changes, changes.time)
                {
                    Ok(_) => Ok(()),
                    Err(e) => match e {
                        crate::graph::ScheduleError::GraphNotFound(_node) => {
                            let _ = changes_clone;
                            Err(e.into())
                        }
                        _ => Err(e.into()),
                    },
                }
            }
            Command::ClearScheduledChanges => self
                .top_level_graph
                .clear_scheduled_changes()
                .map_err(From::from),
            Command::ScheduleBeatCallback(mut callback, start_beat) => {
                // Find the start beat
                let current_beats = self.top_level_graph.get_current_time_musical().unwrap();
                let start_timestamp = match start_beat {
                    StartBeat::Absolute(beats) => beats,
                    StartBeat::Multiple(beats) => {
                        let mut i = 1;
                        while beats * Beats::from_beats(i) < current_beats {
                            i += 1;
                        }
                        beats * Beats::from_beats(i)
                    }
                };
                // println!(
                //     "New callback, current beat: {current_beats:?}, start: {start_timestamp:?}"
                // );
                callback.next_timestamp = start_timestamp;
                self.beat_callbacks.push(callback);
                Ok(())
            }
            Command::RequestInspection(sender) => sender
                .send(self.top_level_graph.generate_inspection())
                .map_err(|_| ControllerError::InspectionResponseChannelClosed.into()),
            Command::RequestGraphSettled(sender) => {
                let (settled, target_generation) = self.top_level_graph.graph_settled_state();
                if settled.load(std::sync::atomic::Ordering::SeqCst) >= target_generation {
                    sender
                        .send(())
                        .map_err(|_| ControllerError::GraphSettledResponseChannelClosed.into())
                } else {
                    self.graph_settled_waiters
                        .push((settled, target_generation, sender));
                    Ok(())
                }
            }
            Command::RequestTransportSettled(sender) => {
                let (settled, target_generation) = self.top_level_graph.transport_settled_state();
                if settled.load(std::sync::atomic::Ordering::SeqCst) >= target_generation {
                    sender
                        .send(())
                        .map_err(|_| ControllerError::TransportSettledResponseChannelClosed.into())
                } else {
                    self.transport_settled_waiters
                        .push((settled, target_generation, sender));
                    Ok(())
                }
            }
            Command::RequestTransportSnapshot(sender) => sender
                .send(self.top_level_graph.shared_transport_snapshot().snapshot())
                .map_err(|_| ControllerError::TransportSnapshotResponseChannelClosed.into()),
            Command::RequestObservabilitySnapshot(sender) => sender
                .send(self.top_level_graph.observability_snapshot())
                .map_err(|_| ControllerError::ObservabilitySnapshotResponseChannelClosed.into()),
            Command::TransportPlay => self.top_level_graph.transport_play().map_err(From::from),
            Command::TransportPause => self.top_level_graph.transport_pause().map_err(From::from),
            Command::TransportSeekSeconds(position) => self
                .top_level_graph
                .transport_seek_to_seconds(position)
                .map_err(From::from),
            Command::TransportSeekBeats(position) => self
                .top_level_graph
                .transport_seek_to_beats(position)
                .map_err(From::from),
            Command::ReportDropouts(count) => {
                self.top_level_graph.increment_dropout_count(count);
                Ok(())
            }
            Command::ReportError(error) => Err(error),
            Command::SetMortality { node, is_mortal } => self
                .top_level_graph
                .set_node_mortality(node, is_mortal)
                .map_err(From::from),
        };

        if let Err(e) = result {
            (*self.error_handler)(e);
        }
    }

    // Receive commands from the queue and apply them to the graph. If
    // `max_commands` commands have been processed, return so that maintenance
    // functions can be run e.g. updating the scheduler.
    //
    // Returns true if all commands in the queue were processed.
    fn receive_and_apply_commands(&mut self, max_commands: usize) -> bool {
        let mut i = 0;
        while let Ok(command) = self.command_receiver.try_recv() {
            // println!("Received command in controller: {:?}", &command);
            self.apply_command(command);
            i += 1;
            if i >= max_commands {
                return false;
            }
        }
        true
    }

    /// Run maintenance tasks: update the graph and run internal maintenance
    fn run_maintenance(&mut self) {
        self.top_level_graph.update();
        let mut i = 0;
        while i < self.graph_settled_waiters.len() {
            if self.graph_settled_waiters[i]
                .0
                .load(std::sync::atomic::Ordering::SeqCst)
                >= self.graph_settled_waiters[i].1
            {
                let (_, _, sender) = self.graph_settled_waiters.remove(i);
                if sender.send(()).is_err() {
                    (*self.error_handler)(
                        ControllerError::GraphSettledResponseChannelClosed.into(),
                    );
                }
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.transport_settled_waiters.len() {
            if self.transport_settled_waiters[i]
                .0
                .load(std::sync::atomic::Ordering::SeqCst)
                >= self.transport_settled_waiters[i].1
            {
                let (_, _, sender) = self.transport_settled_waiters.remove(i);
                if sender.send(()).is_err() {
                    (*self.error_handler)(
                        ControllerError::TransportSettledResponseChannelClosed.into(),
                    );
                }
            } else {
                i += 1;
            }
        }
        while let Ok(response) = self.resources_receiver.pop() {
            match response {
                ResourcesResponse::InsertBuffer(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::RemoveBuffer(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::ReplaceBuffer(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::InsertWavetable(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::RemoveWavetable(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
                ResourcesResponse::ReplaceWavetable(res) => {
                    if let Err(e) = res {
                        (*self.error_handler)(e.into())
                    }
                }
            }
        }
    }

    fn run_callbacks(&mut self) {
        if self.beat_callbacks.is_empty() {
            return;
        }
        // Get current time in MusicalTime
        let current_time_beats = self.top_level_graph.get_current_time_musical();
        let mut k = self.get_knyst_commands();
        if let Some(current_time_beats) = current_time_beats {
            let mut i = self.beat_callbacks.len();
            while i != 0 {
                let c = &mut self.beat_callbacks[i - 1];
                if c.next_timestamp < current_time_beats
                    || c.next_timestamp.checked_sub(current_time_beats).unwrap()
                        < Beats::from_beats_f32(0.25)
                {
                    if let CallbackResult::Delete = c.run_callback(&mut k) {
                        self.beat_callbacks.remove(i - 1);
                    }
                }
                i -= 1;
            }
        }
    }

    /// Receives messages, applies them and then runs maintenance. Maintenance
    /// includes updating the [`Graph`], sending the changes made to the
    /// audio thread.
    ///
    /// `max_commands_before_update` is the maximum number of commands read from
    /// the queue before forcing maintenance. If you are sending a lot of
    /// commands, fine tuning this can probably reduce latency.
    ///
    /// Returns true if all commands in the queue were processed.
    pub fn run(&mut self, max_commands_before_update: usize) -> bool {
        // Run the callbacks first because they may send commands that would
        // then get picked up and applied just after.
        self.run_callbacks();
        let all_commands_received = self.receive_and_apply_commands(max_commands_before_update);
        self.run_maintenance();
        all_commands_received
    }

    fn loop_sleep_duration(&self, all_commands_received: bool) -> Duration {
        if !all_commands_received
            || !self.beat_callbacks.is_empty()
            || !self.graph_settled_waiters.is_empty()
            || !self.transport_settled_waiters.is_empty()
        {
            CONTROLLER_ACTIVE_SLEEP
        } else {
            CONTROLLER_IDLE_SLEEP
        }
    }

    /// Create a [`KnystCommands`] that can communicate with [`Self`]
    pub fn get_knyst_commands(&self) -> MultiThreadedKnystCommands {
        MultiThreadedKnystCommands {
            sender: self.command_sender.clone(),
            top_level_graph_id: self.top_level_graph.id(),
            top_level_graph_settings: self.top_level_graph.graph_settings(),
            selected_graph_remote_graph: self.top_level_graph.id(),
            bundle_changes: false,
            changes_bundle: vec![],
            changes_bundle_time: Time::Immediately,
            transport_snapshot_state: self.top_level_graph.shared_transport_snapshot(),
        }
    }

    /// Consumes the [`Controller`] and moves it to a new thread where it will `run` in a loop.
    pub fn start_on_new_thread(self) -> MultiThreadedKnystCommands {
        let top_level_graph_id = self.top_level_graph.id();
        let top_level_graph_settings = self.top_level_graph.graph_settings();
        let transport_snapshot_state = self.top_level_graph.shared_transport_snapshot();
        let controller_block_size = top_level_graph_settings.block_size as u32;
        let controller_sample_rate = top_level_graph_settings.sample_rate.round() as u32;
        let mut controller = self;
        let sender = controller.command_sender.clone();

        std::thread::Builder::new()
            .name("knyst-controller".to_string())
            .spawn(move || {
                elevate_controller_thread_priority(controller_block_size, controller_sample_rate);
                loop {
                    let all_commands_received = controller.run(300);
                    if all_commands_received {
                        std::thread::sleep(controller.loop_sleep_duration(all_commands_received));
                    } else {
                        std::thread::yield_now();
                    }
                }
            })
            .expect("failed to spawn knyst controller thread");

        MultiThreadedKnystCommands {
            sender,
            top_level_graph_id,
            top_level_graph_settings,
            selected_graph_remote_graph: top_level_graph_id,
            bundle_changes: false,
            changes_bundle: vec![],
            changes_bundle_time: Time::Immediately,
            transport_snapshot_state,
        }
    }
}

fn elevate_controller_thread_priority(block_size: u32, sample_rate: u32) {
    if block_size == 0 || sample_rate == 0 {
        return;
    }

    if let Err(error) = promote_current_thread_to_real_time(block_size, sample_rate) {
        eprintln!("Knyst controller thread priority promotion failed: {error}");
    }
}

/// Simple error handler that just prints the error using `eprintln!`
pub fn print_error_handler(e: KnystError) {
    eprintln!("Error in Controller: {e}");
}

#[cfg(test)]
mod tests {
    use super::{
        schedule_bundle, BeatCallback, Command, Controller, ControllerError,
        CONTROLLER_ACTIVE_SLEEP, CONTROLLER_IDLE_SLEEP,
    };
    use crate as knyst;
    use crate::{
        graph::{Graph, GraphSettings, NodeId, TransportState},
        knyst_commands,
        offline::KnystOffline,
        prelude::*,
        scheduling::TempoChange,
        trig::once_trig,
        KnystError,
    };
    use crossbeam_channel::bounded;
    use std::sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use std::time::Duration;

    fn new_test_controller(errors: Arc<Mutex<Vec<KnystError>>>) -> Controller {
        let graph = Graph::new(GraphSettings::default());
        let (resources_sender, _resources_receiver) = rtrb::RingBuffer::new(8);
        let (_resources_response_sender, resources_receiver) = rtrb::RingBuffer::new(8);
        Controller::new(
            graph,
            move |error| {
                errors
                    .lock()
                    .expect("test error sink lock should not be poisoned")
                    .push(error);
            },
            resources_sender,
            resources_receiver,
        )
    }

    // Outputs its input value + 1
    struct OneGen {}
    #[impl_gen]
    impl OneGen {
        fn new() -> Self {
            Self {}
        }
        #[process]
        fn process(&mut self, passthrough: &[Sample], out: &mut [Sample]) -> GenState {
            for (i, o) in passthrough.iter().zip(out.iter_mut()) {
                *o = *i + 1.0;
            }
            GenState::Continue
        }
    }

    struct ConstantSignalGen;
    #[impl_gen]
    impl ConstantSignalGen {
        fn new() -> Self {
            Self
        }
        #[process]
        fn process(&mut self, out: &mut [Sample]) -> GenState {
            for sample in out.iter_mut() {
                *sample = 0.25;
            }
            GenState::Continue
        }
    }

    struct GainGen;
    #[impl_gen]
    impl GainGen {
        fn new() -> Self {
            Self
        }
        #[process]
        fn process(&mut self, input: &[Sample], gain: &[Sample], out: &mut [Sample]) -> GenState {
            for ((input, gain), out) in input.iter().zip(gain.iter()).zip(out.iter_mut()) {
                *out = *input * *gain;
            }
            GenState::Continue
        }
    }

    fn output_has_signal(offline: &KnystOffline, channel: usize) -> bool {
        offline
            .output_channel(channel)
            .is_some_and(|output| output.iter().any(|sample| sample.abs() > 1.0e-4))
    }

    struct SleepUntilEventGen {
        awake_blocks_left: usize,
    }

    impl SleepUntilEventGen {
        fn new() -> Self {
            Self {
                awake_blocks_left: 0,
            }
        }
    }

    impl Gen for SleepUntilEventGen {
        fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
            if !ctx.events.is_empty() {
                self.awake_blocks_left = 1;
            }

            for channel in ctx.outputs.iter_mut() {
                channel.fill(0.0);
            }

            if self.awake_blocks_left > 0 {
                self.awake_blocks_left -= 1;
                for channel in ctx.outputs.iter_mut() {
                    channel.fill(0.5);
                }
                GenState::Continue
            } else {
                GenState::Sleep
            }
        }

        fn num_inputs(&self) -> usize {
            0
        }

        fn num_outputs(&self) -> usize {
            1
        }

        fn num_event_inputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "SleepUntilEventGen"
        }
    }

    struct SleepUntilGainGen;

    impl SleepUntilGainGen {
        fn new() -> Self {
            Self
        }
    }

    impl Gen for SleepUntilGainGen {
        fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
            let gain = ctx.inputs.get_channel(0);
            let mut any_signal = false;
            for (gain, out) in gain
                .iter()
                .zip(ctx.outputs.iter_mut().next().unwrap().iter_mut())
            {
                let sample = if *gain > 0.0 { *gain } else { 0.0 };
                any_signal |= sample > 0.0;
                *out = sample;
            }
            if any_signal {
                GenState::Continue
            } else {
                GenState::Sleep
            }
        }

        fn num_inputs(&self) -> usize {
            1
        }

        fn num_outputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "SleepUntilGainGen"
        }
    }

    struct CountAndSleepGen {
        calls: Arc<AtomicUsize>,
    }

    impl CountAndSleepGen {
        fn new(calls: Arc<AtomicUsize>) -> Self {
            Self { calls }
        }
    }

    impl Gen for CountAndSleepGen {
        fn process(&mut self, ctx: GenContext, _resources: &mut Resources) -> GenState {
            self.calls.fetch_add(1, Ordering::SeqCst);
            for channel in ctx.outputs.iter_mut() {
                channel.fill(0.0);
            }
            GenState::Sleep
        }

        fn num_inputs(&self) -> usize {
            0
        }

        fn num_outputs(&self) -> usize {
            1
        }

        fn name(&self) -> &'static str {
            "CountAndSleepGen"
        }
    }

    #[test]
    fn schedule_bundle_test() {
        let sr = 44100;
        let mut kt = KnystOffline::new(sr, 64, 0, 1);
        schedule_bundle(crate::graph::Time::Immediately, || {
            graph_output(0, once_trig());
        });
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(5, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(10, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        let mut og = None;
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(16, sr as u64)),
            || {
                og = Some(one_gen());
                graph_output(0, og.unwrap());
            },
        );
        let og = og.unwrap();
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(17, sr as u64)),
            || {
                og.passthrough(2.0);
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(19, sr as u64)),
            || {
                og.passthrough(3.0);
            },
        );
        // Try with the pure KnystCommands methods as well.
        knyst_commands().start_scheduling_bundle(knyst::graph::Time::Seconds(
            Seconds::from_samples(20, sr as u64),
        ));
        og.passthrough(4.0);
        knyst_commands().upload_scheduling_bundle();
        kt.process_block();
        let o = kt.output_channel(0).unwrap();
        dbg!(o);
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[4], 0.0);
        assert_eq!(o[5], 1.0);
        assert_eq!(o[6], 0.0);
        assert_eq!(o[10], 1.0);
        assert_eq!(o[11], 0.0);
        assert_eq!(o[16], 1.0);
        assert_eq!(o[17], 3.0);
        assert_eq!(o[19], 4.0);
        assert_eq!(o[20], 5.0);
    }

    #[test]
    fn schedule_bundle_inner_graph_test() {
        let sr = 44100;
        let mut kt = KnystOffline::new(sr, 64, 0, 1);
        // We create a first graph so that the top graph will try to schedule on this one first and fail.
        let mut ignored_graph_node = None;
        let _ignored_graph = upload_graph(knyst_commands().default_graph_settings(), || {
            ignored_graph_node = Some(one_gen());
        })
        .expect("test graph upload should succeed");
        let mut inner_graph = None;
        let graph = upload_graph(knyst_commands().default_graph_settings(), || {
            let g = upload_graph(knyst_commands().default_graph_settings(), || ())
                .expect("test graph upload should succeed");
            graph_output(0, g);
            inner_graph = Some(g);
        })
        .expect("test graph upload should succeed");
        graph_output(0, graph);
        inner_graph.unwrap().activate();
        schedule_bundle(crate::graph::Time::Immediately, || {
            graph_output(0, once_trig());
        });
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(5, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(10, sr as u64)),
            || {
                graph_output(0, once_trig());
            },
        );
        let mut og = None;
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(16, sr as u64)),
            || {
                og = Some(one_gen());
                graph_output(0, og.unwrap());
            },
        );
        let og = og.unwrap();
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(17, sr as u64)),
            || {
                og.passthrough(2.0);

                // This will set a value of a node in a different graph, but won't change the output
                ignored_graph_node.unwrap().passthrough(10.);
            },
        );
        schedule_bundle(
            crate::graph::Time::Seconds(Seconds::from_samples(19, sr as u64)),
            || {
                og.passthrough(3.0);
            },
        );
        // Try with the pure KnystCommands methods as well.
        knyst_commands().start_scheduling_bundle(knyst::graph::Time::Seconds(
            Seconds::from_samples(20, sr as u64),
        ));
        og.passthrough(4.0);
        knyst_commands().upload_scheduling_bundle();
        kt.process_block();
        let o = kt.output_channel(0).unwrap();
        dbg!(o);
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[4], 0.0);
        assert_eq!(o[5], 1.0);
        assert_eq!(o[6], 0.0);
        assert_eq!(o[10], 1.0);
        assert_eq!(o[11], 0.0);
        assert_eq!(o[16], 1.0);
        assert_eq!(o[17], 3.0);
        assert_eq!(o[19], 4.0);
        assert_eq!(o[20], 5.0);
    }

    #[test]
    fn request_inspection_reports_dropped_receiver() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);
        drop(receiver);

        controller.apply_command(Command::RequestInspection(sender));

        let errors = errors
            .lock()
            .expect("test error sink lock should not be poisoned");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            KnystError::ControllerError(ControllerError::InspectionResponseChannelClosed)
        ));
    }

    #[test]
    fn request_transport_snapshot_reports_dropped_receiver() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);
        drop(receiver);

        controller.apply_command(Command::RequestTransportSnapshot(sender));

        let errors = errors
            .lock()
            .expect("test error sink lock should not be poisoned");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            KnystError::ControllerError(ControllerError::TransportSnapshotResponseChannelClosed)
        ));
    }

    #[test]
    fn request_graph_settled_reports_dropped_receiver() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);
        drop(receiver);

        controller.apply_command(Command::RequestGraphSettled(sender));

        let errors = errors
            .lock()
            .expect("test error sink lock should not be poisoned");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            KnystError::ControllerError(ControllerError::GraphSettledResponseChannelClosed)
        ));
    }

    #[test]
    fn request_graph_settled_returns_immediately_without_pending_audio_changes() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();
        offline.process_block();

        let receiver = commands.request_graph_settled();
        offline.process_block();
        receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("settled graph should respond immediately");
    }

    #[test]
    fn request_graph_settled_waits_for_audio_thread_task_application() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();

        let _ = commands.push(OneGen::new(), crate::inputs![]);
        offline.run_controller_only();
        let receiver = commands.request_graph_settled();
        offline.run_controller_only();
        assert!(
            receiver.recv_timeout(Duration::from_millis(10)).is_err(),
            "graph should not be settled before the audio thread swaps task data"
        );
        offline.run_audio_only();
        offline.run_controller_only();
        receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("graph should settle after controller commit and audio-thread application");
    }

    #[test]
    fn request_graph_settled_waits_after_controller_commit_but_before_audio_swap() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();
        offline.process_block();

        let _ = commands.push(OneGen::new(), crate::inputs![]);
        offline.run_controller_only();

        let receiver = commands.request_graph_settled();
        offline.run_controller_only();
        assert!(
            receiver.recv_timeout(Duration::from_millis(10)).is_err(),
            "graph should not be reported settled after controller commit alone"
        );

        offline.run_audio_only();
        offline.run_controller_only();
        receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("graph should settle once the audio thread swaps to the committed task data");
    }

    #[test]
    fn request_transport_settled_reports_dropped_receiver() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);
        drop(receiver);

        controller.apply_command(Command::RequestTransportSettled(sender));

        let errors = errors
            .lock()
            .expect("test error sink lock should not be poisoned");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            KnystError::ControllerError(ControllerError::TransportSettledResponseChannelClosed)
        ));
    }

    #[test]
    fn request_transport_settled_returns_immediately_without_pending_audio_changes() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();
        offline.process_block();

        let receiver = commands.request_transport_settled();
        offline.process_block();
        receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("settled transport should respond immediately");
    }

    #[test]
    fn request_transport_settled_waits_for_audio_thread_transport_application() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();

        commands.transport_seek_to_beats(Beats::from_beats(4));
        offline.run_controller_only();
        let receiver = commands.request_transport_settled();
        offline.run_controller_only();
        assert!(
            receiver.recv_timeout(Duration::from_millis(10)).is_err(),
            "transport should not be settled until the audio thread consumes the update"
        );
        offline.run_audio_only();
        offline.run_controller_only();
        receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("transport should settle after controller commit and audio-thread application");
    }

    #[test]
    fn request_transport_settled_waits_after_controller_commit_but_before_audio_apply() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();
        offline.process_block();

        commands.transport_seek_to_beats(Beats::from_beats(4));
        offline.run_controller_only();

        let receiver = commands.request_transport_settled();
        offline.run_controller_only();
        assert!(
            receiver.recv_timeout(Duration::from_millis(10)).is_err(),
            "transport should not be reported settled after controller commit alone"
        );

        offline.run_audio_only();
        offline.run_controller_only();
        receiver
            .recv_timeout(Duration::from_millis(50))
            .expect("transport should settle once the audio thread consumes the committed update");
    }

    #[test]
    fn request_transport_snapshot_returns_none_without_running_scheduler() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);

        controller.apply_command(Command::RequestTransportSnapshot(sender));
        let snapshot = receiver
            .recv()
            .expect("transport snapshot response should be sent");
        assert!(snapshot.is_none());
    }

    #[test]
    fn current_transport_snapshot_returns_none_without_running_scheduler() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let controller = new_test_controller(errors);
        let commands = controller.get_knyst_commands();

        assert!(commands.current_transport_snapshot().is_none());
    }

    #[test]
    fn current_transport_snapshot_tracks_play_pause_and_seek() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();

        let initial = commands
            .current_transport_snapshot()
            .expect("offline scheduler should expose an initial transport snapshot");
        assert_eq!(initial.state, TransportState::Playing);
        assert_eq!(initial.samples, 0);
        assert_eq!(initial.seconds, Seconds::ZERO);

        commands.transport_pause();
        offline.process_block();
        let paused = commands
            .current_transport_snapshot()
            .expect("paused transport snapshot should be available");
        assert_eq!(paused.state, TransportState::Paused);
        assert_eq!(paused.samples, 0);

        let target = Seconds::from_seconds_f64(0.5);
        commands.transport_seek_to_seconds(target);
        offline.process_block();
        let sought = commands
            .current_transport_snapshot()
            .expect("sought transport snapshot should be available");
        assert_eq!(sought.state, TransportState::Paused);
        assert_eq!(sought.seconds, target);
        assert_eq!(sought.samples, target.to_samples(48_000));

        commands.transport_play();
        offline.process_block();
        let resumed = commands
            .current_transport_snapshot()
            .expect("resumed transport snapshot should be available");
        assert_eq!(resumed.state, TransportState::Playing);
        assert!(resumed.samples >= sought.samples);
    }

    #[test]
    fn current_transport_snapshot_keeps_reliable_beats_after_tempo_change() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 2);
        let mut commands = offline.context().commands();

        commands.change_musical_time_map(|map| {
            map.replace(0, TempoChange::NewTempo { bpm: 120.0 });
        });
        commands.transport_pause();
        offline.process_block();
        commands.transport_seek_to_beats(Beats::from_beats(4));
        offline.process_block();

        let snapshot = commands
            .current_transport_snapshot()
            .expect("transport snapshot should be available");
        assert_eq!(snapshot.state, TransportState::Paused);
        assert_eq!(snapshot.beats, Some(Beats::from_beats(4)));
        assert_eq!(snapshot.seconds, Seconds::from_seconds_f64(2.0));
    }

    #[test]
    fn controller_loop_uses_idle_sleep_without_callbacks_or_waiters() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let controller = new_test_controller(errors);

        assert_eq!(controller.loop_sleep_duration(true), CONTROLLER_IDLE_SLEEP);
    }

    #[test]
    fn controller_loop_uses_active_sleep_with_callbacks() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors);
        controller
            .beat_callbacks
            .push(BeatCallback::new(|_, _| None, Beats::ZERO));

        assert_eq!(
            controller.loop_sleep_duration(true),
            CONTROLLER_ACTIVE_SLEEP
        );
    }

    #[test]
    fn controller_loop_uses_active_sleep_with_pending_waiters() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors);
        let settled = Arc::new(AtomicU64::new(0));
        let (sender, _receiver) = bounded(1);
        controller.graph_settled_waiters.push((settled, 1, sender));

        assert_eq!(
            controller.loop_sleep_duration(true),
            CONTROLLER_ACTIVE_SLEEP
        );
    }

    #[test]
    fn request_observability_snapshot_reports_dropped_receiver() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);
        drop(receiver);

        controller.apply_command(Command::RequestObservabilitySnapshot(sender));

        let errors = errors
            .lock()
            .expect("test error sink lock should not be poisoned");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            KnystError::ControllerError(
                ControllerError::ObservabilitySnapshotResponseChannelClosed
            )
        ));
    }

    #[test]
    fn request_observability_snapshot_returns_none_without_running_scheduler() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let (sender, receiver) = bounded(1);

        controller.apply_command(Command::RequestObservabilitySnapshot(sender));
        let snapshot = receiver
            .recv()
            .expect("observability snapshot response should be sent");
        assert!(snapshot.is_none());
    }

    #[test]
    fn report_error_command_forwards_to_error_handler() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut controller = new_test_controller(errors.clone());
        let expected_error = KnystError::ControllerError(ControllerError::CommandChannelClosed);

        controller.apply_command(Command::ReportError(expected_error));

        let errors = errors
            .lock()
            .expect("test error sink lock should not be poisoned");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            &errors[0],
            KnystError::ControllerError(ControllerError::CommandChannelClosed)
        ));
    }

    #[test]
    fn command_api_does_not_panic_when_controller_is_dropped() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let controller = new_test_controller(errors);
        let mut commands = controller.get_knyst_commands();
        let graph_id = commands.current_graph();
        drop(controller);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            commands.free_disconnected_nodes();
            commands.free_node(NodeId::new(graph_id));
            commands.schedule_event(EventChange::now(
                NodeId::new(graph_id).event_input(0),
                crate::graph::EventPayload::U32(1),
            ));
            commands.clear_scheduled_changes();
            commands.change_musical_time_map(|_| {});
            let _ = commands.request_inspection();
            commands.transport_play();
            commands.transport_pause();
            commands.transport_seek_to_seconds(Seconds::ZERO);
            commands.transport_seek_to_beats(Beats::ZERO);
            let _ = commands.request_transport_snapshot();
            let _ = commands.request_observability_snapshot();
        }));

        assert!(result.is_ok());
    }

    #[test]
    fn preexisting_gain_node_goes_silent_after_nonzero_seek() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 1);
        offline.context().with_activation(|| {
            let source = knyst_commands().push(ConstantSignalGen::new(), inputs![]);
            let gain = knyst_commands().push(GainGen::new(), inputs!((1 : 1.0)));
            knyst_commands().connect(source.to(gain));
            knyst_commands().connect(Connection::graph_output(gain));
        });

        knyst_commands().transport_pause();
        knyst_commands().transport_seek_to_beats(Beats::from_beats(1));
        for _ in 0..8 {
            offline.process_block();
        }
        knyst_commands().transport_play();
        for _ in 0..16 {
            offline.process_block();
        }

        assert!(
            output_has_signal(&offline, 0),
            "preexisting gain node went silent after non-zero transport seek"
        );
    }

    #[test]
    fn settled_preexisting_gain_node_survives_nonzero_seek() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 1);
        offline.context().with_activation(|| {
            let source = knyst_commands().push(ConstantSignalGen::new(), inputs![]);
            let gain = knyst_commands().push(GainGen::new(), inputs!((1 : 1.0)));
            knyst_commands().connect(source.to(gain));
            knyst_commands().connect(Connection::graph_output(gain));
        });

        for _ in 0..16 {
            offline.process_block();
        }
        assert!(
            output_has_signal(&offline, 0),
            "preexisting gain node should be audible before seek"
        );

        knyst_commands().transport_pause();
        knyst_commands().transport_seek_to_beats(Beats::from_beats(1));
        for _ in 0..8 {
            offline.process_block();
        }
        knyst_commands().transport_play();
        for _ in 0..16 {
            offline.process_block();
        }

        assert!(
            output_has_signal(&offline, 0),
            "settled preexisting gain node went silent after non-zero transport seek"
        );
    }

    #[test]
    fn sleeping_node_wakes_on_event() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 1);
        let node = offline.context().with_activation(|| {
            let node = knyst_commands().push(SleepUntilEventGen::new(), inputs![]);
            knyst_commands().connect(Connection::graph_output(node));
            node
        });

        offline.process_block();
        assert!(
            !output_has_signal(&offline, 0),
            "sleeping node should stay silent before an event"
        );

        knyst_commands()
            .schedule_event(EventChange::now(node.event_input(0), EventPayload::U32(1)));
        offline.process_block();
        assert!(
            output_has_signal(&offline, 0),
            "sleeping node did not wake on event delivery"
        );
    }

    #[test]
    fn sleeping_node_wakes_on_parameter_change() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 1);
        let node = offline.context().with_activation(|| {
            let node = knyst_commands().push(SleepUntilGainGen::new(), inputs!((0 : 0.0)));
            knyst_commands().connect(Connection::graph_output(node));
            node
        });

        offline.process_block();
        assert!(
            !output_has_signal(&offline, 0),
            "sleeping node should stay silent before a parameter change"
        );

        knyst_commands().schedule_change(ParameterChange::now(node.input(0), 0.75));
        offline.process_block();
        assert!(
            output_has_signal(&offline, 0),
            "sleeping node did not wake on parameter change"
        );
    }

    #[test]
    fn sleeping_node_is_not_processed_again_until_woken() {
        let mut offline = KnystOffline::new(48_000, 64, 0, 1);
        let calls = Arc::new(AtomicUsize::new(0));
        offline.context().with_activation(|| {
            let node = knyst_commands().push(CountAndSleepGen::new(calls.clone()), inputs![]);
            knyst_commands().connect(Connection::graph_output(node));
        });

        offline.process_block();
        offline.process_block();
        offline.process_block();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "sleeping node kept being processed after returning Sleep"
        );
    }
}
