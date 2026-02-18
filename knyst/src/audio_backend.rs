//! Audio backends for getting up and running quickly.
//! To use the backends in this module you need to enable either the jack or the cpal feature.
//!
//! [`JackBackend`] currently has better support including a duplex client with
//! the same number of inputs and outputs as the [`Graph`].
//!
//! To use an [`AudioBackend`], first create it to get the parameters of the
//! system. When you have created your main graph, call
//! [`AudioBackend::start_processing`]. This will do something similar to
//! creating a [`RunGraph`] from a `&mut Graph` and a `Resources` and populating
//! the backend output buffer with the output of the [`Graph`]. From this point,
//! the [`Graph`] is considered to be running, meaning changes to the [`Graph`]
//! may take longer to perform since they involve the audio thread.

use crate::{
    controller::Controller, graph::RunGraphSettings, prelude::MultiThreadedKnystCommands,
    KnystError,
};
#[allow(unused)]
use crate::{
    graph::{Graph, RunGraph},
    Resources,
};

#[cfg(feature = "cpal")]
pub use cpal_backend::{CpalBackend, CpalBackendOptions};
#[cfg(feature = "jack")]
pub use jack_backend::JackBackend;

/// Unified API for different backends.
pub trait AudioBackend {
    /// Starts processing and returns a [`Controller`]. This is the easiest
    /// option and will run the [`Controller`] in a loop on a new thread.
    fn start_processing(
        &mut self,
        graph: Graph,
        resources: Resources,
        run_graph_settings: RunGraphSettings,
        error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
    ) -> Result<MultiThreadedKnystCommands, AudioBackendError> {
        let controller = self.start_processing_return_controller(
            graph,
            resources,
            run_graph_settings,
            error_handler,
        )?;
        Ok(controller.start_on_new_thread())
    }
    /// Starts processing and returns a [`Controller`]. This is suitable if you
    /// want to run single threaded or handle running the [`Controller`]
    /// manually.
    fn start_processing_return_controller(
        &mut self,
        graph: Graph,
        resources: Resources,
        run_graph_settings: RunGraphSettings,
        error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
    ) -> Result<Controller, AudioBackendError>;
    /// Stop the backend
    fn stop(&mut self) -> Result<(), AudioBackendError>;
    /// Get the native sample rate of the backend
    fn sample_rate(&self) -> usize;
    /// Get the native block size of the backend if there is one
    fn block_size(&self) -> Option<usize>;
    /// Get the native number of output channels for this backend, if any
    fn native_output_channels(&self) -> Option<usize>;
    /// Get the native number of input channels for this backend, if any
    fn native_input_channels(&self) -> Option<usize>;
}

#[allow(missing_docs)]
#[derive(thiserror::Error, Debug)]
pub enum AudioBackendError {
    #[error("You tried to start a backend that was already running. A backend can only be started once.")]
    BackendAlreadyRunning,
    #[error("You tried to stop a backend that was already stopped.")]
    BackendNotRunning,
    #[error("Unable to create a node from the Graph: {0}")]
    CouldNotCreateNode(String),
    #[error(
        "Graph output channel count ({graph_outputs}) does not match backend output channels ({backend_outputs})."
    )]
    GraphOutputChannelsMismatch {
        graph_outputs: usize,
        backend_outputs: usize,
    },
    #[error(
        "Graph input channel count ({graph_inputs}) exceeds backend input channels ({backend_inputs})."
    )]
    GraphInputChannelsMismatch {
        graph_inputs: usize,
        backend_inputs: usize,
    },
    #[error(
        "CPAL input sample rate ({input_sample_rate}) does not match output sample rate ({output_sample_rate})."
    )]
    CpalInputOutputSampleRateMismatch {
        input_sample_rate: usize,
        output_sample_rate: usize,
    },
    #[error("No output device was found for CPAL selection: {0}")]
    OutputDeviceNotFound(String),
    #[error(transparent)]
    RunGraphError(#[from] crate::graph::run_graph::RunGraphError),
    #[cfg(feature = "jack")]
    #[error(transparent)]
    JackError(#[from] jack::Error),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalDevicesError(#[from] cpal::DevicesError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalDeviceNameError(#[from] cpal::DeviceNameError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalDefaultStreamConfigError(#[from] cpal::DefaultStreamConfigError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalStreamError(#[from] cpal::StreamError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalBuildStreamError(#[from] cpal::BuildStreamError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalPlayStreamError(#[from] cpal::PlayStreamError),
    #[cfg(feature = "cpal")]
    #[error(transparent)]
    CpalPauseStreamError(#[from] cpal::PauseStreamError),
    #[cfg(feature = "cpal")]
    #[error("Unsupported CPAL sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),
    #[cfg(feature = "cpal")]
    #[error(
        "CPAL input overflow: dropped {dropped_samples} sample(s) because the input queue was full."
    )]
    CpalInputOverflow { dropped_samples: usize },
    #[cfg(feature = "cpal")]
    #[error("CPAL input underflow: filled {missing_samples} missing sample(s) with zeros.")]
    CpalInputUnderflow { missing_samples: usize },
    #[cfg(feature = "jack")]
    #[error(
        "JACK sample rate changed from {expected_sample_rate} to {actual_sample_rate} while running."
    )]
    JackSampleRateChanged {
        expected_sample_rate: usize,
        actual_sample_rate: usize,
    },
}

#[cfg(feature = "jack")]
mod jack_backend {
    use crate::audio_backend::{AudioBackend, AudioBackendError};
    use crate::controller::Controller;
    use crate::graph::{RunGraph, RunGraphSettings};
    use crate::{graph::Graph, Resources};
    use crate::{KnystError, Sample};
    #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
    use assert_no_alloc::*;
    enum JackClient {
        Passive(jack::Client),
        Active(jack::AsyncClient<JackNotifications, JackProcess>),
    }

    /// A backend using JACK
    pub struct JackBackend {
        client: Option<JackClient>,
        sample_rate: usize,
        block_size: usize,
    }

    impl JackBackend {
        /// Create a new JACK client using the given name
        pub fn new<S: AsRef<str>>(name: S) -> Result<Self, jack::Error> {
            // Create client
            let (client, _status) =
                jack::Client::new(name.as_ref(), jack::ClientOptions::NO_START_SERVER)?;
            let sample_rate = client.sample_rate();
            let block_size = client.buffer_size() as usize;
            Ok(Self {
                client: Some(JackClient::Passive(client)),
                sample_rate,
                block_size,
            })
        }
    }

    impl AudioBackend for JackBackend {
        fn start_processing_return_controller(
            &mut self,
            mut graph: Graph,
            resources: Resources,
            run_graph_settings: RunGraphSettings,
            error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
        ) -> Result<Controller, AudioBackendError> {
            let client = match self.client.take() {
                Some(JackClient::Passive(client)) => client,
                Some(active_client @ JackClient::Active(_)) => {
                    self.client = Some(active_client);
                    return Err(AudioBackendError::BackendAlreadyRunning);
                }
                None => {
                    return Err(AudioBackendError::BackendAlreadyRunning);
                }
            };
            {
                let mut in_ports = vec![];
                let mut out_ports = vec![];
                let num_inputs = graph.num_inputs();
                let num_outputs = graph.num_outputs();
                for i in 0..num_inputs {
                    in_ports
                        .push(client.register_port(&format!("in_{i}"), jack::AudioIn::default())?);
                }
                for i in 0..num_outputs {
                    out_ports.push(
                        client.register_port(&format!("out_{i}"), jack::AudioOut::default())?,
                    );
                }
                let (run_graph, resources_command_sender, resources_command_receiver) =
                    RunGraph::new(&mut graph, resources, run_graph_settings)?;
                let jack_process = JackProcess {
                    run_graph,
                    in_ports,
                    out_ports,
                };
                // Activate the client, which starts the processing.
                let active_client = client
                    .activate_async(JackNotifications::new(self.sample_rate), jack_process)?;
                self.client = Some(JackClient::Active(active_client));
                let controller = Controller::new(
                    graph,
                    error_handler,
                    resources_command_sender,
                    resources_command_receiver,
                );
                Ok(controller)
            }
        }

        fn stop(&mut self) -> Result<(), AudioBackendError> {
            match self.client.take() {
                Some(JackClient::Active(active_client)) => {
                    let (client, _notifications, _process) = active_client.deactivate()?;
                    self.client = Some(JackClient::Passive(client));
                    Ok(())
                }
                Some(passive_client @ JackClient::Passive(_)) => {
                    self.client = Some(passive_client);
                    Err(AudioBackendError::BackendNotRunning)
                }
                None => Err(AudioBackendError::BackendNotRunning),
            }
        }

        fn sample_rate(&self) -> usize {
            self.sample_rate
        }

        fn block_size(&self) -> Option<usize> {
            Some(self.block_size)
        }

        fn native_output_channels(&self) -> Option<usize> {
            None
        }

        fn native_input_channels(&self) -> Option<usize> {
            None
        }
    }

    struct JackProcess {
        run_graph: RunGraph,
        in_ports: Vec<jack::Port<jack::AudioIn>>,
        out_ports: Vec<jack::Port<jack::AudioOut>>,
    }

    impl jack::ProcessHandler for JackProcess {
        fn process(&mut self, _: &jack::Client, ps: &jack::ProcessScope) -> jack::Control {
            // Duplication due to conditional compilation
            #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
            {
                assert_no_alloc(|| {
                    let graph_input_buffers = self.run_graph.graph_input_buffers();
                    for (i, in_port) in self.in_ports.iter().enumerate() {
                        let in_port_slice = in_port.as_slice(ps);
                        let in_buffer = unsafe { graph_input_buffers.get_channel_mut(i) };
                        // in_buffer.clone_from_slice(in_port_slice);
                        for (from_jack, graph_in) in in_port_slice.iter().zip(in_buffer.iter_mut())
                        {
                            *graph_in = *from_jack as Sample;
                        }
                    }
                    self.run_graph.run_resources_communication(50);
                    self.run_graph.process_block();

                    let graph_output_buffers = self.run_graph.graph_output_buffers_mut();
                    for (i, out_port) in self.out_ports.iter_mut().enumerate() {
                        let out_buffer = unsafe { graph_output_buffers.get_channel_mut(i) };
                        for sample in out_buffer.iter_mut() {
                            *sample = sample.clamp(-1.0, 1.0);
                            if sample.is_nan() {
                                *sample = 0.0;
                            }
                        }
                        let out_port_slice = out_port.as_mut_slice(ps);
                        // out_port_slice.clone_from_slice(out_buffer);
                        for (to_jack, graph_out) in out_port_slice.iter_mut().zip(out_buffer.iter())
                        {
                            *to_jack = *graph_out;
                        }
                    }
                    jack::Control::Continue
                })
            }
            #[cfg(not(all(debug_assertions, feature = "assert_no_alloc")))]
            {
                let graph_input_buffers = self.run_graph.graph_input_buffers();
                for (i, in_port) in self.in_ports.iter().enumerate() {
                    let in_port_slice = in_port.as_slice(ps);
                    let in_buffer = unsafe { graph_input_buffers.get_channel_mut(i) };
                    // in_buffer.clone_from_slice(in_port_slice);
                    for (from_jack, graph_in) in in_port_slice.iter().zip(in_buffer.iter_mut()) {
                        *graph_in = *from_jack as Sample;
                    }
                }
                self.run_graph.run_resources_communication(50);
                self.run_graph.process_block();

                let graph_output_buffers = self.run_graph.graph_output_buffers_mut();
                for (i, out_port) in self.out_ports.iter_mut().enumerate() {
                    let out_buffer = unsafe { graph_output_buffers.get_channel_mut(i) };
                    for sample in out_buffer.iter_mut() {
                        *sample = sample.clamp(-1.0, 1.0);
                        if sample.is_nan() {
                            *sample = 0.0;
                        }
                    }
                    let out_port_slice = out_port.as_mut_slice(ps);
                    // out_port_slice.clone_from_slice(out_buffer);
                    for (to_jack, graph_out) in out_port_slice.iter_mut().zip(out_buffer.iter()) {
                        *to_jack = *graph_out as f32;
                    }
                }
                jack::Control::Continue
            }
        }
    }

    struct JackNotifications {
        expected_sample_rate: usize,
    }

    impl JackNotifications {
        fn new(expected_sample_rate: usize) -> Self {
            Self {
                expected_sample_rate,
            }
        }
    }

    fn jack_sample_rate_control(
        expected_sample_rate: usize,
        actual_sample_rate: usize,
    ) -> jack::Control {
        if actual_sample_rate == expected_sample_rate {
            jack::Control::Continue
        } else {
            jack::Control::Quit
        }
    }

    impl jack::NotificationHandler for JackNotifications {
        fn thread_init(&self, _: &jack::Client) {}

        unsafe fn shutdown(&mut self, _status: jack::ClientStatus, _reason: &str) {}

        fn freewheel(&mut self, _: &jack::Client, _is_enabled: bool) {}

        fn sample_rate(&mut self, _: &jack::Client, srate: jack::Frames) -> jack::Control {
            jack_sample_rate_control(self.expected_sample_rate, srate as usize)
        }

        fn client_registration(&mut self, _: &jack::Client, _name: &str, _is_reg: bool) {
            // println!(
            //     "JACK: {} client with name \"{}\"",
            //     if is_reg { "registered" } else { "unregistered" },
            //     name
            // );
        }

        fn port_registration(&mut self, _: &jack::Client, _port_id: jack::PortId, _is_reg: bool) {
            // println!(
            //     "JACK: {} port with id {}",
            //     if is_reg { "registered" } else { "unregistered" },
            //     port_id
            // );
        }

        fn port_rename(
            &mut self,
            _: &jack::Client,
            _port_id: jack::PortId,
            _old_name: &str,
            _new_name: &str,
        ) -> jack::Control {
            // println!(
            //     "JACK: port with id {} renamed from {} to {}",
            //     port_id, old_name, new_name
            // );
            jack::Control::Continue
        }

        fn ports_connected(
            &mut self,
            _: &jack::Client,
            _port_id_a: jack::PortId,
            _port_id_b: jack::PortId,
            _are_connected: bool,
        ) {
            // println!(
            //     "JACK: ports with id {} and {} are {}",
            //     port_id_a,
            //     port_id_b,
            //     if are_connected {
            //         "connected"
            //     } else {
            //         "disconnected"
            //     }
            // );
        }

        fn graph_reorder(&mut self, _: &jack::Client) -> jack::Control {
            // println!("JACK: graph reordered");
            jack::Control::Continue
        }

        fn xrun(&mut self, _: &jack::Client) -> jack::Control {
            // println!("JACK: xrun occurred");
            jack::Control::Continue
        }
    }

    #[cfg(test)]
    mod tests {
        use super::jack_sample_rate_control;

        #[test]
        fn jack_sample_rate_control_continues_on_match() {
            assert!(matches!(
                jack_sample_rate_control(48_000, 48_000),
                jack::Control::Continue
            ));
        }

        #[test]
        fn jack_sample_rate_control_quits_on_change() {
            assert!(matches!(
                jack_sample_rate_control(48_000, 44_100),
                jack::Control::Quit
            ));
        }
    }
}

/// [`AudioBackend`] implementation for CPAL
#[cfg(feature = "cpal")]
pub mod cpal_backend {
    use crate::audio_backend::{AudioBackend, AudioBackendError};
    use crate::controller::Controller;
    use crate::controller::MultiThreadedKnystCommands;
    use crate::graph::{RunGraph, RunGraphSettings};
    use crate::KnystError;
    use crate::Sample;
    use crate::{graph::Graph, Resources};
    #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
    use assert_no_alloc::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::Sample as CpalSampleTrait;

    const CPAL_ISSUE_REPORT_INTERVAL_CALLBACKS: u32 = 128;

    #[derive(Debug)]
    struct CallbackIssueReporter {
        callback_countdown: u32,
        accumulated_samples: u64,
    }

    impl CallbackIssueReporter {
        fn new() -> Self {
            Self {
                callback_countdown: 0,
                accumulated_samples: 0,
            }
        }

        fn record_issue(&mut self, samples: u64) -> Option<usize> {
            if samples == 0 {
                if self.callback_countdown > 0 {
                    self.callback_countdown -= 1;
                }
                return None;
            }

            self.accumulated_samples += samples;
            if self.callback_countdown == 0 {
                self.callback_countdown = CPAL_ISSUE_REPORT_INTERVAL_CALLBACKS;
                let to_report = self.accumulated_samples;
                self.accumulated_samples = 0;
                Some(to_report as usize)
            } else {
                self.callback_countdown -= 1;
                None
            }
        }
    }

    #[allow(missing_docs)]
    pub struct CpalBackendOptions {
        pub device: String,
        pub verbose: bool,
    }

    impl Default for CpalBackendOptions {
        fn default() -> Self {
            Self {
                device: "default".into(),
                verbose: false,
            }
        }
    }

    struct CpalStreams {
        output_stream: cpal::Stream,
        input_stream: Option<cpal::Stream>,
    }

    /// CPAL backend for convenience.
    pub struct CpalBackend {
        streams: Option<CpalStreams>,
        sample_rate: usize,
        config: cpal::SupportedStreamConfig,
        device: cpal::Device,
    }

    impl CpalBackend {
        /// Create a new CpalBackend using the default host, getting a device, but not a stream.
        pub fn new(options: CpalBackendOptions) -> Result<Self, AudioBackendError> {
            let host = cpal::default_host();
            let selected_device = options.device.clone();

            let device = if options.device == "default" {
                host.default_output_device()
            } else {
                host.output_devices()?
                    .find(|x| x.name().map(|y| y == options.device).unwrap_or(false))
            }
            .ok_or(AudioBackendError::OutputDeviceNotFound(selected_device))?;
            if options.verbose {
                println!("Output device: {}", device.name()?);
            }

            let config = device.default_output_config()?;
            if options.verbose {
                println!("Default output config: {:?}", config);
            }
            Ok(Self {
                streams: None,
                sample_rate: config.sample_rate().0 as usize,
                config,
                device,
            })
        }
        /// The number of outputs for the device's default output config
        pub fn num_outputs(&self) -> usize {
            self.config.channels() as usize
        }
    }

    fn validate_cpal_configuration(
        graph_outputs: usize,
        backend_outputs: usize,
        graph_inputs: usize,
        backend_inputs: usize,
        output_sample_rate: usize,
        input_sample_rate: Option<usize>,
    ) -> Result<(), AudioBackendError> {
        if graph_outputs != backend_outputs {
            return Err(AudioBackendError::GraphOutputChannelsMismatch {
                graph_outputs,
                backend_outputs,
            });
        }
        if graph_inputs > backend_inputs {
            return Err(AudioBackendError::GraphInputChannelsMismatch {
                graph_inputs,
                backend_inputs,
            });
        }
        if graph_inputs > 0 {
            if let Some(input_sample_rate) = input_sample_rate {
                if input_sample_rate != output_sample_rate {
                    return Err(AudioBackendError::CpalInputOutputSampleRateMismatch {
                        input_sample_rate,
                        output_sample_rate,
                    });
                }
            }
        }
        Ok(())
    }

    impl AudioBackend for CpalBackend {
        fn start_processing_return_controller(
            &mut self,
            mut graph: Graph,
            resources: Resources,
            run_graph_settings: RunGraphSettings,
            error_handler: Box<dyn FnMut(KnystError) + Send + 'static>,
        ) -> Result<crate::controller::Controller, AudioBackendError> {
            if self.streams.is_some() {
                return Err(AudioBackendError::BackendAlreadyRunning);
            }
            let output_config = self.config.clone();
            let graph_outputs = graph.num_outputs();
            let graph_inputs = graph.num_inputs();
            let backend_outputs = output_config.channels() as usize;
            let input_config = self.device.default_input_config().ok();
            let backend_inputs = input_config
                .as_ref()
                .map(|config| config.channels() as usize)
                .unwrap_or(0);
            let output_sample_rate = output_config.sample_rate().0 as usize;
            let input_sample_rate = input_config
                .as_ref()
                .map(|config| config.sample_rate().0 as usize);

            validate_cpal_configuration(
                graph_outputs,
                backend_outputs,
                graph_inputs,
                backend_inputs,
                output_sample_rate,
                input_sample_rate,
            )?;

            let (mut input_producer, mut input_consumer) = if graph_inputs > 0 {
                let queue_capacity = (graph_inputs * 8_192).next_power_of_two();
                let (producer, consumer) = rtrb::RingBuffer::<Sample>::new(queue_capacity);
                (Some(producer), Some(consumer))
            } else {
                (None, None)
            };

            let stream_config: cpal::StreamConfig = output_config.clone().into();
            let graph_input_channels = graph_inputs;
            let (run_graph, resources_command_sender, resources_command_receiver) =
                RunGraph::new(&mut graph, resources, run_graph_settings)?;
            let controller = Controller::new(
                graph,
                error_handler,
                resources_command_sender,
                resources_command_receiver,
            );
            let error_commands = controller.get_knyst_commands();
            let output_stream = match output_config.sample_format() {
                cpal::SampleFormat::F32 => run_output::<f32>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::I16 => run_output::<i16>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::U16 => run_output::<u16>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::I8 => run_output::<i8>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::I32 => run_output::<i32>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::I64 => run_output::<i64>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::U8 => run_output::<u8>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::U32 => run_output::<u32>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::U64 => run_output::<u64>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                cpal::SampleFormat::F64 => run_output::<f64>(
                    &self.device,
                    &stream_config,
                    run_graph,
                    graph_input_channels,
                    input_consumer.take(),
                    error_commands.clone(),
                ),
                other => Err(AudioBackendError::UnsupportedSampleFormat(other)),
            }?;

            let input_stream = if graph_inputs > 0 {
                let input_config =
                    input_config.expect("input config should exist after validation");
                let input_stream_config: cpal::StreamConfig = input_config.clone().into();
                let input_producer = input_producer
                    .take()
                    .expect("input producer should exist after validation");
                Some(match input_config.sample_format() {
                    cpal::SampleFormat::F32 => run_input::<f32>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::I16 => run_input::<i16>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::U16 => run_input::<u16>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::I8 => run_input::<i8>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::I32 => run_input::<i32>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::I64 => run_input::<i64>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::U8 => run_input::<u8>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::U32 => run_input::<u32>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::U64 => run_input::<u64>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    cpal::SampleFormat::F64 => run_input::<f64>(
                        &self.device,
                        &input_stream_config,
                        input_producer,
                        error_commands.clone(),
                    ),
                    other => Err(AudioBackendError::UnsupportedSampleFormat(other)),
                }?)
            } else {
                None
            };

            self.streams = Some(CpalStreams {
                output_stream,
                input_stream,
            });
            Ok(controller)
        }

        fn stop(&mut self) -> Result<(), AudioBackendError> {
            if let Some(streams) = self.streams.take() {
                let mut pause_error = None;
                if let Some(input_stream) = &streams.input_stream {
                    if let Err(error) = input_stream.pause() {
                        pause_error = Some(error);
                    }
                }
                if let Err(error) = streams.output_stream.pause() {
                    pause_error = Some(error);
                }
                if let Some(error) = pause_error {
                    Err(error.into())
                } else {
                    Ok(())
                }
            } else {
                Err(AudioBackendError::BackendNotRunning)
            }
        }

        fn sample_rate(&self) -> usize {
            self.sample_rate
        }

        fn block_size(&self) -> Option<usize> {
            None
        }

        fn native_output_channels(&self) -> Option<usize> {
            Some(self.num_outputs())
        }

        fn native_input_channels(&self) -> Option<usize> {
            Some(
                self.device
                    .default_input_config()
                    .ok()
                    .map(|config| config.channels() as usize)
                    .unwrap_or(0),
            )
        }
    }

    fn run_output<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        mut run_graph: RunGraph,
        graph_input_channels: usize,
        mut input_consumer: Option<rtrb::Consumer<Sample>>,
        error_commands: MultiThreadedKnystCommands,
    ) -> Result<cpal::Stream, AudioBackendError>
    where
        T: cpal::Sample + cpal::FromSample<Sample> + cpal::SizedSample,
    {
        let channels = config.channels as usize;

        let error_commands_for_output_error = error_commands.clone();
        let err_fn = move |err| {
            error_commands_for_output_error.report_error(AudioBackendError::CpalStreamError(err));
        };

        let mut sample_counter = 0;
        let graph_block_size = run_graph.block_size();
        let mut underflow_reporter = CallbackIssueReporter::new();
        run_graph.run_resources_communication(50);
        run_graph.process_block();
        let stream = device.build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut underflow_report = None;
                let process = || {
                    for frame in output.chunks_mut(channels) {
                        if sample_counter >= graph_block_size {
                            run_graph.run_resources_communication(50);
                            run_graph.process_block();
                            sample_counter = 0;
                        }
                        if graph_input_channels > 0 {
                            let graph_input_buffers = run_graph.graph_input_buffers();
                            let mut missing_samples = 0_u64;
                            for channel_i in 0..graph_input_channels {
                                let value = input_consumer
                                    .as_mut()
                                    .and_then(|consumer| consumer.pop().ok());
                                let value = if let Some(value) = value {
                                    value
                                } else {
                                    missing_samples += 1;
                                    0.0
                                };
                                graph_input_buffers.write(value, channel_i, sample_counter);
                            }
                            if let Some(missing_samples) =
                                underflow_reporter.record_issue(missing_samples)
                            {
                                underflow_report = Some(missing_samples);
                            }
                        }
                        {
                            let buffer = run_graph.graph_output_buffers();
                            for (channel_i, out) in frame.iter_mut().enumerate() {
                                let value: T =
                                    T::from_sample(buffer.read(channel_i, sample_counter));
                                *out = value;
                            }
                        }
                        sample_counter += 1;
                    }
                };
                #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
                assert_no_alloc(process);
                #[cfg(not(all(debug_assertions, feature = "assert_no_alloc")))]
                process();
                if let Some(missing_samples) = underflow_report {
                    error_commands
                        .report_error(AudioBackendError::CpalInputUnderflow { missing_samples });
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        Ok(stream)
    }

    fn run_input<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        mut input_producer: rtrb::Producer<Sample>,
        error_commands: MultiThreadedKnystCommands,
    ) -> Result<cpal::Stream, AudioBackendError>
    where
        T: cpal::Sample + cpal::SizedSample,
        Sample: cpal::FromSample<T>,
    {
        let input_channels = config.channels as usize;
        let error_commands_for_input_error = error_commands.clone();
        let err_fn = move |err| {
            error_commands_for_input_error.report_error(AudioBackendError::CpalStreamError(err));
        };
        let mut overflow_reporter = CallbackIssueReporter::new();
        let stream = device.build_input_stream(
            config,
            move |input: &[T], _: &cpal::InputCallbackInfo| {
                let mut overflow_report = None;
                let process = || {
                    let mut dropped_samples = 0_u64;
                    for frame in input.chunks(input_channels) {
                        for sample in frame {
                            let value = <Sample as CpalSampleTrait>::from_sample(*sample);
                            if input_producer.push(value).is_err() {
                                dropped_samples += 1;
                            }
                        }
                    }
                    if let Some(dropped_samples) = overflow_reporter.record_issue(dropped_samples) {
                        overflow_report = Some(dropped_samples);
                    }
                };
                #[cfg(all(debug_assertions, feature = "assert_no_alloc"))]
                assert_no_alloc(process);
                #[cfg(not(all(debug_assertions, feature = "assert_no_alloc")))]
                process();
                if let Some(dropped_samples) = overflow_report {
                    error_commands
                        .report_error(AudioBackendError::CpalInputOverflow { dropped_samples });
                }
            },
            err_fn,
            None,
        )?;
        stream.play()?;
        Ok(stream)
    }

    #[cfg(test)]
    mod tests {
        use super::{validate_cpal_configuration, CallbackIssueReporter};
        use crate::audio_backend::AudioBackendError;

        #[test]
        fn callback_issue_reporter_batches_reports() {
            let mut reporter = CallbackIssueReporter::new();
            let first = reporter.record_issue(3);
            assert_eq!(first, Some(3));
            // The next issue should be accumulated until the callback budget expires.
            assert_eq!(reporter.record_issue(2), None);
            for _ in 0..127 {
                assert_eq!(reporter.record_issue(0), None);
            }
            assert_eq!(reporter.record_issue(1), Some(3));
        }

        #[test]
        fn callback_issue_reporter_ignores_zero_without_pending_issue() {
            let mut reporter = CallbackIssueReporter::new();
            for _ in 0..256 {
                assert_eq!(reporter.record_issue(0), None);
            }
        }

        #[test]
        fn validate_cpal_configuration_rejects_output_mismatch() {
            let result = validate_cpal_configuration(2, 1, 0, 0, 48_000, None);
            assert!(matches!(
                result,
                Err(AudioBackendError::GraphOutputChannelsMismatch {
                    graph_outputs: 2,
                    backend_outputs: 1
                })
            ));
        }

        #[test]
        fn validate_cpal_configuration_rejects_input_mismatch() {
            let result = validate_cpal_configuration(2, 2, 2, 1, 48_000, Some(48_000));
            assert!(matches!(
                result,
                Err(AudioBackendError::GraphInputChannelsMismatch {
                    graph_inputs: 2,
                    backend_inputs: 1
                })
            ));
        }

        #[test]
        fn validate_cpal_configuration_rejects_sample_rate_mismatch() {
            let result = validate_cpal_configuration(2, 2, 1, 1, 48_000, Some(44_100));
            assert!(matches!(
                result,
                Err(AudioBackendError::CpalInputOutputSampleRateMismatch {
                    input_sample_rate: 44_100,
                    output_sample_rate: 48_000
                })
            ));
        }

        #[test]
        fn validate_cpal_configuration_accepts_valid_setup() {
            let result = validate_cpal_configuration(2, 2, 1, 2, 48_000, Some(48_000));
            assert!(result.is_ok());
        }
    }
}
