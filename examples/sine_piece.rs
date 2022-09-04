use std::time::Duration;

use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions},
    envelope::{Curve, Envelope},
    graph::{GenState, Mult},
    prelude::*,
    wavetable::{Wavetable, WavetableOscillatorOwned},
};
use rand::{prelude::SliceRandom, thread_rng, Rng};
fn main() {
    let mut backend = CpalBackend::new(CpalBackendOptions::default()).unwrap();

    let sample_rate = backend.sample_rate() as f32;
    let block_size = backend.block_size().unwrap_or(64);
    let resources = Resources::new(sample_rate);
    let graph_settings = GraphSettings {
        block_size,
        sample_rate,
        latency: Duration::from_millis(100),
        num_outputs: backend.num_outputs(),
        ..Default::default()
    };
    let mut graph: Graph = Graph::new(graph_settings);
    backend.start_processing(&mut graph, resources).unwrap();

    let mut rng = thread_rng();
    let chord = vec![1.0, 5. / 4., 3. / 2., 7. / 4., 2.0, 17. / 8.];
    let mut fundamental = 440.0;
    loop {
        for _ in 0..32 {
            let attack = 0.5;
            let freq = chord.choose(&mut rng).unwrap() * fundamental;
            add_sine(freq, attack, 1.0, graph_settings, &mut graph);
            add_sine(freq * 0.999, attack, 0.3, graph_settings, &mut graph);
            add_sine(freq * 1.001, attack, 0.5, graph_settings, &mut graph);
            std::thread::sleep(Duration::from_secs_f32(0.5));
        }
        for _ in 0..4 {
            let attack = 0.02;
            let freq = chord.choose(&mut rng).unwrap() * fundamental;
            add_sine(freq, attack, 2.0, graph_settings, &mut graph);
            let freq = chord.choose(&mut rng).unwrap() * fundamental * 0.125;
            add_sine(freq, 5.0, 10.0, graph_settings, &mut graph);
            std::thread::sleep(Duration::from_secs_f32(0.5));
            for _ in 0..15 {
                let attack = rng.gen::<f32>().powi(2) * 0.25 + 0.001;
                let freq = chord.choose(&mut rng).unwrap() * fundamental;
                add_sine(freq, attack, 1.0, graph_settings, &mut graph);
                add_sine(freq * 0.999, attack, 0.3, graph_settings, &mut graph);
                add_sine(freq * 1.001, attack, 0.5, graph_settings, &mut graph);
                std::thread::sleep(Duration::from_secs_f32(0.25));
            }
        }
        fundamental *= chord.choose(&mut rng).unwrap();
        while fundamental > 880. {
            fundamental *= 0.5;
        }
    }
}

fn sine_tone_graph(
    freq: f32,
    attack: f32,
    duration_secs: f32,
    graph_settings: GraphSettings,
) -> Graph {
    let mut g = Graph::new(graph_settings);
    let sin = g.push_gen(WavetableOscillatorOwned::new(Wavetable::sine()));
    g.connect(constl(freq, "freq").to_node(sin)).unwrap();
    let mut rng = thread_rng();
    let env = Envelope {
        points: vec![(0.1, attack), (0.0, duration_secs)],
        curves: vec![Curve::Linear, Curve::Exponential(2.0)],
        // TODO: Shouldnt include the sample number here
        stop_action: StopAction::FreeGraph,
        ..Default::default()
    };
    let mut env = env.to_gen();
    // TODO: It is very unintuitive that you have to manually start the envelope
    env.start();
    let env = g.push_gen(env);
    let mult = g.push_gen(Mult);
    g.connect(sin.to(mult)).unwrap();
    g.connect(env.to(mult).to_index(1)).unwrap();
    g.connect(Connection::out(mult)).unwrap();
    g.connect(Connection::out(mult).to_index(1)).unwrap();
    g.commit_changes();
    g
}

fn add_sine(
    freq: f32,
    attack: f32,
    duration_secs: f32,
    graph_settings: GraphSettings,
    main_graph: &mut Graph,
) {
    let node = main_graph.push_graph(sine_tone_graph(freq, attack, duration_secs, graph_settings));
    main_graph.connect(Connection::out(node));
    let node = main_graph.push_graph(sine_tone_graph(
        freq * 1.001,
        attack,
        duration_secs,
        graph_settings,
    ));
    main_graph.connect(Connection::out(node).to_index(1));
    main_graph.commit_changes();
    main_graph.update();
}
