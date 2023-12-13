use anyhow::Result;
#[allow(unused)]
use knyst::{
    audio_backend::{CpalBackend, CpalBackendOptions, JackBackend},
    controller::print_error_handler,
    envelope::Envelope,
    handles::{graph_output, handle, Handle},
    modal_interface::knyst,
    prelude::{delay::static_sample_delay, *},
    sphere::{KnystSphere, SphereSettings},
};
use knyst_reverb::galactic::galactic;
use rand::{thread_rng, Rng};
fn main() -> Result<()> {
    // let mut backend = CpalBackend::new(CpalBackendOptions::default())?;
    let mut backend = JackBackend::new("Knyst<3JACK")?;
    let _sphere = KnystSphere::start(
        &mut backend,
        SphereSettings {
            num_inputs: 0,
            num_outputs: 2,
            ..Default::default()
        },
        print_error_handler,
    );

    let reverb = galactic()
        .size(0.7)
        .brightness(0.9)
        .detune(0.2)
        .mix(0.2)
        .replace(0.3);
    graph_output(0, reverb);
    let outer_graph = upload_graph(knyst().default_graph_settings(), || {});
    outer_graph.activate();
    reverb.left(outer_graph);
    reverb.right(outer_graph);

    let mut rng = thread_rng();
    let freq_var = bus(1).set(0, 440.);
    for _ in 0..10 {
        let freq = (sine().freq(
            sine()
                .freq(
                    sine()
                        .freq(0.01)
                        .range(0.02, rng.gen_range(0.05..0.3 as Sample)),
                )
                .range(0.0, 400.),
        ) * 100.0)
            + freq_var;
        // let freq = sine().freq(0.5).range(200.0, 200.0 * 9.0 / 8.0);
        let node0 = sine();
        node0.freq(freq);
        let modulator = sine();
        modulator.freq(sine().freq(0.09) * -5.0 + 6.0);
        graph_output(0, (node0 * modulator * 0.025).repeat_outputs(1));
    }

    std::thread::spawn(move || {
        // We're on a new thread so we have to activate the graph we are targeting again.
        outer_graph.activate();
        for &ratio in [1.0, 1.5, 5. / 4.].iter().cycle() {
            // new graph
            let graph = upload_graph(knyst().default_graph_settings().num_inputs(1), || {
                // Since freq_var is in a different graph we can pipe it in via a graph input
                let freq_var = graph_input(0, 1);
                let sig = sine().freq(freq_var * ratio).out("sig") * 0.25;
                let env = Envelope {
                    points: vec![(1.0, 0.005), (0.0, 0.5)],
                    stop_action: StopAction::FreeGraph,
                    ..Default::default()
                };
                let sig = sig * handle(env.to_gen());
                // let sig = sig * handle(env.to_gen());
                graph_output(0, sig.repeat_outputs(1));
            });
            // Make sure we also pass the freq_var signal in
            graph.set(0, freq_var);
            outer_graph.activate();
            // Add the direct signal of the graph together with a delay
            let sig = graph + static_sample_delay(48 * 500).input(graph.out(0));
            // Output to the outer graph
            graph_output(0, sig.repeat_outputs(1));
            std::thread::sleep(std::time::Duration::from_millis(2500));
        }
    });

    let mut input = String::new();
    loop {
        println!("Input a frequency for the root note, or 'q' to quit: ");
        match std::io::stdin().read_line(&mut input) {
            Ok(_n) => {
                let input = input.trim();
                if let Ok(freq) = input.parse::<f32>() {
                    println!("New freq: {}", input.trim());
                    freq_var.set(0, freq as Sample);
                } else if input == "q" {
                    break;
                }
            }
            Err(error) => println!("error: {}", error),
        }
        input.clear();
    }
    Ok(())
}

fn sine() -> Handle<OscillatorHandle> {
    oscillator(WavetableId::cos())
}
