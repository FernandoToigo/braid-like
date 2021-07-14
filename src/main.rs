mod wgpu_renderer;
use crate::wgpu_renderer::WgpuRenderer;
use cgmath::Vector2;
use futures::executor::block_on;
use std::time::Instant;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

const UPDATE_INTERVAL_MICROS: u128 = 33333;

pub struct Game {
    renderer: WgpuRenderer,
    start_instant: Instant,
    last_frame_micros: u128,
    last_update_micros: u128,
    game_state: GameState,
    profiler: puffin_http::Server,
}

pub struct GameState {
    micros_from_start: u128,
    previous_player_position: cgmath::Vector2<f32>,
    player_position: cgmath::Vector2<f32>,
    camera_position: cgmath::Vector2<f32>,
    camera_orthographic_height: f32,
}

struct EventResults {
    events_finished: bool,
    exit_requested: bool,
}

fn main() {
    let (event_loop, renderer) = block_on(WgpuRenderer::init());

    let profiler = init_profiler();

    let game_state = GameState {
        micros_from_start: 0,
        previous_player_position: Vector2::<_>::new(0., 0.),
        player_position: Vector2::<_>::new(0., 0.),
        camera_position: Vector2::<_>::new(0., 0.),
        camera_orthographic_height: 10.,
    };

    let mut game = Game {
        renderer,
        start_instant: Instant::now(),
        last_frame_micros: 0,
        last_update_micros: 0,
        game_state,
        profiler,
    };

    event_loop.run(move |event, _, control_flow| {
        let event_results = read_events(event, &mut game.renderer);

        let frame_result = match event_results.events_finished {
            true => frame(&mut game),
            false => Ok(()),
        };

        *control_flow = resolve_frame(&event_results, frame_result);
    });
}

fn frame(game: &mut Game) -> anyhow::Result<(), wgpu::SwapChainError> {
    puffin::profile_function!();

    let now_micros = game.start_instant.elapsed().as_micros();

    update_profiler(&game.profiler);

    // TODO: add maximum update steps to recover
    while now_micros - game.last_update_micros > UPDATE_INTERVAL_MICROS {
        update(&mut game.game_state, UPDATE_INTERVAL_MICROS);
        game.last_update_micros += UPDATE_INTERVAL_MICROS;
    }
    let interp_percent =
        ((now_micros - game.last_update_micros) as f32) / UPDATE_INTERVAL_MICROS as f32;
    render(&mut game.renderer, &game.game_state, interp_percent)?;

    game.last_frame_micros = now_micros;

    Ok(())
}

fn init_profiler() -> puffin_http::Server {
    let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
    let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    puffin::set_scopes_on(true);

    println!(
        "Serving profile data on {}. Use puffin_viewer to see it.",
        server_addr
    );

    puffin_server
}

fn update_profiler(puffin_server: &puffin_http::Server) {
    puffin::GlobalProfiler::lock().new_frame();
    puffin_server.update();
}

fn resolve_frame(
    event_handling: &EventResults,
    render_result: Result<(), wgpu::SwapChainError>,
) -> ControlFlow {
    puffin::profile_function!();

    match event_handling.exit_requested {
        true => ControlFlow::Exit,
        false => match render_result {
            Ok(_) => ControlFlow::Poll,
            // Recreate the swap_chain if lost
            //Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
            Err(wgpu::SwapChainError::OutOfMemory) => ControlFlow::Exit,
            Err(e) => {
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                eprintln!("Unexpected error: {:?}", e);
                ControlFlow::Poll
            }
        },
    }
}

fn read_events<T>(event: Event<'_, T>, renderer: &mut WgpuRenderer) -> EventResults {
    puffin::profile_function!();

    let mut event_results = EventResults {
        events_finished: false,
        exit_requested: false,
    };

    match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == renderer.window.id() => match event {
            WindowEvent::CloseRequested => event_results.exit_requested = true,
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::Escape),
                    ..
                } => event_results.exit_requested = true,
                _ => {}
            },
            WindowEvent::Resized(physical_size) => {
                renderer.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                renderer.resize(**new_inner_size);
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            event_results.events_finished = true;
            renderer.window.request_redraw();
        }
        _ => {}
    }

    event_results
}

fn update(game_state: &mut GameState, delta_micros: u128) {
    puffin::profile_function!();

    game_state.micros_from_start += delta_micros;

    let radians_per_second = std::f32::consts::PI * 0.5;
    let radians_per_micros = radians_per_second / 1e+6;
    let x = (game_state.micros_from_start as f32 * radians_per_micros).cos();
    let y = (game_state.micros_from_start as f32 * radians_per_micros).sin();

    game_state.previous_player_position = game_state.player_position;
    game_state.player_position.x = x;
    game_state.player_position.y = y;
    game_state.camera_orthographic_height = 10.;
}

fn render(
    renderer: &mut WgpuRenderer,
    game_state: &GameState,
    interp_percent: f32,
) -> Result<(), wgpu::SwapChainError> {
    puffin::profile_function!();
    renderer.render(&game_state, interp_percent)
}
