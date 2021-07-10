mod wgpu_renderer;
use crate::wgpu_renderer::WgpuRenderer;
use futures::executor::block_on;
use std::time::Instant;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;

pub struct GameState {
    player_position: cgmath::Vector2<f32>,
    micros_from_start: u128,
}

fn main() {
    let (event_loop, window, mut renderer) = block_on(WgpuRenderer::init());
    let start = Instant::now();
    let mut last_micros: u128 = 0;

    let profiler = init_profiler();
    let mut game_state = GameState {
        player_position: cgmath::Vector2::<_>::new(0., 0.),
        micros_from_start: 0,
    };

    event_loop.run(move |event, _, control_flow| {
        update_profiler(&profiler);

        let now_micros = start.elapsed().as_micros();
        let delta_micros = now_micros - last_micros;

        let event_results = read_events(event, &window);

        update(&mut game_state, delta_micros);

        let render_result = render(&mut renderer, &event_results, &game_state);

        *control_flow = resolve_frame(&event_results, render_result);

        last_micros = now_micros;
    });
}

struct EventResults {
    should_render: bool,
    should_exit: bool,
}

fn init_profiler() -> puffin_http::Server {
    let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
    let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    puffin::set_scopes_on(true);

    eprintln!("Serving demo profile data on {}", server_addr);

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

    match event_handling.should_exit {
        true => ControlFlow::Exit,
        false => match render_result {
            Ok(_) => ControlFlow::Poll,
            // Recreate the swap_chain if lost
            //Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
            Err(wgpu::SwapChainError::OutOfMemory) => ControlFlow::Exit,
            Err(e) => {
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                eprintln!("{:?}", e);
                ControlFlow::Poll
            }
        },
    }
}

fn read_events<T>(event: Event<'_, T>, window: &Window) -> EventResults {
    puffin::profile_function!();

    let mut event_results = EventResults {
        should_render: false,
        should_exit: false,
    };

    match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            match event {
                WindowEvent::CloseRequested => event_results.should_exit = true,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    } => event_results.should_exit = true,
                    _ => {}
                },
                /*WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }*/
                _ => {}
            }
        }
        Event::RedrawRequested(_) => {
            event_results.should_render = true;
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            window.request_redraw();
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
    game_state.player_position = cgmath::Vector2::<_>::new(x, y);
}

fn render(
    renderer: &mut WgpuRenderer,
    event_handling: &EventResults,
    game_state: &GameState,
) -> Result<(), wgpu::SwapChainError> {
    puffin::profile_function!();

    match event_handling.should_render {
        true => renderer.render(&game_state),
        false => Ok(()),
    }
}
