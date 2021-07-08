mod wgpu_renderer;
use crate::wgpu_renderer::WgpuRenderer;
use futures::executor::block_on;
use std::time::Instant;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;

fn main() {
    let (event_loop, window, renderer) = block_on(WgpuRenderer::init());
    let start = Instant::now();
    let mut last_micros: u128 = 0;

    println!("Started.");

    event_loop.run(move |event, _, control_flow| {
        let now_micros = start.elapsed().as_micros();
        let event_results = read_events(event, &window);
        update();
        let render_result = render(&renderer, &event_results);
        *control_flow = resolve_frame(&event_results, render_result);
        println!(
            "DeltaTime= {}mil Render={})",
            ((now_micros - last_micros) as f32) / 1000.,
            event_results.should_render
        );
        last_micros = now_micros;
    });
}

struct EventResults {
    should_render: bool,
    should_exit: bool,
}

fn resolve_frame(
    event_handling: &EventResults,
    draw_result: Result<(), wgpu::SwapChainError>,
) -> ControlFlow {
    match event_handling.should_exit {
        true => ControlFlow::Exit,
        false => match draw_result {
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

fn update() {}

fn render(
    renderer: &WgpuRenderer,
    event_handling: &EventResults,
) -> Result<(), wgpu::SwapChainError> {
    match event_handling.should_render {
        true => renderer.render(),
        false => Ok(()),
    }
}
