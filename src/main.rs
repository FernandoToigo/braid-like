mod physics;
mod wgpu_renderer;
use crate::physics::{init_physics, Physics};
use crate::wgpu_renderer::{InstanceRaw, WgpuRenderer};
use cgmath::{Vector2, Vector3};
use futures::executor::block_on;
use rapier2d::prelude::*;
use std::time::Instant;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

const UPDATE_INTERVAL_MICROS: u128 = 33333;
const MAXIMUM_UPDATE_STEPS_PER_FRAME: u32 = 10;

pub struct Game {
    renderer: WgpuRenderer,
    physics: Physics,
    start_instant: Instant,
    last_frame_micros: u128,
    frame_count: u128,
    last_update_micros: u128,
    state: GameState,
    profiler: puffin_http::Server,
}

pub struct GameState {
    player: Player,
    camera: Camera,
}

struct EventResults {
    events_finished: bool,
    exit_requested: bool,
}

#[derive(Default)]
struct Input {
    right: bool,
    left: bool,
    jump: bool,
}

struct Player {
    position: cgmath::Vector3<f32>,
    last_position: cgmath::Vector3<f32>,
    texture_index: u32,
    rigid_body_handle: rapier2d::dynamics::RigidBodyHandle,
}

struct Camera {
    position: Vector3<f32>,
    last_position: Vector3<f32>,
    orthographic_height: f32,
}

struct Wall {
    position: cgmath::Vector3<f32>,
    size: cgmath::Vector2<f32>,
}

fn main() {
    env_logger::init();

    let profiler = init_profiler();
    let mut physics = init_physics(UPDATE_INTERVAL_MICROS);

    let walls = vec![
        Wall {
            position: Vector3::new(0., -0.5, 0.),
            size: Vector2::new(20., 1.),
        },
        Wall {
            position: Vector3::new(1.75, 1., 0.),
            size: Vector2::new(1., 1.),
        },
        Wall {
            position: Vector3::new(-14.5, -0.5, 0.),
            size: Vector2::new(2., 1.),
        },
    ];
    let state = create_game_state(&mut physics, &walls);
    let instances = create_instances(&state, &walls);
    let (event_loop, renderer) = block_on(WgpuRenderer::init(instances));

    let mut game = Game {
        renderer,
        physics,
        start_instant: Instant::now(),
        last_frame_micros: 0,
        frame_count: 0,
        last_update_micros: 0,
        state,
        profiler,
    };

    let mut input = Input {
        ..Default::default()
    };

    event_loop.run(move |event, _, control_flow| {
        let event_results = read_events(event, &mut game.renderer, &mut input);

        let frame_result = match event_results.events_finished {
            true => frame(&mut game, &mut input),
            false => Ok(()),
        };

        *control_flow = resolve_frame(&event_results, frame_result);
    });
}

fn create_game_state(physics: &mut Physics, walls: &Vec<Wall>) -> GameState {
    let player_rigid_body = rapier2d::dynamics::RigidBodyBuilder::new_dynamic()
        .lock_rotations()
        .build();
    let player_rigid_body_handle = physics.rigid_bodies.insert(player_rigid_body);

    let collider = ColliderBuilder::cuboid(0.25, 0.5)
        .friction(0.15)
        .collision_groups(InteractionGroups::new(0b1, 0xFFFF))
        .build();
    physics.colliders.insert_with_parent(
        collider,
        player_rigid_body_handle,
        &mut physics.rigid_bodies,
    );

    walls
        .iter()
        .map(|wall| {
            ColliderBuilder::cuboid(wall.size.x * 0.5, wall.size.y * 0.5)
                .translation(vector!(wall.position.x, wall.position.y))
                .collision_groups(InteractionGroups::new(0b10, 0xFFFF))
                .build()
        })
        .for_each(|wall_collider| {
            physics.colliders.insert(wall_collider);
        });

    GameState {
        player: Player {
            position: Vector3::new(0., 0.5, -1.),
            last_position: Vector3::new(0., 0., -1.),
            rigid_body_handle: player_rigid_body_handle,
            texture_index: 0,
        },
        camera: Camera {
            position: Vector3::new(0., 0., 10.),
            last_position: Vector3::new(0., 0., 10.),
            orthographic_height: 10.,
        },
    }
}

fn create_instances(state: &GameState, walls: &Vec<Wall>) -> Vec<InstanceRaw> {
    let mut instances = Vec::new();
    instances.push(InstanceRaw::new(
        state.player.position,
        Vector3::new(1., 1., 1.),
        state.player.texture_index,
    ));

    walls
        .iter()
        .for_each(|wall| instances.push(InstanceRaw::new(wall.position, wall.size.extend(1.), 1)));

    instances.push(InstanceRaw::new(
        Vector3::new(0., 0., 0.),
        Vector3::new(1., 1., 1.),
        2,
    ));

    instances
}

fn frame(game: &mut Game, input: &mut Input) -> anyhow::Result<(), wgpu::SurfaceError> {
    puffin::profile_function!();

    let now_micros = game.start_instant.elapsed().as_micros();

    update_profiler(&game.profiler);

    let mut update_count = 0;
    while now_micros - game.last_update_micros > UPDATE_INTERVAL_MICROS {
        update(game, &input);
        game.last_update_micros += UPDATE_INTERVAL_MICROS;
        update_count += 1;
        if update_count >= MAXIMUM_UPDATE_STEPS_PER_FRAME {
            println!("WARNING. Game is slowing down.");
            break;
        }
    }
    render(game, now_micros)?;

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
    render_result: Result<(), wgpu::SurfaceError>,
) -> ControlFlow {
    puffin::profile_function!();

    match event_handling.exit_requested {
        true => ControlFlow::Exit,
        false => match render_result {
            Ok(_) => ControlFlow::Poll,
            // Recreate the swap_chain if lost
            //Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
            Err(wgpu::SurfaceError::OutOfMemory) => ControlFlow::Exit,
            Err(e) => {
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                eprintln!("Unexpected error: {:?}", e);
                ControlFlow::Poll
            }
        },
    }
}

fn read_events<T>(
    event: Event<'_, T>,
    renderer: &mut WgpuRenderer,
    input: &mut Input,
) -> EventResults {
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
            WindowEvent::KeyboardInput {
                input: keyboard_input,
                ..
            } => match keyboard_input {
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::Escape),
                    ..
                } => event_results.exit_requested = true,
                KeyboardInput {
                    state,
                    virtual_keycode: Some(VirtualKeyCode::A),
                    ..
                } => input.left = is_pressed(state),
                KeyboardInput {
                    state,
                    virtual_keycode: Some(VirtualKeyCode::D),
                    ..
                } => input.right = is_pressed(state),
                KeyboardInput {
                    state,
                    virtual_keycode: Some(VirtualKeyCode::Space),
                    ..
                } => input.jump = is_pressed(state),
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

fn is_pressed(state: &ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        _ => false,
    }
}

fn update(game: &mut Game, input: &Input) {
    puffin::profile_function!();

    game.frame_count += 1;

    update_player(game, input);
    update_physics(game);
    update_camera(game);
}

fn update_camera(game: &mut Game) {
    game.state.camera.orthographic_height = 10.;
    game.state.camera.last_position = game.state.camera.position;
    game.state.camera.position.x = game.state.player.position.x;
    game.state.camera.position.y = game.state.player.position.y;
}

fn update_player(game: &mut Game, input: &Input) {
    const JUMP_HEIGHT: f32 = 2.;
    const JUMP_HORIZONTAL_VELOCITY_PER_SECOND: f32 = 4.;
    const JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE: f32 = 2.;
    const JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING: f32 = 1.75;

    let player_rigid_body = &mut game.physics.rigid_bodies[game.state.player.rigid_body_handle];

    let mut velocity = {
        let current_velocity = player_rigid_body.linvel();
        Vector2::new(current_velocity.x, current_velocity.y)
    };

    if input.jump && is_player_grounded(game) {
        velocity.y = (2. * JUMP_HEIGHT * JUMP_HORIZONTAL_VELOCITY_PER_SECOND)
            / JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE;
    }

    if input.left {
        velocity.x = -JUMP_HORIZONTAL_VELOCITY_PER_SECOND;
    }
    if input.right {
        velocity.x = JUMP_HORIZONTAL_VELOCITY_PER_SECOND;
    }

    let player_rigid_body = &mut game.physics.rigid_bodies[game.state.player.rigid_body_handle];
    let travel_distance = match velocity.y > 0. {
        true => JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE,
        false => JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING,
    };
    let gravity = (-2.
        * JUMP_HEIGHT
        * (JUMP_HORIZONTAL_VELOCITY_PER_SECOND * JUMP_HORIZONTAL_VELOCITY_PER_SECOND))
        / (travel_distance * travel_distance);
    velocity.y += gravity * to_seconds(UPDATE_INTERVAL_MICROS);
    player_rigid_body.set_linvel(vector![velocity.x, velocity.y], true);
}

fn is_player_grounded(game: &mut Game) -> bool {
    let ray = Ray::new(
        point!(game.state.player.position.x, game.state.player.position.y),
        vector![0., -1.],
    );

    game.physics
        .cast_ray(ray, 0.51, InteractionGroups::new(0b10, 0b10))
        .is_some()
}

fn update_physics(game: &mut Game) {
    game.physics.run_frame();

    let player_rigid_body = &game.physics.rigid_bodies[game.state.player.rigid_body_handle];
    let player_position_from_physics = player_rigid_body.translation();
    game.state.player.last_position = game.state.player.position;
    game.state.player.position.x = player_position_from_physics.x;
    game.state.player.position.y = player_position_from_physics.y;
    /*println!(
        "position: [{} ({}secs)] ({};{})",
        game.frame_count,
        to_seconds(game.last_update_micros),
        game.state.player.position.x,
        game.state.player.position.y - 0.5
    );*/
}

fn to_seconds(micros: u128) -> f32 {
    micros as f32 * 1e-6
}

fn render(game: &mut Game, now_micros: u128) -> Result<(), wgpu::SurfaceError> {
    puffin::profile_function!();

    let micros_since_last_updated = (now_micros - game.last_update_micros) as f32;
    let interp_percent = micros_since_last_updated / UPDATE_INTERVAL_MICROS as f32;

    game.renderer.render(&game.state, interp_percent)
}

impl Input {
    fn _has_any_input(&self) -> bool {
        self.left || self.right || self.jump
    }
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[{}] [{}] [{}]",
            bool_to_x(self.left, "____", "LEFT"),
            bool_to_x(self.jump, "____", "JUMP"),
            bool_to_x(self.right, "_____", "RIGHT"),
        )
    }
}

fn bool_to_x(value: bool, false_value: &'static str, true_value: &'static str) -> &'static str {
    match value {
        true => true_value,
        _ => false_value,
    }
}
