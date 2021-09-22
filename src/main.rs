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

const DELTA_MICROS: u128 = 33333;
const DELTA_SECONDS: f32 = DELTA_MICROS as f32 * 1e-6;
const MAXIMUM_UPDATE_STEPS_PER_FRAME: u32 = 10;

pub struct Game {
    renderer: WgpuRenderer,
    start_instant: Instant,
    last_frame_micros: u128,
    last_update_micros: u128,
    scenario: Scenario,
    state: GameState,
    profiler: puffin_http::Server,
    inputs: Vec<InputRegistry>,
    is_replaying: bool,
    replay_registry_index: usize,
    replay_registry_count: u128,
}

pub struct Scenario {
    player_start_position: Vector3<f32>,
    walls: Vec<Wall>,
}

pub struct GameState {
    frame_count: u128,
    physics: Physics,
    player: Player,
    camera: Camera,
}

struct InputRegistry {
    input: Input,
    count: u128,
}

struct EventResults {
    events_finished: bool,
    exit_requested: bool,
}

#[derive(Default, PartialEq, Clone)]
struct Input {
    right: bool,
    left: bool,
    jump: bool,
    confirm: bool,
    debug_one: bool,
}

struct Player {
    position: Vector3<f32>,
    last_position: Vector3<f32>,
    texture_offset: Vector2<f32>,
    texture_index: u32,
    rigid_body_handle: rapier2d::dynamics::RigidBodyHandle,
    force_jumping: bool,
    velocity: Vector2<f32>,
    applied_velocity: Vector2<f32>,
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
    let physics = init_physics(DELTA_SECONDS);

    let scenario = create_first_scenario();
    let state = create_game_state(physics, &scenario);
    let instances = create_instances(&state, &scenario.walls);
    let (event_loop, renderer) = block_on(WgpuRenderer::init(instances));

    let mut game = Game {
        renderer,
        start_instant: Instant::now(),
        last_frame_micros: 0,
        last_update_micros: 0,
        scenario,
        state,
        profiler,
        inputs: Vec::with_capacity(9000),
        is_replaying: false,
        replay_registry_index: 0,
        replay_registry_count: 0,
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

fn create_first_scenario() -> Scenario {
    Scenario {
        player_start_position: Vector3::new(0., 0.0, -1.),
        walls: vec![
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
        ],
    }
}

fn create_game_state(mut physics: Physics, scenario: &Scenario) -> GameState {
    let player_rigid_body = rapier2d::dynamics::RigidBodyBuilder::new_dynamic()
        .translation(vector![
            scenario.player_start_position.x,
            scenario.player_start_position.y
        ])
        .lock_rotations()
        .build();
    let player_rigid_body_handle = physics.rigid_bodies.insert(player_rigid_body);

    let collider = ColliderBuilder::cuboid(0.25, 0.5)
        .translation(vector![0., 0.5])
        .friction(0.15)
        .collision_groups(InteractionGroups::new(0b1, 0xFFFF))
        .build();
    physics.colliders.insert_with_parent(
        collider,
        player_rigid_body_handle,
        &mut physics.rigid_bodies,
    );

    scenario
        .walls
        .iter()
        .map(|wall| {
            ColliderBuilder::cuboid(wall.size.x * 0.5, wall.size.y * 0.5)
                .translation(vector!(wall.position.x, wall.position.y))
                .friction(0.15)
                .collision_groups(InteractionGroups::new(0b10, 0xFFFF))
                .build()
        })
        .for_each(|wall_collider| {
            physics.colliders.insert(wall_collider);
        });

    GameState {
        frame_count: 0,
        physics,
        player: Player {
            position: scenario.player_start_position,
            last_position: scenario.player_start_position,
            rigid_body_handle: player_rigid_body_handle,
            texture_offset: Vector2::new(0., 0.5),
            texture_index: 0,
            force_jumping: false,
            velocity: Vector2::new(0., 0.),
            applied_velocity: Vector2::new(0., 0.),
        },
        camera: Camera {
            position: Vector3::new(0., 0., 10.),
            last_position: Vector3::new(0., 0., 10.),
            orthographic_height: 10.,
        },
    }
}

fn reset_state(game: &mut Game) {
    game.state.player.position = game.scenario.player_start_position;
    game.state.player.last_position = game.scenario.player_start_position;
    let player_rigid_body =
        &mut game.state.physics.rigid_bodies[game.state.player.rigid_body_handle];
    player_rigid_body.set_linvel(vector![0.0, 0.0], true);
    player_rigid_body.set_translation(
        vector![
            game.scenario.player_start_position.x,
            game.scenario.player_start_position.y
        ],
        true,
    );
}

fn create_instances(state: &GameState, walls: &Vec<Wall>) -> Vec<InstanceRaw> {
    let mut instances = Vec::new();
    instances.push(InstanceRaw::new(
        state.player.position + state.player.texture_offset.extend(0.),
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

    if input.confirm && !game.is_replaying {
        reset_state(game);
        game.is_replaying = true;
        game.replay_registry_index = 0;
        game.replay_registry_count = 0;
    }

    let now_micros = game.start_instant.elapsed().as_micros();

    update_profiler(&game.profiler);

    let mut update_count = 0;
    while now_micros - game.last_update_micros > DELTA_MICROS {
        if game.is_replaying {
            match get_next_replay_input(game) {
                Some(new_input) => *input = new_input,
                None => {
                    *input = Input {
                        ..Default::default()
                    }
                }
            };
        } else {
            store_input(game, input.clone());
        }

        update(&mut game.state, &input);
        game.last_update_micros += DELTA_MICROS;
        update_count += 1;
        if update_count >= MAXIMUM_UPDATE_STEPS_PER_FRAME {
            println!("WARNING. Game is slowing down.");
            break;
        }

        let (index, count, input_log) = match game.is_replaying {
            true => (
                game.replay_registry_index,
                game.replay_registry_count,
                game.inputs[game.replay_registry_index].input.clone(),
            ),
            false => {
                let index = game.inputs.len() - 1 as usize;
                (index, game.inputs[index].count, input.clone())
            }
        };

        let player_rigid_body =
            &game.state.physics.rigid_bodies[game.state.player.rigid_body_handle];
        println!(
            "[{} ({}s)] P({};{}) ->'({};{}) ->({};{}) ([{}]:{}:{})",
            game.state.frame_count,
            micros_to_seconds(game.last_update_micros),
            game.state.player.position.x,
            game.state.player.position.y,
            game.state.player.applied_velocity.x,
            game.state.player.applied_velocity.y,
            player_rigid_body.linvel().x,
            player_rigid_body.linvel().y,
            index,
            count,
            input_log,
        );
    }
    render(game, now_micros)?;

    game.last_frame_micros = now_micros;

    Ok(())
}

fn micros_to_seconds(micros: u128) -> f32 {
    micros as f32 * 1e-6
}

fn get_next_replay_input(game: &mut Game) -> Option<Input> {
    if game.replay_registry_index >= game.inputs.len() {
        return None;
    }

    game.replay_registry_count += 1;

    if game.replay_registry_count > game.inputs[game.replay_registry_index].count {
        game.replay_registry_index += 1;
        game.replay_registry_count = 1;
    }

    if game.replay_registry_index >= game.inputs.len() {
        return None;
    }

    Some(game.inputs[game.replay_registry_index].input.clone())
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
                KeyboardInput {
                    state,
                    virtual_keycode: Some(VirtualKeyCode::Return),
                    ..
                } => input.confirm = is_pressed(state),
                KeyboardInput {
                    state,
                    virtual_keycode: Some(VirtualKeyCode::F1),
                    ..
                } => input.debug_one = is_pressed(state),
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

fn update(state: &mut GameState, input: &Input) {
    puffin::profile_function!();

    update_player(state, input);
    update_physics(state);
    update_camera(state);

    state.frame_count += 1;
}

fn store_input(game: &mut Game, input: Input) {
    if game.inputs.len() == 0 {
        game.inputs.push(InputRegistry {
            input: input.clone(),
            count: 1,
        });
    } else {
        let last_input_index = game.inputs.len() - 1;
        if game.inputs[last_input_index].input == input {
            game.inputs[last_input_index].count += 1;
        } else {
            game.inputs.push(InputRegistry { input, count: 1 });
        }
    }
}

fn update_camera(state: &mut GameState) {
    state.camera.orthographic_height = 10.;
    /*game.state.camera.last_position = game.state.camera.position;
    game.state.camera.position.x = game.state.player.position.x;
    game.state.camera.position.y = game.state.player.position.y + 1.;*/
}

const HORIZONTAL_ACCELERATION_PER_SECOND: f32 = 16.;
const JUMP_HORIZONTAL_ACCELERATION_PER_SECOND: f32 = 8.;
const MAX_HORIZONTAL_VELOCITY_PER_SECOND: f32 = 4.;
const JUMP_HEIGHT: f32 = 2.0;
const JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE: f32 = 1.8;
const JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING: f32 = 1.3;

fn update_player(state: &mut GameState, input: &Input) {
    let is_grounded = is_player_grounded(state);
    let player = &mut state.player;

    let sign = match (input.left, input.right) {
        (true, false) => -1.,
        (false, true) => 1.,
        _ => 0.,
    };

    let mut acceleration = Vector2::new(0., 0.);
    let horizontal_acceleration = get_horizontal_acceleration(is_grounded);
    if player.velocity.x > -MAX_HORIZONTAL_VELOCITY_PER_SECOND
        && player.velocity.x < MAX_HORIZONTAL_VELOCITY_PER_SECOND
    {
        let delta_velocity_x = horizontal_acceleration * DELTA_SECONDS * sign;
        if player.velocity.x + delta_velocity_x > MAX_HORIZONTAL_VELOCITY_PER_SECOND {
            acceleration.x =
                (MAX_HORIZONTAL_VELOCITY_PER_SECOND - player.velocity.x) / DELTA_SECONDS;
        } else if player.velocity.x + delta_velocity_x < -MAX_HORIZONTAL_VELOCITY_PER_SECOND {
            acceleration.x =
                (-MAX_HORIZONTAL_VELOCITY_PER_SECOND - player.velocity.x) / DELTA_SECONDS;
        } else {
            acceleration.x = horizontal_acceleration * sign;
        }
    }

    if !is_grounded && !input.jump {
        player.force_jumping = false;
    }

    let travel_distance = match player.velocity.y >= 0. && player.force_jumping {
        true => JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE,
        false => JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING,
    };

    let gravity = (-2.
        * JUMP_HEIGHT
        * (MAX_HORIZONTAL_VELOCITY_PER_SECOND * MAX_HORIZONTAL_VELOCITY_PER_SECOND))
        / (travel_distance * travel_distance);
    acceleration.y += gravity;

    player.velocity += acceleration * DELTA_SECONDS;
    if is_grounded && input.jump {
        player.velocity.y = (2. * JUMP_HEIGHT * MAX_HORIZONTAL_VELOCITY_PER_SECOND)
            / JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE;

        player.force_jumping = true;
    }

    // Do our own integration. We know the position we want the player to go, so we just set the
    // velocity to the value we know will produce a translation to the position we want.
    let velocity = (player.velocity * DELTA_SECONDS
        + 0.5 * acceleration * DELTA_SECONDS * DELTA_SECONDS)
        / DELTA_SECONDS;

    player.applied_velocity = velocity;
    let player_rigid_body = &mut state.physics.rigid_bodies[player.rigid_body_handle];
    player_rigid_body.set_linvel(vector![velocity.x, velocity.y], true);
}

fn get_horizontal_acceleration(is_grounded: bool) -> f32 {
    match is_grounded {
        true => HORIZONTAL_ACCELERATION_PER_SECOND,
        false => JUMP_HORIZONTAL_ACCELERATION_PER_SECOND,
    }
}

fn is_player_grounded(state: &mut GameState) -> bool {
    let ray = Ray::new(
        point!(state.player.position.x, state.player.position.y),
        vector![0., -1.],
    );

    state
        .physics
        .cast_ray(ray, 0.01, InteractionGroups::new(0b10, 0b10))
        .is_some()
}

fn update_physics(state: &mut GameState) {
    state.physics.run_frame();

    let player_rigid_body = &state.physics.rigid_bodies[state.player.rigid_body_handle];
    let player_position_from_physics = player_rigid_body.translation();
    state.player.last_position = state.player.position;
    state.player.position.x = player_position_from_physics.x;
    state.player.position.y = player_position_from_physics.y;
    let velocity = player_rigid_body.linvel();
    let changed_velocity = Vector2::new(velocity.x, velocity.y) - state.player.applied_velocity;
    state.player.velocity += changed_velocity;
}

fn render(game: &mut Game, now_micros: u128) -> Result<(), wgpu::SurfaceError> {
    puffin::profile_function!();

    let micros_since_last_updated = (now_micros - game.last_update_micros) as f32;
    let interp_percent = micros_since_last_updated / DELTA_MICROS as f32;

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
            "[{}{}{}]",
            bool_to_x(self.left, "_", "L"),
            bool_to_x(self.jump, "_", "J"),
            bool_to_x(self.right, "_", "R"),
        )
    }
}

fn bool_to_x(value: bool, false_value: &'static str, true_value: &'static str) -> &'static str {
    match value {
        true => true_value,
        _ => false_value,
    }
}

#[test]
fn jump_test() {
    let physics = init_physics(DELTA_SECONDS);
    let scenario = create_first_scenario();
    let mut state = create_game_state(physics, &scenario);

    let mut input = Input {
        ..Default::default()
    };

    let mut max_player_y = state.player.position.y;

    loop {
        if state.frame_count > 100 {
            input.jump = true;
        }

        update(&mut state, &input);

        max_player_y = state.player.position.y.max(max_player_y);
        println!("{} {}", state.player.position.x, state.player.position.y);

        if state.frame_count > (10. / DELTA_SECONDS) as u128 {
            break;
        }
    }

    assert!(
        max_player_y < JUMP_HEIGHT + 0.01 && max_player_y > JUMP_HEIGHT - 0.01,
        "Jump did not peak near y {}. Peaked at y {}.",
        JUMP_HEIGHT,
        max_player_y
    );
}

#[test]
fn independent_jump_test() {
    let mut position = Vector2::new(0., 0.);
    let mut velocity = Vector2::new(0., 0.);

    let jump_velocity = (2. * JUMP_HEIGHT * MAX_HORIZONTAL_VELOCITY_PER_SECOND)
        / JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE;

    let mut frames = 0;
    let gravity = (-2.
        * JUMP_HEIGHT
        * (MAX_HORIZONTAL_VELOCITY_PER_SECOND * MAX_HORIZONTAL_VELOCITY_PER_SECOND))
        / (JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE * JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE);
    let acceleration = gravity;
    velocity.y = jump_velocity;

    loop {
        let orig_velocity = velocity;
        let delta_position =
            velocity.y * DELTA_SECONDS + 0.5 * acceleration * DELTA_SECONDS * DELTA_SECONDS;
        position.y += delta_position;
        velocity.y += acceleration * DELTA_SECONDS;
        frames += 1;
        println!(
            "[{}] P({};{}) ({};{})-V>({};{})",
            frames as f32 * DELTA_SECONDS,
            position.x,
            position.y,
            orig_velocity.x,
            orig_velocity.y,
            velocity.x,
            velocity.y,
        );

        if frames >= 20 {
            break;
        }
    }
}
