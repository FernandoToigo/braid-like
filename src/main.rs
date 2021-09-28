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
const SCENARIO_COUNT: usize = 2;

pub struct Game {
    renderer: WgpuRenderer,
    start_instant: Instant,
    last_frame_micros: u128,
    last_update_micros: u128,
    scenario_index: usize,
    scenario: Scenario,
    state: GameState,
    profiler: puffin_http::Server,
    inputs: Vec<InputRegistry>,
    is_replaying: bool,
    is_replaying_finished: bool,
    replay_registry_index: usize,
    replay_registry_count: u128,
}

pub struct Scenario {
    player_start_position: Vector3<f32>,
    player_clone_start_position: Vector3<f32>,
    finish_position: Vector3<f32>,
    walls: Vec<Wall>,
}

pub struct GameState {
    frame_count: u128,
    physics: Physics,
    player: Player,
    player_clone: Player,
    camera: Camera,
    finish_collider_handle: ColliderHandle,
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
    jump_down: bool,
    confirm_down: bool,
    debug_one: bool,
}

enum UpdateResult {
    None,
    FinishedScenario,
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

    let scenario_index = 0;
    let scenario = create_scenario(scenario_index);
    let state = create_game_state(physics, &scenario);
    let instances = create_instances(&state, &scenario);
    let (event_loop, renderer) = block_on(WgpuRenderer::init(instances));

    let mut game = Game {
        renderer,
        start_instant: Instant::now(),
        last_frame_micros: 0,
        last_update_micros: 0,
        scenario_index,
        scenario,
        state,
        profiler,
        inputs: Vec::with_capacity(9000),
        is_replaying: false,
        is_replaying_finished: false,
        replay_registry_index: 0,
        replay_registry_count: 0,
    };

    let mut last_input = Input {
        ..Default::default()
    };
    let mut input = Input {
        ..Default::default()
    };

    event_loop.run(move |event, _, control_flow| {
        let event_results = read_events(event, &mut game.renderer, &mut input, &mut last_input);

        let frame_result = match event_results.events_finished {
            true => frame(&mut game, &mut input),
            false => Ok(()),
        };

        *control_flow = resolve_frame(&event_results, frame_result);
    });
}

fn create_scenario(index: usize) -> Scenario {
    match index {
        0 => create_first_scenario(),
        1 => create_second_scenario(),
        _ => panic!("There is no scenario with index {}", index),
    }
}

fn create_first_scenario() -> Scenario {
    Scenario {
        player_start_position: Vector3::new(-4.91666, 0.25, -1.),
        player_clone_start_position: Vector3::new(-4.91666, -4.5, -1.),
        finish_position: Vector3::new(4.83333, -4.3, 1.),
        walls: vec![
            Wall {
                // Floor
                position: Vector3::new(0., -5.0, 0.),
                size: Vector2::new(100., 1.0),
            },
            Wall {
                // Dividing Floor
                position: Vector3::new(0., 0., 0.),
                size: Vector2::new(100., 0.5),
            },
            Wall {
                // Ceiling
                position: Vector3::new(0., 5.0, 0.),
                size: Vector2::new(100., 1.0),
            },
            Wall {
                // Left wall
                position: Vector3::new(-6.41666, 0., 0.),
                size: Vector2::new(0.5, 20.0),
            },
            Wall {
                // Right wall
                position: Vector3::new(6.41666, 0., 0.),
                size: Vector2::new(0.5, 20.0),
            },
            Wall {
                // Top Obstacle
                position: Vector3::new(-3.33333, 0.75, 0.),
                size: Vector2::new(1.0, 1.0),
            },
            Wall {
                // Bottom Obstacle
                position: Vector3::new(-3.33333, -4.0, 0.),
                size: Vector2::new(1.0, 1.0),
            },
        ],
    }
}

fn create_second_scenario() -> Scenario {
    Scenario {
        player_start_position: Vector3::new(-4.91666, 0.25, -1.),
        player_clone_start_position: Vector3::new(-4.91666, -4.5, -1.),
        finish_position: Vector3::new(-5.0, -2.55, 1.),
        walls: vec![
            Wall {
                // Floor
                position: Vector3::new(0., -5.0, 0.),
                size: Vector2::new(100., 1.0),
            },
            Wall {
                // Dividing Floor
                position: Vector3::new(0., 0., 0.),
                size: Vector2::new(100., 0.5),
            },
            Wall {
                // Ceiling
                position: Vector3::new(0., 5.0, 0.),
                size: Vector2::new(100., 1.0),
            },
            Wall {
                // Left wall
                position: Vector3::new(-6.41666, 0., 0.),
                size: Vector2::new(0.5, 20.0),
            },
            Wall {
                // Right wall
                position: Vector3::new(6.41666, 0., 0.),
                size: Vector2::new(0.5, 20.0),
            },
            Wall {
                // Bottom Platform
                position: Vector3::new(-2.6666, -3.0, 0.),
                size: Vector2::new(8.0, 0.5),
            },
        ],
    }
}

#[cfg(test)]
fn create_test_scenario() -> Scenario {
    Scenario {
        player_start_position: Vector3::new(0., 0., -1.),
        player_clone_start_position: Vector3::new(10., 0., -1.),
        finish_position: Vector3::new(10., 0., 0.),
        walls: vec![Wall {
            // Floor
            position: Vector3::new(0., -0.5, 0.),
            size: Vector2::new(100., 1.0),
        }],
    }
}

fn create_game_state(mut physics: Physics, scenario: &Scenario) -> GameState {
    let player = create_player(&mut physics, scenario.player_start_position);
    let player_clone = create_player(&mut physics, scenario.player_clone_start_position);

    let finish_collider = ColliderBuilder::cuboid(0.25, 0.25)
        .translation(vector!(
            scenario.finish_position.x,
            scenario.finish_position.y
        ))
        .active_events(ActiveEvents::INTERSECTION_EVENTS)
        .sensor(true)
        .build();
    let finish_collider_handle = physics.colliders.insert(finish_collider);

    scenario
        .walls
        .iter()
        .map(|wall| {
            ColliderBuilder::cuboid(wall.size.x * 0.5, wall.size.y * 0.5)
                .translation(vector!(wall.position.x, wall.position.y))
                .friction(0.0)
                .collision_groups(InteractionGroups::new(0b10, 0xFFFF))
                .build()
        })
        .for_each(|wall_collider| {
            physics.colliders.insert(wall_collider);
        });

    GameState {
        frame_count: 0,
        physics,
        player,
        player_clone,
        camera: Camera {
            position: Vector3::new(0., 0., 10.),
            last_position: Vector3::new(0., 0., 10.),
            orthographic_height: 10.,
        },
        finish_collider_handle,
    }
}

fn create_player(physics: &mut Physics, start_position: Vector3<f32>) -> Player {
    let rigid_body = rapier2d::dynamics::RigidBodyBuilder::new_dynamic()
        .translation(vector![start_position.x, start_position.y])
        .lock_rotations()
        .build();
    let rigid_body_handle = physics.rigid_bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(0.25, 0.5)
        .translation(vector![0., 0.5])
        .friction(0.0)
        .collision_groups(InteractionGroups::new(0b1, 0xFFFF))
        .build();
    physics
        .colliders
        .insert_with_parent(collider, rigid_body_handle, &mut physics.rigid_bodies);

    Player {
        position: start_position,
        last_position: start_position,
        rigid_body_handle,
        texture_offset: Vector2::new(0., 0.5),
        texture_index: 0,
        force_jumping: false,
        velocity: Vector2::new(0., 0.),
        applied_velocity: Vector2::new(0., 0.),
    }
}

fn reset_state(game: &mut Game) {
    reset_player(
        &mut game.state.physics,
        &mut game.state.player,
        game.scenario.player_start_position,
    );
    reset_player(
        &mut game.state.physics,
        &mut game.state.player_clone,
        game.scenario.player_clone_start_position,
    );
}

fn reset_player(physics: &mut Physics, player: &mut Player, start_position: Vector3<f32>) {
    player.velocity = Vector2::new(0., 0.);
    player.applied_velocity = Vector2::new(0., 0.);
    player.position = start_position;
    player.last_position = start_position;
    let player_rigid_body = &mut physics.rigid_bodies[player.rigid_body_handle];
    player_rigid_body.set_linvel(vector![0.0, 0.0], true);
    player_rigid_body.set_translation(vector![start_position.x, start_position.y], true);
}

fn create_instances(state: &GameState, scenario: &Scenario) -> Vec<InstanceRaw> {
    // Instances will be rendered in the same order which they are added into this list.
    // So they should be added from back to front.

    let mut instances = Vec::new();
    instances.push(InstanceRaw::new(
        state.player.position + state.player.texture_offset.extend(0.),
        Vector3::new(1., 1., 1.),
        state.player.texture_index,
    ));
    instances.push(InstanceRaw::new(
        state.player_clone.position + state.player_clone.texture_offset.extend(0.),
        Vector3::new(1., 1., 1.),
        state.player_clone.texture_index,
    ));
    scenario
        .walls
        .iter()
        .for_each(|wall| instances.push(InstanceRaw::new(wall.position, wall.size.extend(1.), 1)));
    instances.push(InstanceRaw::new(
        scenario.finish_position,
        Vector3::new(0.75, 0.75, 0.75),
        2,
    ));

    instances
}

fn frame(game: &mut Game, input: &mut Input) -> anyhow::Result<(), wgpu::SurfaceError> {
    puffin::profile_function!();

    let now_micros = game.start_instant.elapsed().as_micros();

    if input.confirm_down && !game.is_replaying {
        reset_state(game);
        game.is_replaying = true;
        game.replay_registry_index = 0;
        game.replay_registry_count = 0;
    }

    update_profiler(&game.profiler);

    update_loop(game, now_micros, input);
    render(game, now_micros)?;

    game.last_frame_micros = now_micros;
    Ok(())
}

fn update_loop(game: &mut Game, now_micros: u128, input: &mut Input) {
    let mut frame_update_count = 0;
    while now_micros - game.last_update_micros > DELTA_MICROS {
        if game.is_replaying_finished {
            if input.confirm_down {
                game.is_replaying = false;
                game.is_replaying_finished = false;
                game.inputs.clear();
                reset_state(game);
            } else {
                return;
            }
        }

        let replay_input = {
            if game.is_replaying {
                match get_next_replay_input(game) {
                    Some(new_input) => Some(new_input),
                    None => {
                        game.is_replaying_finished = true;
                        return;
                    }
                }
            } else {
                store_input(game, input.clone());
                None
            }
        };

        let update_result = update(
            &mut game.state,
            &replay_input.as_ref().unwrap_or(input),
            game.is_replaying,
        );
        game.last_update_micros += DELTA_MICROS;
        frame_update_count += 1;
        if frame_update_count >= MAXIMUM_UPDATE_STEPS_PER_FRAME {
            println!("WARNING. Game is slowing down.");
            break;
        }

        print_update_log(game, input);

        if matches!(update_result, UpdateResult::FinishedScenario) {
            finish_scenario(game);
        }

        input.confirm_down = false;
        input.jump_down = false;
    }
}

fn finish_scenario(game: &mut Game) {
    game.is_replaying = false;
    game.inputs.clear();
    if game.scenario_index + 1 < SCENARIO_COUNT {
        game.scenario_index += 1;
        game.scenario = create_scenario(game.scenario_index);
        let physics = init_physics(DELTA_SECONDS);
        game.state = create_game_state(physics, &game.scenario);
        let instances = create_instances(&game.state, &game.scenario);
        game.renderer.replace_instances(instances);
    } else {
        reset_state(game);
    }
}

fn print_update_log(game: &Game, input: &Input) {
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

    println!(
        "[{} ({}s)] ([{}]:{}:{}) P1({};{}) P2({};{})",
        game.state.frame_count,
        micros_to_seconds(game.last_update_micros),
        index,
        count,
        input_log,
        game.state.player.position.x,
        game.state.player.position.y,
        game.state.player_clone.position.x,
        game.state.player_clone.position.y,
    );
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
    last_input: &mut Input,
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
                } => {
                    input.jump = is_pressed(state);
                    input.jump_down = is_pressed_down(state, last_input.jump_down);
                    last_input.jump_down = is_pressed(state);
                }
                KeyboardInput {
                    state,
                    virtual_keycode: Some(VirtualKeyCode::E),
                    ..
                } => {
                    input.confirm_down = is_pressed_down(state, last_input.confirm_down);
                    last_input.confirm_down = is_pressed(state);
                }
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

fn is_pressed_down(state: &ElementState, was_pressed: bool) -> bool {
    matches!(state, ElementState::Pressed) && !was_pressed
}

fn update(state: &mut GameState, input: &Input, update_clone: bool) -> UpdateResult {
    puffin::profile_function!();

    update_player(&mut state.physics, &mut state.player, input);
    if update_clone {
        update_player(&mut state.physics, &mut state.player_clone, input);
    }
    let update_result = update_physics(state);
    update_camera(state);

    state.frame_count += 1;

    update_result
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

const GROUND_HORIZONTAL_DECCELERATION_PER_SECOND: f32 = 10.;
const HORIZONTAL_ACCELERATION_PER_SECOND: f32 = 25.;
const JUMP_HORIZONTAL_ACCELERATION_PER_SECOND: f32 = 12.5;
const MAX_HORIZONTAL_VELOCITY_PER_SECOND: f32 = 4.;
const JUMP_HEIGHT: f32 = 2.0;
const JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE: f32 = 1.8;
const JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING: f32 = 1.3;
const JUMP_VELOCITY: f32 =
    (2. * JUMP_HEIGHT * MAX_HORIZONTAL_VELOCITY_PER_SECOND) / JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE;

fn update_player(physics: &mut Physics, player: &mut Player, input: &Input) {
    let is_grounded = is_player_grounded(physics, player);

    if !is_grounded && !input.jump {
        player.force_jumping = false;
    }

    let mut acceleration = Vector2::new(0., 0.);
    acceleration.x = get_horizontal_acceleration(player, input, is_grounded);
    acceleration.y = get_vertical_acceleration(player);

    player.velocity += acceleration * DELTA_SECONDS;

    if is_grounded && input.jump_down {
        player.velocity.y = JUMP_VELOCITY;
        player.force_jumping = true;
    }

    // Do our own integration. We know the position we want the player to go, so we just set the
    // velocity to the value we know will produce a translation into that position.
    let velocity = integrate_velocity_to_delta_position(player, acceleration) / DELTA_SECONDS;

    set_physics_velocity(physics, player, velocity);
}

fn set_physics_velocity(physics: &mut Physics, player: &mut Player, velocity: Vector2<f32>) {
    player.applied_velocity = velocity;
    let player_rigid_body = &mut physics.rigid_bodies[player.rigid_body_handle];
    player_rigid_body.set_linvel(vector![velocity.x, velocity.y], true);
}

fn integrate_velocity_to_delta_position(
    player: &Player,
    acceleration: Vector2<f32>,
) -> Vector2<f32> {
    player.velocity * DELTA_SECONDS + 0.5 * acceleration * DELTA_SECONDS * DELTA_SECONDS
}

fn get_horizontal_acceleration(player: &mut Player, input: &Input, is_grounded: bool) -> f32 {
    let sign = match (input.left, input.right) {
        (true, false) => -1.,
        (false, true) => 1.,
        _ => 0.,
    };

    let mut horizontal_acceleration = match is_grounded {
        true => HORIZONTAL_ACCELERATION_PER_SECOND,
        false => JUMP_HORIZONTAL_ACCELERATION_PER_SECOND,
    } * sign;

    if is_grounded {
        horizontal_acceleration += get_horizontal_decceleration(player);
    }

    let delta_velocity_x = horizontal_acceleration * DELTA_SECONDS;

    if delta_velocity_x > 0. && player.velocity.x < MAX_HORIZONTAL_VELOCITY_PER_SECOND {
        if player.velocity.x + delta_velocity_x > MAX_HORIZONTAL_VELOCITY_PER_SECOND {
            return (MAX_HORIZONTAL_VELOCITY_PER_SECOND - player.velocity.x) / DELTA_SECONDS;
        }

        return horizontal_acceleration;
    }

    if delta_velocity_x < 0. && player.velocity.x > -MAX_HORIZONTAL_VELOCITY_PER_SECOND {
        if player.velocity.x + delta_velocity_x < -MAX_HORIZONTAL_VELOCITY_PER_SECOND {
            return (-MAX_HORIZONTAL_VELOCITY_PER_SECOND - player.velocity.x) / DELTA_SECONDS;
        }

        return horizontal_acceleration;
    }

    0.
}

fn get_horizontal_decceleration(player: &Player) -> f32 {
    let velocity_sign = match player.velocity.x {
        x if x >= 0. => 1.,
        x if x < 0. => -1.,
        _ => 0.,
    };

    let delta_decceleration =
        GROUND_HORIZONTAL_DECCELERATION_PER_SECOND * -velocity_sign * DELTA_SECONDS;

    match player.velocity.x.abs() < delta_decceleration.abs() {
        true => -player.velocity.x,
        false => GROUND_HORIZONTAL_DECCELERATION_PER_SECOND * -velocity_sign,
    }
}

fn get_vertical_acceleration(player: &Player) -> f32 {
    let travel_distance = match player.velocity.y >= 0. && player.force_jumping {
        true => JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE,
        false => JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING,
    };

    let gravity = (-2.
        * JUMP_HEIGHT
        * (MAX_HORIZONTAL_VELOCITY_PER_SECOND * MAX_HORIZONTAL_VELOCITY_PER_SECOND))
        / (travel_distance * travel_distance);

    gravity
}

fn is_player_grounded(physics: &mut Physics, player: &Player) -> bool {
    let ray = Ray::new(
        point!(player.position.x, player.position.y),
        vector![0., -1.],
    );

    physics
        .cast_ray(ray, 0.01, InteractionGroups::new(0b10, 0b10))
        .is_some()
}

fn update_physics(state: &mut GameState) -> UpdateResult {
    state.physics.run_frame();

    update_player_physics(&mut state.player, &state.physics);
    update_player_physics(&mut state.player_clone, &state.physics);

    match state.physics.last_step_intersections().iter().any(|e| {
        e.intersecting
            && (e.collider1 == state.finish_collider_handle
                || e.collider2 == state.finish_collider_handle)
    }) {
        true => UpdateResult::FinishedScenario,
        false => UpdateResult::None,
    }
}

fn update_player_physics(player: &mut Player, physics: &Physics) {
    let player_rigid_body = &physics.rigid_bodies[player.rigid_body_handle];
    let player_position_from_physics = player_rigid_body.translation();
    player.last_position = player.position;
    player.position.x = player_position_from_physics.x;
    player.position.y = player_position_from_physics.y;
    let velocity = player_rigid_body.linvel();
    let changed_velocity = Vector2::new(velocity.x, velocity.y) - player.applied_velocity;
    player.velocity += changed_velocity;
}

fn render(game: &mut Game, now_micros: u128) -> Result<(), wgpu::SurfaceError> {
    puffin::profile_function!();

    let micros_since_last_updated = (now_micros - game.last_update_micros) as f32;
    let interp_percent = (micros_since_last_updated / DELTA_MICROS as f32).min(1.);

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
fn jump_height_test() {
    let physics = init_physics(DELTA_SECONDS);
    let scenario = create_test_scenario();
    let mut state = create_game_state(physics, &scenario);
    let mut max_player_y = 0.;

    let input = Input {
        jump: true,
        jump_down: true,
        ..Default::default()
    };

    loop {
        update(&mut state, &input, false);

        max_player_y = state.player.position.y.max(max_player_y);
        println!("{} {}", state.player.position.x, state.player.position.y);
        if state.frame_count > (10. / DELTA_SECONDS) as u128 {
            break;
        }
    }

    assert!(
        are_approximate(max_player_y, JUMP_HEIGHT),
        "Jump did not peak near y {}. Peaked at y {}.",
        JUMP_HEIGHT,
        max_player_y
    );
}

#[test]
fn jump_distance_test() {
    let physics = init_physics(DELTA_SECONDS);
    let scenario = create_test_scenario();
    let mut state = create_game_state(physics, &scenario);

    let mut input = Input {
        right: true,
        ..Default::default()
    };

    let mut jump_start_x = 0.;
    let mut last_player_position = state.player.position;
    loop {
        if state.frame_count > (1. / DELTA_SECONDS) as u128 {
            if !input.jump {
                jump_start_x = state.player.position.x;
            }
            input.jump = true;
            input.jump_down = true;
        }

        update(&mut state, &input, false);

        if input.jump && state.player.position.y < 0.01 {
            let min_distance = last_player_position.x - jump_start_x;
            let max_distance = state.player.position.x - jump_start_x;
            let required_distance =
                JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE + JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE_FALLING;

            assert!(
                min_distance < required_distance && max_distance > required_distance,
                "Not enough distance traveled while jumping. Incorret: {} <- {} -> {}.",
                min_distance,
                required_distance,
                max_distance,
            );
            break;
        }
        last_player_position = state.player.position;

        if state.frame_count > (10. / DELTA_SECONDS) as u128 {
            break;
        }
    }
}

#[cfg(test)]
fn are_approximate(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
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
