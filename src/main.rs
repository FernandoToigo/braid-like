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
                size: Vector2::new(100., 1.),
            },
            Wall {
                position: Vector3::new(-1.75, 1., 0.),
                size: Vector2::new(1., 1.),
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
        println!(
            "BEFORE {} {}",
            game.state.player.position.x, game.state.player.position.y
        );
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
                    println!(
                        "AFTER {} {}",
                        game.state.player.position.x, game.state.player.position.y
                    );
                    game.is_replaying = false;
                    game.inputs.clear();
                    store_input(game, input.clone());
                    reset_state(game);
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

        print_update_log(game, input);
    }
    render(game, now_micros)?;

    game.last_frame_micros = now_micros;

    Ok(())
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
        "[{} ({}s)] ([{}]:{}:{}) P({};{}) ->({};{})",
        game.state.frame_count,
        micros_to_seconds(game.last_update_micros),
        index,
        count,
        input_log,
        game.state.player.position.x,
        game.state.player.position.y,
        game.state.player.velocity.x,
        game.state.player.velocity.y,
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
const JUMP_VELOCITY: f32 =
    (2. * JUMP_HEIGHT * MAX_HORIZONTAL_VELOCITY_PER_SECOND) / JUMP_HORIZONTAL_HALF_TOTAL_DISTANCE;

fn update_player(state: &mut GameState, input: &Input) {
    let is_grounded = is_player_grounded(state);
    let player = &mut state.player;

    if !is_grounded && !input.jump {
        player.force_jumping = false;
    }

    let mut acceleration = Vector2::new(0., 0.);
    acceleration.x = get_horizontal_acceleration(player, input, is_grounded);
    acceleration.y = get_vertical_acceleration(player);

    player.velocity += acceleration * DELTA_SECONDS;

    if is_grounded && input.jump {
        player.velocity.y = JUMP_VELOCITY;
        player.force_jumping = true;
    }

    // Do our own integration. We know the position we want the player to go, so we just set the
    // velocity to the value we know will produce a translation to the position we want.
    let velocity = integrate_velocity_to_delta_position(player, acceleration) / DELTA_SECONDS;

    set_physics_velocity(state, velocity);
}

/*
[44 (1.466652s)] P(-0.29411638;0) ->'(-2.5198958;-1.2623149) ->(-2.3305485;0) ([1]:6:[L__])
[45 (1.499985s)] P(-0.38957798;0.27525496) ->'(-2.8638766;8.257731) ->(-2.8638766;8.257731) ([2]:1:[LJ_])
[46 (1.533318s)] P(-0.48948395;0.53862715) ->'(-2.9972086;7.9012446) ->(-2.9972086;7.9012446) ([2]:2:[LJ_])
[47 (1.566651s)] P(-0.59827864;0.78005195) ->'(-3.2638724;7.2428145) ->(-3.2638724;7.2428145) ([2]:3:[LJ_])
[48 (1.599984s)] P(-0.715962;0.99952924) ->'(-3.5305364;6.5843854) ->(-3.5305364;6.5843854) ([2]:4:[LJ_])
[49 (1.633317s)] P(-0.84253407;1.1970592) ->'(-3.7972007;5.9259553) ->(-3.7972007;5.9259553) ([2]:5:[LJ_])
[50 (1.6666499s)] P(-0.97799486;1.3424476) ->'(-4.0638647;4.361698) ->(-4.0638647;4.361698) ([3]:1:[L__])
[51 (1.699983s)] P(-1.1124847;1.4457594) ->'(-4.034734;3.0993834) ->(-4.034734;3.0993834) ([3]:2:[L__])
[52 (1.733316s)] P(-1.2458167;1.5069945) ->'(-4;1.8370686) ->(-4;1.8370686) ([4]:1:[___])
[53 (1.766649s)] P(-1.3791487;1.5261528) ->'(-4;0.5747535) ->(-4;0.5747535) ([4]:2:[___])
[54 (1.799982s)] P(-1.5124807;1.5032344) ->'(-4;-0.6875614) ->(-4;-0.6875614) ([4]:3:[___])
[55 (1.833315s)] P(-1.6458127;1.4382391) ->'(-4;-1.9498764) ->(-4;-1.9498764) ([4]:4:[___])
[56 (1.866648s)] P(-1.7630839;1.458673) ->'(-4;-3.212191) ->(-3.5181713;0) ([4]:5:[___])
[57 (1.899981s)] P(-1.8740436;1.4717507) ->'(-3.5181713;-1.2623149) ->(-3.328824;0) ([4]:6:[___])
[58 (1.933314s)] P(-1.9786918;1.4801205) ->'(-3.328824;-1.2623149) ->(-3.1394768;0) ([4]:7:[___])
[59 (1.966647s)] P(-2.0770285;1.4854772) ->'(-3.1394768;-1.2623149) ->(-2.9501295;0) ([4]:8:[___])
[60 (1.99998s)] P(-2.1690536;1.4889054) ->'(-2.9501295;-1.2623149) ->(-2.7607822;0) ([4]:9:[___])
[61 (2.033313s)] P(-2.2547672;1.4910995) ->'(-2.7607822;-1.2623149) ->(-2.571435;0) ([4]:10:[___])
[62 (2.066646s)] P(-2.3341694;1.4925036) ->'(-2.571435;-1.2623149) ->(-2.3820877;0) ([4]:11:[___])
[63 (2.099979s)] P(-2.40726;1.4934024) ->'(-2.3820877;-1.2623149) ->(-2.1927404;0) ([4]:12:[___])
[64 (2.133312s)] P(-2.474039;1.4939774) ->'(-2.1927404;-1.2623149) ->(-2.0033932;0) ([4]:13:[___])
[65 (2.166645s)] P(-2.5345066;1.4943455) ->'(-2.0033932;-1.2623149) ->(-1.8140459;0) ([4]:14:[___])
[66 (2.199978s)] P(-2.594974;1.4522688) ->'(-1.8140459;-1.2623149) ->(-1.8140459;-1.2623149) ([4]:15:[___])
[67 (2.233311s)] P(-2.6554415;1.3681153) ->'(-1.8140459;-2.5246298) ->(-1.8140459;-2.5246298) ([4]:16:[___])
[68 (2.266644s)] P(-2.715909;1.2418851) ->'(-1.8140459;-3.786945) ->(-1.8140459;-3.786945) ([4]:17:[___])
[69 (2.299977s)] P(-2.7763765;1.0735781) ->'(-1.8140459;-5.0492597) ->(-1.8140459;-5.0492597) ([4]:18:[___])
[70 (2.33331s)] P(-2.836844;0.86319447) ->'(-1.8140459;-6.3115745) ->(-1.8140459;-6.3115745) ([4]:19:[___])
[71 (2.366643s)] P(-2.8973114;0.610734) ->'(-1.8140459;-7.5738893) ->(-1.8140459;-7.5738893) ([4]:20:[___])
[72 (2.399976s)] P(-2.957779;0.3161968) ->'(-1.8140459;-8.836205) ->(-1.8140459;-8.836205) ([4]:21:[___])
[73 (2.433309s)] P(-3.0182464;-0.020417154) ->'(-1.8140459;-10.098519) ->(-1.8140459;-10.098519) ([4]:22:[___])
[74 (2.466642s)] P(-3.0219104;-0.014866979) ->'(-1.8140459;-11.360834) ->(-0.10992074;0) ([4]:23:[___])
[75 (2.499975s)] P(-3.0219104;-0.011314877) ->'(-0.10992074;-1.2623146) ->(0;0) ([4]:24:[___])
[76 (2.533308s)] P(-3.0219104;-0.0090415245) ->'(0;-1.2623149) ->(0;0) ([4]:25:[___])
[77 (2.566641s)] P(-3.0219104;-0.007586567) ->'(0;-1.2623149) ->(0;0) ([4]:26:[___])
[78 (2.599974s)] P(-3.0219104;-0.006655392) ->'(0;-1.2623149) ->(0;0) ([4]:27:[___])
[79 (2.633307s)] P(-3.0219104;-0.0060594496) ->'(0;-1.2623149) ->(0;0) ([4]:28:[___])
[80 (2.66664s)] P(-3.0219104;-0.005678063) ->'(0;-1.2623149) ->(0;0) ([4]:29:[___])
[81 (2.699973s)] P(-3.0219104;-0.0054339617) ->'(0;-1.2623149) ->(0;0) ([4]:30:[___])
[82 (2.733306s)] P(-3.0219104;-0.00527773) ->'(0;-1.2623149) ->(0;0) ([4]:31:[___])
[83 (2.766639s)] P(-3.0219104;-0.005177739) ->'(0;-1.2623149) ->(0;0) ([4]:32:[___])
[84 (2.799972s)] P(-3.0219104;-0.005113754) ->'(0;-1.2623149) ->(0;0) ([4]:33:[___])
[85 (2.833305s)] P(-3.0219104;-0.0050728144) ->'(0;-1.2623149) ->(0;0) ([4]:34:[___])
[86 (2.866638s)] P(-3.0219104;-0.005046595) ->'(0;-1.2623149) ->(0;0) ([4]:35:[___])
[87 (2.899971s)] P(-3.0219104;-0.0050298166) ->'(0;-1.2623149) ->(0;0) ([4]:36:[___])


[125 (4.166625s)] P(0;0) ->'(0;-1.2623149) ->(0;0) ([0]:38:[___])
[126 (4.199958s)] P(-0.020354621;0) ->'(-0.799992;-1.2623149) ->(-0.61064476;0) ([1]:1:[L__])
[127 (4.233291s)] P(-0.052175153;0) ->'(-1.1439728;-1.2623149) ->(-0.9546255;0) ([1]:2:[L__])
[128 (4.266624s)] P(-0.09546159;0) ->'(-1.4879533;-1.2623149) ->(-1.298606;0) ([1]:3:[L__])
[129 (4.299957s)] P(-0.15021394;0) ->'(-1.8319341;-1.2623149) ->(-1.6425868;0) ([1]:4:[L__])
[130 (4.33329s)] P(-0.21643221;0) ->'(-2.175915;-1.2623149) ->(-1.9865677;0) ([1]:5:[L__])
[131 (4.366623s)] P(-0.29411638;0) ->'(-2.5198958;-1.2623149) ->(-2.3305485;0) ([1]:6:[L__])
[132 (4.399956s)] P(-0.38957798;0.27525496) ->'(-2.8638766;8.257731) ->(-2.8638766;8.257731) ([2]:1:[LJ_])
[133 (4.433289s)] P(-0.48948395;0.53862715) ->'(-2.9972086;7.9012446) ->(-2.9972086;7.9012446) ([2]:2:[LJ_])
[134 (4.466622s)] P(-0.59827864;0.78005195) ->'(-3.2638724;7.2428145) ->(-3.2638724;7.2428145) ([2]:3:[LJ_])
[135 (4.499955s)] P(-0.715962;0.99952924) ->'(-3.5305364;6.5843854) ->(-3.5305364;6.5843854) ([2]:4:[LJ_])
[136 (4.533288s)] P(-0.84253407;1.1970592) ->'(-3.7972007;5.9259553) ->(-3.7972007;5.9259553) ([2]:5:[LJ_])
[137 (4.566621s)] P(-0.97799486;1.3424476) ->'(-4.0638647;4.361698) ->(-4.0638647;4.361698) ([3]:1:[L__])
[138 (4.599954s)] P(-1.1124847;1.4457594) ->'(-4.034734;3.0993834) ->(-4.034734;3.0993834) ([3]:2:[L__])
[139 (4.633287s)] P(-1.2458167;1.5069945) ->'(-4;1.8370686) ->(-4;1.8370686) ([4]:1:[___])
[140 (4.66662s)] P(-1.3791487;1.5261528) ->'(-4;0.5747535) ->(-4;0.5747535) ([4]:2:[___])
[141 (4.699953s)] P(-1.5124807;1.5032344) ->'(-4;-0.6875614) ->(-4;-0.6875614) ([4]:3:[___])
[142 (4.733286s)] P(-1.6458127;1.4382391) ->'(-4;-1.9498764) ->(-4;-1.9498764) ([4]:4:[___])
[143 (4.766619s)] P(-1.7630839;1.458673) ->'(-4;-3.212191) ->(-3.5181713;0) ([4]:5:[___])
[144 (4.799952s)] P(-1.8740436;1.4717507) ->'(-3.5181713;-1.2623149) ->(-3.328824;0) ([4]:6:[___])
[145 (4.833285s)] P(-1.9786918;1.4801205) ->'(-3.328824;-1.2623149) ->(-3.1394768;0) ([4]:7:[___])
[146 (4.866618s)] P(-2.0770285;1.4854772) ->'(-3.1394768;-1.2623149) ->(-2.9501295;0) ([4]:8:[___])
[147 (4.899951s)] P(-2.1690536;1.4889054) ->'(-2.9501295;-1.2623149) ->(-2.7607822;0) ([4]:9:[___])
[148 (4.933284s)] P(-2.2547672;1.4910995) ->'(-2.7607822;-1.2623149) ->(-2.571435;0) ([4]:10:[___])
[149 (4.966617s)] P(-2.3341694;1.4925036) ->'(-2.571435;-1.2623149) ->(-2.3820877;0) ([4]:11:[___])
[150 (4.99995s)] P(-2.40726;1.4934024) ->'(-2.3820877;-1.2623149) ->(-2.1927404;0) ([4]:12:[___])
[151 (5.0332828s)] P(-2.474039;1.4939774) ->'(-2.1927404;-1.2623149) ->(-2.0033932;0) ([4]:13:[___])
[152 (5.066616s)] P(-2.5345066;1.4943455) ->'(-2.0033932;-1.2623149) ->(-1.8140459;0) ([4]:14:[___])
[153 (5.099949s)] P(-2.594974;1.4522688) ->'(-1.8140459;-1.2623149) ->(-1.8140459;-1.2623149) ([4]:15:[___])
[154 (5.133282s)] P(-2.6554415;1.3681153) ->'(-1.8140459;-2.5246298) ->(-1.8140459;-2.5246298) ([4]:16:[___])
[155 (5.166615s)] P(-2.715909;1.2418851) ->'(-1.8140459;-3.786945) ->(-1.8140459;-3.786945) ([4]:17:[___])
[156 (5.199948s)] P(-2.7763765;1.0735781) ->'(-1.8140459;-5.0492597) ->(-1.8140459;-5.0492597) ([4]:18:[___])
[157 (5.233281s)] P(-2.836844;0.86319447) ->'(-1.8140459;-6.3115745) ->(-1.8140459;-6.3115745) ([4]:19:[___])
[158 (5.266614s)] P(-2.8973114;0.610734) ->'(-1.8140459;-7.5738893) ->(-1.8140459;-7.5738893) ([4]:20:[___])
[159 (5.299947s)] P(-2.957779;0.3161968) ->'(-1.8140459;-8.836205) ->(-1.8140459;-8.836205) ([4]:21:[___])
[160 (5.33328s)] P(-3.0182464;-0.020417154) ->'(-1.8140459;-10.098519) ->(-1.8140459;-10.098519) ([4]:22:[___])
[161 (5.366613s)] P(-3.0219104;-0.014866979) ->'(-1.8140459;-11.360834) ->(-0.10992074;0) ([4]:23:[___])
[162 (5.399946s)] P(-3.0219104;-0.011314877) ->'(-0.10992074;-1.2623146) ->(0;0) ([4]:24:[___])
[75 (2.499975s)] P(-3.0219104;-0.011314877) ->'(-0.10992074;-1.2623146) ->(0;0) ([4]:24:[___])
*/

fn set_physics_velocity(state: &mut GameState, velocity: Vector2<f32>) {
    state.player.applied_velocity = velocity;
    let player_rigid_body = &mut state.physics.rigid_bodies[state.player.rigid_body_handle];
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

    let horizontal_acceleration = match is_grounded {
        true => HORIZONTAL_ACCELERATION_PER_SECOND,
        false => JUMP_HORIZONTAL_ACCELERATION_PER_SECOND,
    };

    if player.velocity.x > -MAX_HORIZONTAL_VELOCITY_PER_SECOND
        && player.velocity.x < MAX_HORIZONTAL_VELOCITY_PER_SECOND
    {
        let delta_velocity_x = horizontal_acceleration * DELTA_SECONDS * sign;
        if player.velocity.x + delta_velocity_x > MAX_HORIZONTAL_VELOCITY_PER_SECOND {
            return (MAX_HORIZONTAL_VELOCITY_PER_SECOND - player.velocity.x) / DELTA_SECONDS;
        } else if player.velocity.x + delta_velocity_x < -MAX_HORIZONTAL_VELOCITY_PER_SECOND {
            return (-MAX_HORIZONTAL_VELOCITY_PER_SECOND - player.velocity.x) / DELTA_SECONDS;
        }

        return horizontal_acceleration * sign;
    }

    0.
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
fn jump_height_test() {
    let physics = init_physics(DELTA_SECONDS);
    let scenario = create_first_scenario();
    let mut state = create_game_state(physics, &scenario);
    let mut max_player_y = 0.;

    let input = Input {
        jump: true,
        ..Default::default()
    };

    loop {
        update(&mut state, &input);

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
    let scenario = create_first_scenario();
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
        }

        update(&mut state, &input);

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
