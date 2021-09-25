use rapier2d::prelude::*;

pub struct Physics {
    pub rigid_bodies: RigidBodySet,
    pub colliders: ColliderSet,
    gravity: Vector<f32>,
    integration_parameters: IntegrationParameters,
    pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    joints: JointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    event_handler: PhysicsEventHandler,
    intersections: Option<Vec<IntersectionEvent>>,
}

pub fn init_physics(step_delta_seconds: f32) -> Physics {
    let rigid_bodies = RigidBodySet::new();
    let colliders = ColliderSet::new();

    //let gravity = vector![0.0, -9.81];
    let gravity = vector![0.0, 0.0];
    let mut integration_parameters = IntegrationParameters::default();
    integration_parameters.dt = step_delta_seconds;
    let pipeline = PhysicsPipeline::new();
    let island_manager = IslandManager::new();
    let broad_phase = BroadPhase::new();
    let narrow_phase = NarrowPhase::new();
    let joints = JointSet::new();
    let ccd_solver = CCDSolver::new();
    let query_pipeline = QueryPipeline::new();
    let event_handler = PhysicsEventHandler::new();

    Physics {
        rigid_bodies,
        colliders,
        gravity,
        integration_parameters,
        pipeline,
        island_manager,
        broad_phase,
        narrow_phase,
        joints,
        ccd_solver,
        query_pipeline,
        event_handler,
        intersections: Some(Vec::with_capacity(10)),
    }
}

struct PhysicsEventHandler {
    intersections: std::sync::Mutex<Option<Vec<IntersectionEvent>>>,
}

impl PhysicsEventHandler {
    fn new() -> Self {
        PhysicsEventHandler {
            intersections: std::sync::Mutex::new(None),
        }
    }
}

impl EventHandler for PhysicsEventHandler {
    fn handle_intersection_event(&self, event: IntersectionEvent) {
        self.intersections
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .push(event);
    }
    fn handle_contact_event(&self, _: ContactEvent, _: &ContactPair) {
        todo!();
    }
}

impl Physics {
    pub fn run_frame(&mut self) {
        let physics_hooks = ();
        self.intersections.as_mut().unwrap().clear();
        let intersections = std::mem::replace(&mut self.intersections, None);
        *self.event_handler.intersections.lock().unwrap() = intersections;

        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_bodies,
            &mut self.colliders,
            &mut self.joints,
            &mut self.ccd_solver,
            &physics_hooks,
            &mut self.event_handler,
        );

        let intersections =
            std::mem::replace(&mut *self.event_handler.intersections.lock().unwrap(), None);
        self.intersections = Some(intersections.unwrap());
    }

    pub fn last_step_intersections(&self) -> &Vec<IntersectionEvent> {
        self.intersections.as_ref().unwrap()
    }

    pub fn cast_ray(
        &mut self,
        ray: Ray,
        max_toi: Real,
        groups: InteractionGroups,
    ) -> Option<(ColliderHandle, Real)> {
        self.query_pipeline
            .update(&self.island_manager, &self.rigid_bodies, &self.colliders);
        self.query_pipeline
            .cast_ray(&self.colliders, &ray, max_toi, true, groups, None)
    }
}
