use bevy::color::palettes::{css, tailwind};
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{storage_buffer, uniform_buffer};
use bevy::render::render_resource::{
    BindGroupLayoutEntryBuilder, Buffer, BufferUsages, UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::shader::ShaderRef;
use bevy_marching_cubes::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            bevy_panorbit_camera::PanOrbitCameraPlugin,
            MarchingCubesPlugin::<MyComputeSampler, MyExtraBufferCache, StandardMaterial>::default(
            ),
            ExtractResourcePlugin::<Poi>::default(),
        ))
        .insert_resource(WireframeConfig {
            global: true,
            default_color: css::WHITE.into(),
        })
        .insert_resource(ChunkGeneratorSettings::<MyComputeSampler>::new(32, 8.0))
        .add_systems(Startup, setup)
        .add_systems(Update, draw_gizmos)
        .add_observer(add_extra_readbacks)
        .add_observer(clear_extra_readbacks)
        .run();
}

#[derive(TypePath)]
struct MyComputeSampler;

impl ChunkComputeShader for MyComputeSampler {
    fn shader() -> ShaderRef {
        "sample.wgsl".into()
    }
}

struct MyExtraBufferCache {
    poi_positions: Buffer,
    poi_positions_final: Buffer,
}

impl GpuExtraBufferCache for MyExtraBufferCache {
    fn define_extra_buffers() -> Vec<BindGroupLayoutEntryBuilder> {
        vec![
            uniform_buffer::<[Vec3; 1]>(false),
            storage_buffer::<[Vec3; 1]>(false),
        ]
    }

    fn create_extra_buffers() -> ScheduleConfigs<ScheduleSystem> {
        IntoSystem::into_system(
            |render_device: Res<RenderDevice>,
             render_queue: Res<RenderQueue>,
             buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
             mut cache: ResMut<GpuChunkGeneratorCache<MyComputeSampler, MyExtraBufferCache>>,
             poi: Res<Poi>| {
                for key in cache.drain_needed_extra_buffers() {
                    let mut poi_positions_buffer = UniformBuffer::from(&[Vec3::ZERO; 1]);
                    poi_positions_buffer.write_buffer(&render_device, &render_queue);

                    let poi_positions_final_buffer = poi.positions_final.get(&key).unwrap();

                    cache.insert_extra_buffers(
                        key,
                        MyExtraBufferCache {
                            poi_positions: poi_positions_buffer.buffer().unwrap().clone(),
                            poi_positions_final: buffers
                                .get(poi_positions_final_buffer)
                                .unwrap()
                                .buffer
                                .clone(),
                        },
                    );
                }
            },
        )
        .into_configs()
    }

    fn clear_extra_buffers() -> ScheduleConfigs<ScheduleSystem> {
        IntoSystem::into_system(
            |render_queue: Res<RenderQueue>,
             mut cache: ResMut<GpuChunkGeneratorCache<MyComputeSampler, MyExtraBufferCache>>,
             poi: Res<Poi>|
             -> Result<()> {
                for extra_buffers in cache.extra_buffers_mut() {
                    render_queue.write_buffer(
                        &extra_buffers.poi_positions,
                        0,
                        &value_data(&poi.positions)?,
                    );
                }
                Ok(())
            },
        )
        .into_configs()
    }

    fn buffers(&self) -> Vec<Buffer> {
        vec![self.poi_positions.clone(), self.poi_positions_final.clone()]
    }

    fn num_extra_readbacks() -> usize {
        1
    }
}

fn add_extra_readbacks(
    add: On<Add, ChunkRenderData<MyComputeSampler>>,
    chunks: Query<(&Chunk<MyComputeSampler>, &ChunkRenderData<MyComputeSampler>)>,
    mut poi: ResMut<Poi>,
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) -> Result {
    let entity = add.entity;
    let (chunk, chunk_data) = chunks.get(entity)?;
    let buffer = poi
        .positions_final
        .entry(chunk_data.buffers().clone())
        .or_insert_with(|| {
            let mut poi_positions_final_buffer = ShaderStorageBuffer::from([Vec3::ZERO; 1]);
            poi_positions_final_buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
            buffers.add(poi_positions_final_buffer)
        });
    commands
        .spawn((
            Name::new(format!("Chunk {:?} poi readback", chunk.position)),
            Readback::buffer(buffer.clone()),
            ChildOf(entity),
        ))
        .observe(
            move |readback: On<ReadbackComplete>, mut commands: Commands| -> Result {
                commands.entity(readback.entity).despawn();
                commands.entity(entity).insert(ChunkPoi {
                    position: readback.to_shader_type(),
                });
                commands.trigger(ReadbackReallyComplete(entity));
                Ok(())
            },
        );
    Ok(())
}

fn clear_extra_readbacks(_clear: On<ClearBufferCache<MyComputeSampler>>, mut poi: ResMut<Poi>) {
    poi.positions_final.clear();
}

#[derive(Resource, ExtractResource, Clone)]
struct Poi {
    positions: [Vec3; 1],
    positions_final: HashMap<BufferCache, Handle<ShaderStorageBuffer>>,
}

#[derive(Component)]
struct ChunkPoi {
    position: Vec3,
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Transform::from_xyz(4.0, 6.5, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy_panorbit_camera::PanOrbitCamera {
            button_orbit: MouseButton::Left,
            button_pan: MouseButton::Left,
            modifier_pan: Some(KeyCode::ShiftLeft),
            reversed_zoom: true,
            ..default()
        },
    ));

    commands.spawn((
        Name::new("Light"),
        DirectionalLight {
            illuminance: 4000.0,
            ..default()
        },
        Transform {
            rotation: Quat::from_euler(EulerRot::XYZ, -1.9, 0.8, 0.0),
            ..default()
        },
    ));

    commands.spawn(ChunkLoader::<MyComputeSampler>::new(5));

    commands.insert_resource(ChunkMaterial::<MyComputeSampler, StandardMaterial>::new(
        materials.add(Color::from(tailwind::EMERALD_500)),
    ));

    let poi_positions = [Vec3::ZERO; 1];
    commands.insert_resource(Poi {
        positions: poi_positions,
        positions_final: HashMap::new(),
    });
}

fn draw_gizmos(mut gizmos: Gizmos, pois: Query<(&GlobalTransform, &ChunkPoi)>) {
    // debug gizmo to make sure it's running
    gizmos.axes(Transform::default(), 1.0);

    for (transform, poi) in pois {
        gizmos.sphere(
            Isometry3d::from_translation(transform.transform_point(poi.position)),
            1.0,
            css::BLUE,
        );
    }
}
