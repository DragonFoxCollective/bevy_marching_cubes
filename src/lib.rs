use std::num::NonZero;

use bevy::asset::{RenderAssetUsages, embedded_asset, load_embedded_asset};
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{self, RenderGraph, RenderLabel};
use bevy::render::render_resource::binding_types::{
    storage_buffer, storage_buffer_read_only, storage_buffer_sized, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntryBuilder, Buffer, BufferUsages, CachedComputePipelineId,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
    StorageBuffer, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{MainWorld, Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::ShaderRef;

struct MarchingCubesGlobalPlugin;

impl Plugin for MarchingCubesGlobalPlugin {
    fn build(&self, app: &mut App) {
        {
            embedded_asset!(app, "marching_cubes.wgsl");
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, add_compute_render_graph_node)
            .add_systems(Render, clear_old_dispatches.in_set(RenderSystems::Cleanup))
            .init_resource::<ChunkGeneratorDispatches>();
    }
}

pub struct MarchingCubesPlugin<Sampler, Material> {
    _marker: std::marker::PhantomData<(Sampler, Material)>,
}

impl<Sampler, Material> Default for MarchingCubesPlugin<Sampler, Material> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Sampler: ChunkComputeShader + Send + Sync + 'static, Material: Asset + bevy::prelude::Material>
    Plugin for MarchingCubesPlugin<Sampler, Material>
{
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<MarchingCubesGlobalPlugin>() {
            app.add_plugins(MarchingCubesGlobalPlugin);
        }

        app.add_plugins((
            ExtractResourcePlugin::<ChunkGeneratorSettings<Sampler>>::default(),
            ExtractComponentPlugin::<ChunkRenderData<Sampler>>::default(),
        ))
        .add_systems(
            Update,
            (
                init_cache::<Sampler>,
                (
                    update_chunk_loaders::<Sampler>,
                    queue_chunks::<Sampler>,
                    start_chunks::<Sampler, Material>.run_if(
                        |done_loading: If<
                            Res<ChunkGeneratorComputePipelinesDoneLoading<Sampler>>,
                        >| done_loading.done,
                    ),
                )
                    .run_if(resource_exists::<ChunkGeneratorCache<Sampler>>),
            )
                .chain()
                .in_set(ChunkGenSystems),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ChunkGeneratorComputePipelinesDoneLoading<Sampler>>()
            .init_resource::<GpuChunkGeneratorCache<Sampler>>()
            .add_systems(ExtractSchedule, extract_pipelines_done::<Sampler>)
            .add_systems(
                Render, // RenderStartup
                init_compute_pipelines::<Sampler>
                    .run_if(resource_exists::<ChunkGeneratorSettings<Sampler>>) // Might not exist on the first frame due to extraction
                    .run_if(not(resource_exists::<
                        ChunkGeneratorComputePipelines<Sampler>,
                    >)),
            )
            .add_systems(
                Render,
                (
                    clear_gpu_cache::<Sampler>,
                    Sampler::prepare_extra_buffers(),
                    prepare_bind_groups::<Sampler>,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups)
                    .run_if(resource_exists::<ChunkGeneratorComputePipelines<Sampler>>),
            );
    }
}

#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ChunkGenSystems;

const WORKGROUP_SIZE: u32 = 8;

#[derive(ShaderType)]
struct MeshSettings {
    num_voxels_per_axis: u32,
    num_samples_per_axis: u32,
    chunk_size: f32,
    surface_threshold: f32,
}

#[derive(ShaderType, Default, Debug, Clone, Copy, Reflect)]
struct Vertex {
    position: Vec3,
    _padding1: f32,
    normal: Vec3,
    _padding2: f32,
}

#[derive(ShaderType, Default, Debug, Clone, Copy, Reflect)]
struct Triangle {
    vertex_a: u32,
    vertex_b: u32,
    vertex_c: u32,
}

#[derive(RenderLabel, Hash, Debug, PartialEq, Eq, Clone)]
struct ChunkGeneratorNodeLabel;

#[derive(Default)]
struct ChunkGeneratorNode;

#[derive(Resource, Default)]
struct ChunkGeneratorDispatches {
    dispatches: Vec<GpuBufferCache>,
}

impl render_graph::Node for ChunkGeneratorNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let dispatches = world.resource::<ChunkGeneratorDispatches>();
        let pipeline_cache = world.resource::<PipelineCache>();
        info!("Rendering {} chunks", dispatches.dispatches.len());

        for dispatch in dispatches.dispatches.iter() {
            let sample_pipeline = pipeline_cache
                .get_compute_pipeline(dispatch.sample_pipeline)
                .expect("sample pipeline wasn't finished generating");
            let march_pipeline = pipeline_cache
                .get_compute_pipeline(dispatch.march_pipeline)
                .expect("march pipeline wasn't finished generating");

            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("marching cubes sample pass"),
                            ..default()
                        });

                pass.set_bind_group(0, &dispatch.sample_bind_group, &[]);
                pass.set_pipeline(sample_pipeline);
                pass.dispatch_workgroups(
                    dispatch.sample_workgroups,
                    dispatch.sample_workgroups,
                    dispatch.sample_workgroups,
                );
            }

            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("marching cubes march pass"),
                            ..default()
                        });

                pass.set_bind_group(0, &dispatch.march_bind_group, &[]);
                pass.set_pipeline(march_pipeline);
                pass.dispatch_workgroups(
                    dispatch.march_workgroups,
                    dispatch.march_workgroups,
                    dispatch.march_workgroups,
                );
            }
        }

        Ok(())
    }
}

#[derive(Resource)]
struct ChunkGeneratorComputePipelines<Sampler> {
    sample_layout: BindGroupLayout,
    sample_pipeline: CachedComputePipelineId,
    march_layout: BindGroupLayout,
    march_pipeline: CachedComputePipelineId,
    _marker: std::marker::PhantomData<Sampler>,
}

fn init_compute_pipelines<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut pipelines_done_loading_res: ResMut<ChunkGeneratorComputePipelinesDoneLoading<Sampler>>,
) -> Result<()> {
    info!("Init compute pipelines");

    pipelines_done_loading_res.done = false;

    let sample_layout = render_device.create_bind_group_layout(
        "marching cubes sample bind group layout",
        &[
            uniform_buffer::<IVec3>(false),
            uniform_buffer::<MeshSettings>(false),
            storage_buffer::<Vec<f32>>(false),
        ]
        .into_iter()
        .chain(Sampler::define_extra_buffers())
        .enumerate()
        .map(|(i, b)| b.build(i as u32, ShaderStages::COMPUTE))
        .collect::<Vec<_>>(),
    );
    let sample_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("marching cubes sample compute shader".into()),
        layout: vec![sample_layout.clone()],
        shader: match Sampler::shader() {
            ShaderRef::Default => {
                return Err(format!("{} shader was not given", ShortName::of::<Sampler>()).into());
            }
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        },
        ..default()
    });

    let march_layout = render_device.create_bind_group_layout(
        "marching cubes march bind group layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<Vec<f32>>(false),
                uniform_buffer::<MeshSettings>(false),
                storage_buffer_sized(false, Some(settings.vertices_buffer_size())),
                storage_buffer::<u32>(false),
                storage_buffer_sized(false, Some(settings.triangles_buffer_size())),
                storage_buffer::<u32>(false),
            ),
        ),
    );
    let march_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("marching cubes march compute shader".into()),
        layout: vec![march_layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "marching_cubes.wgsl"),
        entry_point: Some("main".into()),
        ..default()
    });

    commands.insert_resource(ChunkGeneratorComputePipelines::<Sampler> {
        sample_layout,
        sample_pipeline,
        march_layout,
        march_pipeline,
        _marker: default(),
    });

    Ok(())
}

pub trait ChunkComputeShader {
    fn shader() -> ShaderRef;
    // TODO: needs to be cached. currently runs every frame, only being *used* if the bindgroups get rebuilt
    fn prepare_extra_buffers() -> ScheduleConfigs<ScheduleSystem> {
        IntoSystem::into_system(|| {}).into_configs()
    }
    fn define_extra_buffers() -> Vec<BindGroupLayoutEntryBuilder> {
        vec![]
    }
}

#[derive(Component, Debug)]
pub struct Chunk<Sampler> {
    pub position: IVec3,
    _marker: std::marker::PhantomData<Sampler>,
}

#[derive(Component, Debug)]
struct ChunkGenData {
    vertices: Option<Vec<Vertex>>,
    triangles: Option<Vec<Triangle>>,
}

#[derive(Component, ExtractComponent, Debug)]
pub struct ChunkRenderData<Sampler: Send + Sync + 'static> {
    position: IVec3,
    buffers: BufferCache,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler: Send + Sync + 'static> Clone for ChunkRenderData<Sampler> {
    fn clone(&self) -> Self {
        Self {
            position: self.position,
            buffers: self.buffers.clone(),
            _marker: self._marker,
        }
    }
}

#[derive(Component, Default, Debug)]
pub struct ChunkLoader<T> {
    pub position: IVec3,
    pub loading_radius: i32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> ChunkLoader<T> {
    pub fn new(loading_radius: i32) -> Self {
        Self {
            position: IVec3::ZERO,
            loading_radius,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Resource, Debug)]
pub struct ChunkMaterial<Sampler, Material: Asset> {
    pub material: Handle<Material>,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler, Material: Asset> ChunkMaterial<Sampler, Material> {
    pub fn new(material: Handle<Material>) -> Self {
        Self {
            material,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Controls for whether the generator should be running.
#[derive(Debug, PartialEq, Eq)]
pub enum ChunkGeneratorRunning {
    /// Generator will run as usual.
    Run,
    /// Generator will pause.
    Pause,
    /// Generator will stop and caches will be deleted. Loaded chunks will not be deleted!
    Stop,
    /// Generator will reset, refreshing caches. Loaded chunks will not be deleted!
    Reset,
}

#[derive(Resource, ExtractResource, Debug)]
pub struct ChunkGeneratorSettings<Sampler: Send + Sync + 'static> {
    pub running: ChunkGeneratorRunning,
    clear_gpu_cache: bool,
    surface_threshold: f32, // not pub so you can't change it later
    num_voxels_per_axis: u32,
    chunk_size: f32,
    max_chunks_per_frame: usize,
    num_buffers: usize,
    bounds: Option<GenBounds>,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler: Send + Sync + 'static> Clone for ChunkGeneratorSettings<Sampler> {
    fn clone(&self) -> Self {
        Self {
            running: ChunkGeneratorRunning::Run,
            clear_gpu_cache: self.clear_gpu_cache,
            surface_threshold: self.surface_threshold,
            num_voxels_per_axis: self.num_voxels_per_axis,
            chunk_size: self.chunk_size,
            max_chunks_per_frame: self.max_chunks_per_frame,
            num_buffers: self.num_buffers,
            bounds: self.bounds.clone(),
            _marker: self._marker,
        }
    }
}

#[derive(Debug, Clone)]
struct GenBounds {
    min: Vec3,
    max: Vec3,
}

impl<Sampler: Send + Sync + 'static> ChunkGeneratorSettings<Sampler> {
    pub fn new(num_voxels_per_axis: u32, chunk_size: f32) -> Self {
        Self {
            running: ChunkGeneratorRunning::Run,
            clear_gpu_cache: false,
            surface_threshold: 0.0,
            num_voxels_per_axis,
            chunk_size,
            max_chunks_per_frame: 1,
            num_buffers: 3,
            bounds: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_surface_threshold(mut self, surface_threshold: f32) -> Self {
        self.surface_threshold = surface_threshold;
        self
    }

    pub fn with_bounds(mut self, min: Vec3, max: Vec3) -> Self {
        self.bounds = Some(GenBounds { min, max });
        self
    }

    pub fn with_max_chunks_per_frame(mut self, max_chunks_per_frame: usize) -> Self {
        self.max_chunks_per_frame = max_chunks_per_frame;
        self
    }

    pub fn with_num_buffers(mut self, num_buffers: usize) -> Self {
        self.num_buffers = num_buffers;
        self
    }

    pub fn stopped(mut self) -> Self {
        self.running = ChunkGeneratorRunning::Stop;
        self
    }

    pub fn num_samples_per_axis(&self) -> u32 {
        self.num_voxels_per_axis + 3 // We sample the next chunk over too for normals
    }

    pub fn max_num_vertices(&self) -> u64 {
        self.max_num_triangles() * 3
    }

    pub fn vertices_buffer_size(&self) -> NonZero<u64> {
        (size_of::<Vertex>() as u64 * self.max_num_vertices())
            .try_into()
            .expect("zero vertices")
    }

    pub fn max_num_triangles(&self) -> u64 {
        (self.num_voxels_per_axis as u64).pow(3) * 5
    }

    pub fn triangles_buffer_size(&self) -> NonZero<u64> {
        (size_of::<Triangle>() as u64 * self.max_num_triangles())
            .try_into()
            .expect("zero triangles")
    }

    pub fn voxel_size(&self) -> f32 {
        self.chunk_size / self.num_voxels_per_axis as f32
    }

    pub fn position_to_chunk(&self, position: Vec3) -> IVec3 {
        (position / self.chunk_size).floor().as_ivec3()
    }

    pub fn chunk_to_position(&self, chunk: IVec3) -> Vec3 {
        chunk.as_vec3() * self.chunk_size
    }

    fn is_chunk_in_bounds(&self, chunk_position: IVec3) -> bool {
        if let Some(bounds) = &self.bounds {
            let position = self.chunk_to_position(chunk_position);
            position.x >= bounds.min.x
                && position.x <= bounds.max.x
                && position.y >= bounds.min.y
                && position.y <= bounds.max.y
                && position.z >= bounds.min.z
                && position.z <= bounds.max.z
        } else {
            true
        }
    }
}

#[derive(Resource, Debug, Clone)]
pub struct ChunkGeneratorCache<Sampler> {
    loaded_chunks: HashMap<IVec3, LoadState>,
    chunks_to_load: Vec<IVec3>,
    buffer_cache: HashMap<BufferCache, BufferCacheAvailable>,
    max_chunks_per_frame: usize,
    paused: bool,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler: Send + Sync + 'static> ChunkGeneratorCache<Sampler> {
    fn new(
        settings: &ChunkGeneratorSettings<Sampler>,
        buffers: &mut Assets<ShaderStorageBuffer>,
    ) -> Self {
        Self {
            loaded_chunks: default(),
            chunks_to_load: default(),
            buffer_cache: (0..settings.num_buffers)
                .map(|_| {
                    (
                        BufferCache::new(
                            settings.vertices_buffer_size(),
                            settings.triangles_buffer_size(),
                            buffers,
                        ),
                        BufferCacheAvailable::Available,
                    )
                })
                .collect(),
            max_chunks_per_frame: settings.max_chunks_per_frame,
            paused: matches!(settings.running, ChunkGeneratorRunning::Pause),
            _marker: default(),
        }
    }

    pub fn is_chunk_marked(
        &self,
        settings: &ChunkGeneratorSettings<Sampler>,
        chunk_position: IVec3,
    ) -> bool {
        !settings.is_chunk_in_bounds(chunk_position)
            || self.loaded_chunks.contains_key(&chunk_position)
    }

    pub fn is_chunk_generated(
        &self,
        settings: &ChunkGeneratorSettings<Sampler>,
        chunk_position: IVec3,
    ) -> bool {
        !settings.is_chunk_in_bounds(chunk_position)
            || matches!(
                self.loaded_chunks.get(&chunk_position),
                Some(LoadState::Finished)
            )
    }

    pub fn is_chunk_with_position_marked(
        &self,
        settings: &ChunkGeneratorSettings<Sampler>,
        position: Vec3,
    ) -> bool {
        self.is_chunk_marked(settings, settings.position_to_chunk(position))
    }

    pub fn is_chunk_with_position_generated(
        &self,
        settings: &ChunkGeneratorSettings<Sampler>,
        position: Vec3,
    ) -> bool {
        self.is_chunk_generated(settings, settings.position_to_chunk(position))
    }

    fn drain_chunks_to_load(&mut self) -> impl Iterator<Item = (IVec3, BufferCache)> {
        let num_chunks = if self.paused {
            0
        } else {
            self.chunks_to_load
                .len()
                .min(
                    self.buffer_cache
                        .iter()
                        .filter(|(_, a)| matches!(a, BufferCacheAvailable::Available))
                        .count(),
                )
                .min(self.max_chunks_per_frame)
        };

        let chunks = self.chunks_to_load.drain(..num_chunks);

        let buffers = self
            .buffer_cache
            .iter()
            .filter(|(_, a)| matches!(a, BufferCacheAvailable::Available))
            .map(|(b, _)| b.clone())
            .take(num_chunks)
            .collect::<Vec<_>>();
        for buffer in buffers.iter() {
            self.buffer_cache
                .insert(buffer.clone(), BufferCacheAvailable::Unavailable);
        }

        chunks.zip(buffers)
    }

    fn return_buffer(&mut self, buffer_cache: &BufferCache) {
        if self.buffer_cache.contains_key(buffer_cache) {
            self.buffer_cache
                .insert(buffer_cache.clone(), BufferCacheAvailable::Available);
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LoadState {
    Loading,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferCache {
    vertices: Handle<ShaderStorageBuffer>,
    num_vertices: Handle<ShaderStorageBuffer>,
    triangles: Handle<ShaderStorageBuffer>,
    num_triangles: Handle<ShaderStorageBuffer>,
}

impl BufferCache {
    fn new(
        vertices_buffer_size: NonZero<u64>,
        triangles_buffer_size: NonZero<u64>,
        buffers: &mut Assets<ShaderStorageBuffer>,
    ) -> Self {
        let vertices_buffer_size: u64 = vertices_buffer_size.into();
        let mut vertices = ShaderStorageBuffer::with_size(
            vertices_buffer_size as usize,
            RenderAssetUsages::default(),
        );
        vertices.buffer_description.usage |= BufferUsages::COPY_SRC;

        let mut num_vertices =
            ShaderStorageBuffer::with_size(size_of::<u32>(), RenderAssetUsages::default());
        num_vertices.buffer_description.usage |= BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        let triangles_buffer_size: u64 = triangles_buffer_size.into();
        let mut triangles = ShaderStorageBuffer::with_size(
            triangles_buffer_size as usize,
            RenderAssetUsages::default(),
        );
        triangles.buffer_description.usage |= BufferUsages::COPY_SRC;

        let mut num_triangles =
            ShaderStorageBuffer::with_size(size_of::<u32>(), RenderAssetUsages::default());
        num_triangles.buffer_description.usage |= BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        Self {
            vertices: buffers.add(vertices),
            num_vertices: buffers.add(num_vertices),
            triangles: buffers.add(triangles),
            num_triangles: buffers.add(num_triangles),
        }
    }
}

#[derive(Debug, Clone)]
enum BufferCacheAvailable {
    Available,
    Unavailable,
}

#[derive(Resource, Debug, Clone)]
pub struct GpuChunkGeneratorCache<Sampler> {
    buffer_cache: HashMap<BufferCache, GpuBufferCache>,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler> Default for GpuChunkGeneratorCache<Sampler> {
    fn default() -> Self {
        Self {
            buffer_cache: default(),
            _marker: default(),
        }
    }
}

#[derive(Debug, Clone)]
struct GpuBufferCache {
    chunk_position_buffer: Buffer,
    sample_pipeline: CachedComputePipelineId,
    sample_workgroups: u32,
    sample_bind_group: BindGroup,
    march_pipeline: CachedComputePipelineId,
    march_workgroups: u32,
    march_bind_group: BindGroup,
}

impl GpuBufferCache {
    fn new<Sampler: Send + Sync + 'static>(
        chunk: &ChunkRenderData<Sampler>,
        extra_buffers: Option<&ChunkRenderExtraBuffers>,
        settings: &ChunkGeneratorSettings<Sampler>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        buffers: &RenderAssets<GpuShaderStorageBuffer>,
        pipelines: &ChunkGeneratorComputePipelines<Sampler>,
    ) -> Self {
        let num_voxels_per_axis = settings.num_voxels_per_axis;
        let num_samples_per_axis = settings.num_samples_per_axis();
        let chunk_size = settings.chunk_size;
        let surface_threshold = settings.surface_threshold;
        let sample_workgroups = (num_samples_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;
        let march_workgroups = (num_voxels_per_axis as f32 / WORKGROUP_SIZE as f32).ceil() as u32;

        let mut chunk_position_buffer = UniformBuffer::from(chunk.position);
        chunk_position_buffer.write_buffer(render_device, render_queue);

        let mut settings_buffer = UniformBuffer::from(MeshSettings {
            num_voxels_per_axis,
            num_samples_per_axis,
            chunk_size,
            surface_threshold,
        });
        settings_buffer.write_buffer(render_device, render_queue);

        let mut densities_buffer = StorageBuffer::from(vec![
            0.0f32;
            settings.num_samples_per_axis().pow(3)
                as usize
        ]);
        densities_buffer.write_buffer(render_device, render_queue);

        let vertices_buffer = buffers.get(&chunk.buffers.vertices).unwrap();
        let num_vertices_buffer = buffers.get(&chunk.buffers.num_vertices).unwrap();
        let triangles_buffer = buffers.get(&chunk.buffers.triangles).unwrap();
        let num_triangles_buffer = buffers.get(&chunk.buffers.num_triangles).unwrap();

        let sample_bind_group = render_device.create_bind_group(
            Some("marching cubes sample bind group"),
            &pipelines.sample_layout,
            &[
                chunk_position_buffer.binding().unwrap(),
                settings_buffer.binding().unwrap(),
                densities_buffer.binding().unwrap(),
            ]
            .into_iter()
            .chain(
                extra_buffers
                    .map(|b| &b.buffers)
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|b| b.as_entire_binding()),
            )
            .enumerate()
            .map(|(i, res)| BindGroupEntry {
                binding: i as u32,
                resource: res,
            })
            .collect::<Vec<_>>(),
        );

        let march_bind_group = render_device.create_bind_group(
            Some("marching cubes march bind group"),
            &pipelines.march_layout,
            &BindGroupEntries::sequential((
                densities_buffer.binding().unwrap(),
                settings_buffer.binding().unwrap(),
                vertices_buffer.buffer.as_entire_binding(),
                num_vertices_buffer.buffer.as_entire_binding(),
                triangles_buffer.buffer.as_entire_binding(),
                num_triangles_buffer.buffer.as_entire_binding(),
            )),
        );

        GpuBufferCache {
            chunk_position_buffer: chunk_position_buffer.buffer().unwrap().clone(),
            sample_pipeline: pipelines.sample_pipeline,
            sample_workgroups,
            sample_bind_group,
            march_pipeline: pipelines.march_pipeline,
            march_workgroups,
            march_bind_group,
        }
    }
}

fn init_cache<Sampler: Send + Sync + 'static>(
    mut commands: Commands,
    mut settings: ResMut<ChunkGeneratorSettings<Sampler>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    cache: Option<ResMut<ChunkGeneratorCache<Sampler>>>,
) {
    settings.clear_gpu_cache = matches!(
        settings.running,
        ChunkGeneratorRunning::Stop | ChunkGeneratorRunning::Reset
    );

    if matches!(
        settings.running,
        ChunkGeneratorRunning::Stop | ChunkGeneratorRunning::Reset
    ) && cache.is_some()
    {
        commands.remove_resource::<ChunkGeneratorCache<Sampler>>();
    }

    if matches!(settings.running, ChunkGeneratorRunning::Reset) {
        settings.running = ChunkGeneratorRunning::Run;
    }

    if matches!(
        settings.running,
        ChunkGeneratorRunning::Run | ChunkGeneratorRunning::Pause
    ) {
        if let Some(mut cache) = cache {
            cache.paused = matches!(settings.running, ChunkGeneratorRunning::Pause);
        } else {
            commands.insert_resource(ChunkGeneratorCache::<Sampler>::new(&settings, &mut buffers));
        }
    }
}

fn update_chunk_loaders<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut chunk_loaders: Query<
        (&mut ChunkLoader<Sampler>, &GlobalTransform),
        Changed<GlobalTransform>,
    >,
) {
    for (mut chunk_loader, transform) in chunk_loaders.iter_mut() {
        let chunk_position = (transform.translation() / settings.chunk_size)
            .floor()
            .as_ivec3();

        // Properly update change detection
        if chunk_loader.position != chunk_position {
            chunk_loader.position = chunk_position;
        }
    }
}

fn queue_chunks<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut cache: ResMut<ChunkGeneratorCache<Sampler>>,
    chunk_loaders: Query<&ChunkLoader<Sampler>, Changed<ChunkLoader<Sampler>>>,
) {
    for chunk_loader in chunk_loaders.iter() {
        let mut load_order = Vec::new();
        for x in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
            for y in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                for z in -chunk_loader.loading_radius..=chunk_loader.loading_radius {
                    load_order.push(Vec3::new(x as f32, y as f32, z as f32));
                }
            }
        }

        load_order.sort_by(|a, b| {
            // Sort ascending so that the closest chunks are loaded first (drained from front)
            a.length_squared()
                .partial_cmp(&b.length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for offset in load_order {
            let chunk_position = chunk_loader.position + offset.as_ivec3();
            if !cache.is_chunk_marked(&settings, chunk_position) {
                cache
                    .loaded_chunks
                    .insert(chunk_position, LoadState::Loading);
                cache.chunks_to_load.push(chunk_position);
                info!("Queued chunk for loading: {chunk_position:?}");
            }
        }
    }
}

fn start_chunks<
    Sampler: ChunkComputeShader + Send + Sync + 'static,
    Material: Asset + bevy::prelude::Material,
>(
    mut commands: Commands,
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut cache: ResMut<ChunkGeneratorCache<Sampler>>,
    material: Res<ChunkMaterial<Sampler, Material>>,
) {
    for (chunk_position, buffers) in cache.drain_chunks_to_load() {
        info!("start_chunks {chunk_position:?}");

        let chunk_entity = commands
            .spawn((
                Name::new(format!("Chunk {chunk_position:?}")),
                Transform::from_translation(settings.chunk_to_position(chunk_position)),
                MeshMaterial3d(material.material.clone()),
                Chunk::<Sampler> {
                    position: chunk_position,
                    _marker: default(),
                },
                ChunkGenData {
                    vertices: None,
                    triangles: None,
                },
                ChunkRenderData::<Sampler> {
                    position: chunk_position,
                    buffers: buffers.clone(),
                    _marker: default(),
                },
            ))
            .observe(finish_chunk::<Sampler>)
            .id();

        commands
            .spawn((
                Name::new(format!("Chunk {chunk_position:?} num_vertices readback")),
                Readback::buffer(buffers.num_vertices),
                ChildOf(chunk_entity),
            ))
            .observe(
                move |readback: On<ReadbackComplete>,
                      mut chunks: Query<&mut ChunkGenData>,
                      mut commands: Commands|
                      -> Result {
                    let mut chunk = chunks.get_mut(chunk_entity)?;
                    let num_vertices: u32 = readback.to_shader_type();
                    info!("num_vertices readback {chunk_position:?} {num_vertices}");
                    commands.entity(readback.entity).despawn();
                    if num_vertices > 0 {
                        commands
                            .spawn((
                                Name::new(format!("Chunk {chunk_position:?} vertices readback")),
                                Readback::buffer_range(
                                    buffers.vertices.clone(),
                                    0,
                                    size_of::<Vertex>() as u64 * num_vertices as u64,
                                ),
                                ChildOf(chunk_entity),
                            ))
                            .observe(
                                move |readback: On<ReadbackComplete>,
                                      mut chunks: Query<&mut ChunkGenData>,
                                      mut commands: Commands|
                                      -> Result {
                                    let vertices: Vec<Vertex> = readback.to_shader_type();
                                    info!(
                                        "vertices readback {chunk_position:?} {}",
                                        vertices.len()
                                    );
                                    let mut chunk = chunks.get_mut(chunk_entity)?;
                                    chunk.vertices = Some(vertices);
                                    commands.trigger(ReadbackReallyComplete(chunk_entity));
                                    commands.entity(readback.entity).despawn();
                                    Ok(())
                                },
                            );
                    } else {
                        chunk.vertices = Some(vec![]);
                        commands.trigger(ReadbackReallyComplete(chunk_entity));
                    }
                    Ok(())
                },
            );

        commands
            .spawn((
                Name::new(format!("Chunk {chunk_position:?} num_triangles readback")),
                Readback::buffer(buffers.num_triangles),
                ChildOf(chunk_entity),
            ))
            .observe(
                move |readback: On<ReadbackComplete>,
                      mut chunks: Query<&mut ChunkGenData>,
                      mut commands: Commands|
                      -> Result {
                    let mut chunk = chunks.get_mut(chunk_entity)?;
                    let num_triangles: u32 = readback.to_shader_type();
                    info!("num_triangles readback {chunk_position:?} {num_triangles}");
                    commands.entity(readback.entity).despawn();
                    if num_triangles > 0 {
                        commands
                            .spawn((
                                Name::new(format!("Chunk {chunk_position:?} triangles readback")),
                                Readback::buffer_range(
                                    buffers.triangles.clone(),
                                    0,
                                    size_of::<Triangle>() as u64 * num_triangles as u64,
                                ),
                                ChildOf(chunk_entity),
                            ))
                            .observe(
                                move |readback: On<ReadbackComplete>,
                                      mut chunks: Query<&mut ChunkGenData>,
                                      mut commands: Commands|
                                      -> Result {
                                    let triangles: Vec<Triangle> = readback.to_shader_type();
                                    info!(
                                        "triangles readback {chunk_position:?} {}",
                                        triangles.len()
                                    );
                                    let mut chunk = chunks.get_mut(chunk_entity)?;
                                    chunk.triangles = Some(triangles);
                                    commands.trigger(ReadbackReallyComplete(chunk_entity));
                                    commands.entity(readback.entity).despawn();
                                    Ok(())
                                },
                            );
                    } else {
                        chunk.triangles = Some(vec![]);
                        commands.trigger(ReadbackReallyComplete(chunk_entity));
                    }
                    Ok(())
                },
            );
    }
}

#[derive(Component)]
pub struct ChunkRenderExtraBuffers {
    pub buffers: Vec<Buffer>,
}

fn clear_old_dispatches(mut dispatches: ResMut<ChunkGeneratorDispatches>) {
    dispatches.dispatches.clear();
}

fn prepare_bind_groups<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    chunks: Query<(&ChunkRenderData<Sampler>, Option<&ChunkRenderExtraBuffers>)>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    pipelines: Res<ChunkGeneratorComputePipelines<Sampler>>,
    pipeline_cache: Res<PipelineCache>,
    mut dispatches: ResMut<ChunkGeneratorDispatches>,
    mut cache: ResMut<GpuChunkGeneratorCache<Sampler>>,
    mut pipelines_done_loading_res: ResMut<ChunkGeneratorComputePipelinesDoneLoading<Sampler>>,
    mut processed: Local<HashSet<IVec3>>,
) -> Result<()> {
    let pipelines_done_loading = pipeline_cache
        .get_compute_pipeline(pipelines.sample_pipeline)
        .is_some()
        && pipeline_cache
            .get_compute_pipeline(pipelines.march_pipeline)
            .is_some();
    pipelines_done_loading_res.done = pipelines_done_loading;
    if !pipelines_done_loading {
        return Ok(());
    }

    for (chunk, extra_buffers) in chunks.iter() {
        if processed.contains(&chunk.position) {
            continue;
        }
        processed.insert(chunk.position);

        info!(
            "prepare_bind_groups {} {} with {:?} extra buffers, currently {} buffer sets loaded",
            ShortName::of::<Sampler>(),
            chunk.position,
            extra_buffers.map(|b| b.buffers.len()),
            cache.buffer_cache.len()
        );

        let buffer_cache = cache
            .buffer_cache
            .entry(chunk.buffers.clone())
            .or_insert_with(|| {
                GpuBufferCache::new(
                    chunk,
                    extra_buffers,
                    &settings,
                    &render_device,
                    &render_queue,
                    &buffers,
                    &pipelines,
                )
            })
            .clone();

        let mut writer = encase::StorageBuffer::<Vec<u8>>::new(Vec::new());
        writer.write(&chunk.position)?;
        render_queue.write_buffer(&buffer_cache.chunk_position_buffer, 0, writer.as_ref());

        let mut writer = encase::StorageBuffer::<Vec<u8>>::new(Vec::new());
        writer.write(&0u32)?;
        let num_vertices_buffer = buffers.get(&chunk.buffers.num_vertices).unwrap();
        render_queue.write_buffer(&num_vertices_buffer.buffer, 0, writer.as_ref());

        let mut writer = encase::StorageBuffer::<Vec<u8>>::new(Vec::new());
        writer.write(&0u32)?;
        let num_triangles_buffer = buffers.get(&chunk.buffers.num_triangles).unwrap();
        render_queue.write_buffer(&num_triangles_buffer.buffer, 0, writer.as_ref());

        dispatches.dispatches.push(buffer_cache);
    }

    Ok(())
}

#[derive(EntityEvent)]
struct ReadbackReallyComplete(Entity);

fn finish_chunk<Sampler: ChunkComputeShader + Send + Sync + 'static>(
    readback: On<ReadbackReallyComplete>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cache: ResMut<ChunkGeneratorCache<Sampler>>,
    chunks: Query<(&ChunkRenderData<Sampler>, &ChunkGenData)>,
) -> Result {
    let Ok((
        &ChunkRenderData {
            position: chunk_position,
            ref buffers,
            ..
        },
        chunk,
    )) = chunks.get(readback.0)
    else {
        return Ok(()); // It's already done
    };

    info!("finish_chunk {chunk_position}");

    let Some(ref vertices) = chunk.vertices else {
        return Ok(());
    };
    let Some(ref triangles) = chunk.triangles else {
        return Ok(());
    };

    if !vertices.is_empty() && !triangles.is_empty() {
        let mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_indices(bevy::mesh::Indices::U32(
            triangles
                .iter()
                .flat_map(|t| [t.vertex_c, t.vertex_b, t.vertex_a])
                .collect(),
        ))
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vertices.iter().map(|v| v.normal).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vertices.iter().map(|v| v.position.xy()).collect::<Vec<_>>(),
        );

        commands.entity(readback.0).insert(Mesh3d(meshes.add(mesh)));
    }
    cache
        .loaded_chunks
        .insert(chunk_position, LoadState::Finished);
    cache.return_buffer(buffers);

    commands
        .entity(readback.0)
        .remove::<ChunkGenData>()
        .remove::<ChunkRenderData<Sampler>>();

    Ok(())
}

fn add_compute_render_graph_node(mut render_graph: ResMut<RenderGraph>) {
    render_graph.add_node(ChunkGeneratorNodeLabel, ChunkGeneratorNode);
    // add_node_edge guarantees that ComputeNodeLabel will run before CameraDriverLabel
    render_graph.add_node_edge(
        ChunkGeneratorNodeLabel,
        bevy::render::graph::CameraDriverLabel,
    );
}

fn clear_gpu_cache<Sampler: Send + Sync + 'static>(
    settings: Res<ChunkGeneratorSettings<Sampler>>,
    mut cache: ResMut<GpuChunkGeneratorCache<Sampler>>,
) {
    if settings.clear_gpu_cache {
        cache.buffer_cache.clear();
    }
}

#[derive(Resource, Debug)]
struct ChunkGeneratorComputePipelinesDoneLoading<Sampler> {
    done: bool,
    _marker: std::marker::PhantomData<Sampler>,
}

impl<Sampler> Default for ChunkGeneratorComputePipelinesDoneLoading<Sampler> {
    fn default() -> Self {
        Self {
            done: false,
            _marker: default(),
        }
    }
}

impl<Sampler> Clone for ChunkGeneratorComputePipelinesDoneLoading<Sampler> {
    fn clone(&self) -> Self {
        Self {
            done: self.done,
            _marker: self._marker,
        }
    }
}

fn extract_pipelines_done<Sampler: Send + Sync + 'static>(
    render_resource: Option<Res<ChunkGeneratorComputePipelinesDoneLoading<Sampler>>>,
    mut main_world: ResMut<MainWorld>,
) {
    if let Some(render_resource) = render_resource.as_ref() {
        if let Some(mut target_resource) =
            main_world.get_resource_mut::<ChunkGeneratorComputePipelinesDoneLoading<Sampler>>()
        {
            if render_resource.is_changed() {
                *target_resource = (*render_resource).clone();
            }
        } else {
            main_world.insert_resource((*render_resource).clone());
        }
    }
}
