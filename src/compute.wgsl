struct Point {
    u: f32,
    v: f32,
    s: f32,
    rho: f32,
    avg_div: f32,
    nu: f32,
    nv: f32,
    nrho: f32,
}

@group(0) @binding(0)
var<storage, read_write> field: array<Point>;
@group(1) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;

struct Ids {
    @builtin(local_invocation_index) local_index: u32,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
}

fn get_id(global_id: vec3<u32>, num_groups: vec3<u32>, group_size: u32) -> u32 {
    let row = num_groups.x * group_size;
    return global_id.y * row + global_id.x;
}

@compute @workgroup_size(16, 16)
fn integrate(ids: Ids) {
    // TODO pass dt & g as constants
    let dt = 0.01;
    let g = -9.81;
    // TODO pass size as a constant
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    if field[index].s == 1.0 && field[index - row_size].s == 1.0 {
        field[index].v += g * dt;
    }
}

@compute @workgroup_size(16, 16)
fn gather(ids: Ids) {
    // TODO pass size as a constant
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    //let index = get_id(ids.global_id, ids.num_groups, 1u);
    let here = &field[index];
    if (*here).s == 1.0 {
        let left = field[index - 1u];
        let right = field[index + 1u];
        let bottom = field[index - row_size];
        let top = field[index + row_size];
        let s = left.s + right.s + bottom.s + top.s;
        let avg_div = (right.u - (*here).u + top.v - (*here).v) / s;
        (*here).avg_div = avg_div;
    }
}

@compute @workgroup_size(16, 16)
fn scatter_bottom_left(ids: Ids) {
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    if field[index].s == 1.0 {
        let left = field[index - 1u];
        let bottom = field[index - row_size];
        let avg_div = field[index].avg_div;
        field[index].u += left.s * avg_div;
        field[index].v += bottom.s * avg_div;
    }
}

@compute @workgroup_size(16, 16)
fn scatter_top_right(ids: Ids) {
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    if field[index].s == 1.0 {
        let right = &field[index + 1u];
        let top = &field[index + row_size];
        let avg_div = field[index].avg_div;
        (*right).u -= (*right).s * avg_div;
        (*top).v -= (*top).s * avg_div;
    }
}

var<workgroup> foo: atomic<u32>;

@compute @workgroup_size(16, 16)
fn advect_u(ids: Ids) {
    let dt = 0.01;
    let cell_size = 0.1;
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    let here = field[index];
    let p = &foo;
    let val = atomicLoad(p);
    field[index].nu = here.u;
    if here.s == 1.0 && field[index - 1u].s == 1.0 {
        let u = here.u;
        let v = (here.v + field[index - 1u].v + field[index + row_size].v + field[index + row_size - 1u].v) / 4.0;
        // TODO try abstracting field sampling
        let p = vec2<f32>(ids.global_id.xy) - vec2(u, v) * dt / cell_size;
        let pf = vec2<u32>(floor(p));
        let p_id = clamp(vec2<u32>(floor(p)), vec2(1u, 1u), vec2(row_size - 1u, row_size - 1u));
        let b = p - floor(p);
//        let a = 1.0 - b;
        let a = vec2(1.0, 1.0) - b;
        let p_index = p_id.y * row_size + p_id.x;
        let nu =
              field[p_index].u * a.x * a.y
            + field[p_index + 1u].u * b.x * a.y
            + field[p_index + row_size].u * a.x * b.y
            + field[p_index + row_size + 1u].u * b.x * b.y;
        field[index].nu = nu;
    }
}

@compute @workgroup_size(16, 16)
fn advect_v(ids: Ids) {
    let dt = 0.01;
    let cell_size = 0.1;
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    let here = field[index];
    field[index].nv = here.v;
    if here.s == 1.0 && field[index - row_size].s == 1.0 {
        let u = (here.u + field[index - row_size].u + field[index + 1u].u + field[index - row_size + 1u].u) / 4.0;
        let v = here.v;
        let p = vec2<f32>(ids.global_id.xy) - vec2(u, v) * dt / cell_size;
        let p_id = clamp(vec2<u32>(floor(p)), vec2(1u, 1u), vec2(row_size - 1u, row_size - 1u));
        let b = p - floor(p);
        let a = 1.0 - b;
        let p_index = p_id.y * row_size + p_id.x;
        let nv =
              field[p_index].v * a.x * a.y
            + field[p_index + 1u].v * b.x * a.y
            + field[p_index + row_size].v * a.x * b.y
            + field[p_index + row_size + 1u].v * b.x * b.y;
        field[index].nv = nv;
    }
}

@compute @workgroup_size(16, 16)
fn advect_density(ids: Ids) {
    let dt = 0.01;
    let cell_size = 0.1;
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    let here = field[index];
    field[index].nrho = here.rho;
    if here.s == 1.0 {
        let u = (here.u + field[index + 1u].u) / 2.0;
        let v = (here.v + field[index + row_size].v) / 2.0;
        let p = vec2<f32>(ids.global_id.xy) - vec2(u, v) * dt / cell_size;
        let p_id = clamp(vec2<u32>(floor(p)), vec2(1u, 1u), vec2(row_size - 1u, row_size - 1u));
        let b = p - floor(p);
        let a = 1.0 - b;
        let p_index = p_id.y * row_size + p_id.x;
        let nrho =
              field[p_index].rho * a.x * a.y
            + field[p_index + 1u].rho * b.x * a.y
            + field[p_index + row_size].rho * a.x * b.y
            + field[p_index + row_size + 1u].rho * b.x * b.y;
        field[index].nrho = nrho;
    }
}

@compute @workgroup_size(16, 16)
fn copy(ids: Ids) {
    let id = get_id(ids.global_id, ids.num_groups, 16u);
    let f = field[id];
    field[id].u = f.nu;
    field[id].v = f.nv;
    // TODO Copy velocity before advecting density
    field[id].rho = f.nrho;
    let f = field[id];
    let speed = length(vec2(f.u, f.v));
    let norm = 10.0;
    //let color = vec4(f.u / norm, 0.0, -f.u / norm, f.s);
    let color = vec4(abs(f.u) / norm, 0.0, abs(f.v) / norm, f.s);
    //let color = vec4(f.rho, f.rho, f.rho, 1.0);
    textureStore(output, vec2<i32>(ids.global_id.xy), color);
}