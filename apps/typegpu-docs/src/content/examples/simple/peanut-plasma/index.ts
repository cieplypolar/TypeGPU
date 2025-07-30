import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { perlin3d } from '@typegpu/noise';

// == INIT ==
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu') as GPUCanvasContext;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

const root = await tgpu.init();

context.configure({
  device: root.device,
  format: presentationFormat,
  alphaMode: 'premultiplied',
});

const time = root.createUniform(d.f32);
const resolution = root.createUniform(d.vec2f);

// == VERTEX SHADER ==
const vertexMain = tgpu['~unstable'].vertexFn({
  in: { idx: d.builtin.vertexIndex },
  out: { pos: d.builtin.position, uv: d.vec2f },
})(({ idx }) => {
  const pos = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  const uv = [d.vec2f(0, 0), d.vec2f(2, 0), d.vec2f(0, 2)];

  return {
    pos: d.vec4f(pos[idx], 0.0, 1.0),
    uv: uv[idx],
  };
});

// == INTERESTING PART (FRAGMENT SHADER) ==
const peanut = d.vec3f(0.76, 0.58, 0.29);
const jam = d.vec3f(0.3, 0, 0);

const noise = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((p, t) => {
  return perlin3d.sample(d.vec3f(p, t * 0.1)) + 0.25; // perlin noise wa too dark
});

const numOctaves = 4;

const fbm = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((uv, t) => {
  let f = d.f32(1.0);
  let a = d.f32(1.0);
  let R = d.f32(0.0);
  for (let i = 0; i < numOctaves; i++) {
    R += a * noise(std.mul(uv, f), t);
    f *= 2.0;
    a *= 0.5;
  }
  return d.f32(R);
});

const domainWarp = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((uv, t) => {
  const q = d.vec2f(
    fbm(std.add(uv, d.vec2f(0, 0)), t),
    fbm(std.add(uv, d.vec2f(5.2, 1.3)), t),
  );

  const r = d.vec2f(
    fbm(std.add(std.add(uv, std.mul(2, q)), d.vec2f(1.7, 9.2)), t),
    fbm(std.add(std.add(uv, std.mul(2, q)), d.vec2f(8.3, 2.8)), t),
  );

  return fbm(std.add(uv, std.mul(2, r)), t);
});

const zoom = d.f32(2.5); // zoom out

const fragmentMain = tgpu['~unstable'].fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  let scaledUV = std.sub(std.mul(input.uv, 2), 1);
  scaledUV.x *= resolution.$.x / resolution.$.y;

  scaledUV = std.mul(scaledUV, zoom);

  const color = std.mix(
    peanut,
    jam,
    std.clamp(domainWarp(scaledUV, time.$), 0, 1),
  );
  return d.vec4f(color, 1);
});

// == RENDER PIPELINE ==
const cache = perlin3d.staticCache({ root, size: d.vec3u(7, 7, 7) });
const renderPipeline = root['~unstable']
  .pipe(cache.inject())
  .withVertex(vertexMain, {})
  .withFragment(fragmentMain, { format: presentationFormat })
  .createPipeline();

let animationFrame: number;
function run(timestamp: number) {
  time.write((timestamp / 1000) % 1000);
  resolution.write(d.vec2f(canvas.width, canvas.height));

  renderPipeline
    .withColorAttachment({
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      storeOp: 'store',
    })
    .draw(3);

  animationFrame = requestAnimationFrame(run);
}

animationFrame = requestAnimationFrame(run);

// #region Example controls and cleanup
export function onCleanup() {
  cancelAnimationFrame(animationFrame);
  root.destroy();
}
// #endregion
