import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { perlin3d } from "@typegpu/noise";

const canvas = document.querySelector("canvas") as HTMLCanvasElement;
const context = canvas.getContext("webgpu") as GPUCanvasContext;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

const root = await tgpu.init();

context.configure({
  device: root.device,
  format: presentationFormat,
  alphaMode: "premultiplied",
});

const time = root.createUniform(d.f32);
const resolution = root.createUniform(d.vec2f);

const vertexMain = tgpu["~unstable"].vertexFn({
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

const peanut = d.vec3f(0.75, 0.57, 0.31);
const jam = d.vec3f(0.73, 0.16, 0.09);

const noise = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((p, t) => {
  return perlin3d.sample(d.vec3f(p, t));
});

const numOctaves = 4;

const fbm = tgpu.fn(
  [d.vec2f, d.f32, d.f32],
  d.f32,
)((uv, t, H) => {
  // H may be hardcoded to 1
  const G = std.exp2(-H);
  let f = d.f32(1.0);
  let a = d.f32(1.0);
  let R = d.f32(0.0);
  for (let i = 0; i < numOctaves; i++) {
    R += a * noise(std.mul(uv, d.vec2f(f)), t);
    f *= 2.0;
    a *= G;
  }
  return d.f32(R);
});

const domainWarp = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((uv, t) => {
  let fbm1 = fbm(std.add(uv, d.vec2f(0)), t, 1);
  let fbm2 = fbm(std.add(uv, d.vec2f(5.1, 5.7)), t, 1);
  return fbm(d.vec2f(fbm1, fbm2), t, 1);
});

const fragmentMain = tgpu["~unstable"].fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  let uv = std.sub(std.mul(input.uv, 2), 1);
  uv.x *= resolution.$.x / resolution.$.y;
  uv = std.mul(uv, 2.0); // zoom out

  let color = std.mix(jam, peanut, domainWarp(uv, time.$));
  // return d.vec4f(color, 1);
  return d.vec4f(d.vec3f(fbm(uv, time.$, 1)), 1);
});

const renderPipeline = root["~unstable"]
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
      loadOp: "clear",
      storeOp: "store",
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
