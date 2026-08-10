# surge

A production LLM inference engine in C and Metal, built for one machine: the
Mac Studio M3 Ultra. No dependencies beyond what macOS ships.

surge exists because this specific machine was measured to death first. It grew
out of the llm-rnd experimentation campaign: the 630 GB/s streaming-bandwidth
and 21.9 TFLOPS ceilings, the firmware GPU power limiter (clamps to 338 MHz
after ~3 minutes of sustained load, releases after 60 to 120 seconds, die
temperature is an input), and a benchmarking methodology hardened against all
of it. The engine is designed around those facts rather than discovering them
at runtime: a limiter-aware pacing scheduler, clamp detection, an optional
[fanpro](https://github.com/omar16100/fanpro) pre-spin hook, byte-identical
prompt-lookup speculation, and kernels tuned against the measured roofline.

Background reading: [macOS clamps my M3 Ultra's GPU to 338 MHz before the fans
even try](https://omarshabab.com/mac-studio-firmware-gpu-limiter/), and the
benchmark harness at
[llm-benchmark](https://github.com/omar16100/llm-benchmark).

Status: M0-M2 complete (loads hybrid qwen3_5 GGUF+safetensors, CPU reference
passes M1 vs mlx-lm, Metal decode passes M2 byte-exact; ~76 tok/s decode on
Qwen3.5-2B). First target: Qwen3.6-27B hybrid (full-attention + gated
DeltaNet) at 8-bit, with the stated bar of beating mlx-lm single-stream
decode on the same quant on this hardware. Design spec:
`docs/superpowers/specs/2026-08-08-surge-design.md`.

License: MIT.
