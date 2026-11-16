# Instructions

Your goal is to write TwoFFT, a simple and fast SIMD FFT implementation for power-of-two
single-precision out-of-place tranformations on arm64 and x86-64. The result should be a
maintainable, production ready FFT library. The implementation should be one-shot by AI (you).
There will no further user input, work independently and resolve problems and decisions by
yourself.

Writing custom instructions is necessary achieve the necessary control over register usage. To
avoid messing with calling conventions, I suggest using normal functions with a single big inline
assembly block. Inline assembly uses the GCC/Clang syntax, supporting MSVCs cl.exe is not required.

`DevelopmentPlan.md` holds a suggested development plan, edit and update it as you see fit.

`TestHardware.md` lists the benchmarking hardware available to you. Later stages will require
more X86 hardware, request it from the user when the time comes.