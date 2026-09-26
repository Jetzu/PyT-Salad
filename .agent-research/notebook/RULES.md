# Rules
- Never generate a JAX reference fixture on GPU: XLA:GPU silently lowers fp32 `dot` to TF32 unless `jax_default_matmul_precision="highest"`, which looks exactly like a large port bug (aatype still matches, coords off by 1e-2 to 3e-2 A) and isn't reproducible run-to-run. Pin `highest` and record backend+precision in every fixture. (K014; cited by N001, N005, N007, N011, N012, N015)
