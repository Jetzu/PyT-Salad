# P2: why the lean step can change the written structure, and the fix if it does

Measured end to end (job 903106, three designs, same `--seed 1234`):

    allon  (lean on)   design 0  CA-CA 3.767  Rg 13.2
    nolean (lean off)  design 0  CA-CA 3.763  Rg 15.2

Different folds, not different rounding. The RNG stream is identical in both
(`apply_diffusion` draws the same shapes, `FastDenoiser.forward` draws the same
`(draws, N, N)` uniform, and the three heads consume no randomness), and
`data["seq"]`/`["atom_pos"]`/`["aatype"]` are never read back by the denoiser
for any released config (`diffuse_sequence=False` everywhere). So the only
channel left is `pos` itself.

`need_aa` is a Dynamo constant, so `need_aa=False` is a SEPARATE inductor
compilation. Inductor is free to schedule, tile and fuse the shared trunk
differently when the graph has three outputs instead of eleven - and a
last-ulp difference in `pos` fed back 400 times is a different protein
(notes/04 section 2: the trajectory is chaotic by construction).

THE FIX, if the single-forward check confirms a non-zero `full vs lean` delta:
split `forward_core` into `forward_trunk` and `forward_heads`, compile each
ONCE, and capture two CUDA graphs:

  * graph T: trunk -> (pos, local, trajectory), captured once, replayed every
    step. Its codegen is identical on every step by construction, so `pos` is
    bit-identical whether or not the heads run.
  * graph H: heads, captured with its INPUTS BOUND TO T's static output
    buffers, so no copy is needed between them. Replayed only on out-steps.

That keeps P2's saving (the heads really do not run on 400 of 401 steps) while
making the lean step bit-identical to the full one in `pos` - which is what the
exact path has to guarantee.
