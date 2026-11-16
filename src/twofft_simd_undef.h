/* Removes the macros of a twofft_simd_*.h header so that the next one can be included. */

#undef TWOFFT_ISA
#undef TWOFFT_TARGET
#undef TWOFFT_VW
#undef TWOFFT_VW_LOG2
#undef TWOFFT_LANE_ELEM
#undef vf
#undef v_load
#undef v_store
#undef v_set1
#undef v_add
#undef v_sub
#undef v_mul
#undef v_neg
#undef v_fmadd
#undef v_fnmadd
#undef V_LOAD2
#undef V_STORE2
#undef V_TRANSPOSE
#undef v4_lo64
#undef v4_hi64
#undef v4_lohi
#undef v4_even
#undef v4_odd
