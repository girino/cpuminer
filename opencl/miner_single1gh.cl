
#ifdef __ECLIPSE_EDITOR__
#include "OpenCLKernel.hpp"
#include "common.cl"
#endif

#define KERNEL_ATTRIB __attribute__(( ))

//	kernel_single_noinit->resetArgs();
//	kernel_single_noinit->addGlobalArg(u);
//	kernel_single_noinit->addGlobalArg(buff);
//	kernel_single_noinit->addLocalArg(8*wg_size*sizeof(cl_uint));
//	kernel_single_noinit->addGlobalArg(shavite_lookup);
//	kernel_single_noinit->addGlobalArg(fugue_lookup);
//	kernel_single_noinit->addGlobalArg(out);
//	kernel_single_noinit->addGlobalArg(out_count);
//	kernel_single_noinit->addGlobalArg(begin_nonce);
//	kernel_single_noinit->addGlobalArg(target);

void
sph_enc64le_aligned(local void *dst, ulong val)
{
	((local unsigned char *)dst)[0] = val;
	((local unsigned char *)dst)[1] = (val >> 8);
	((local unsigned char *)dst)[2] = (val >> 16);
	((local unsigned char *)dst)[3] = (val >> 24);
	((local unsigned char *)dst)[4] = (val >> 32);
	((local unsigned char *)dst)[5] = (val >> 40);
	((local unsigned char *)dst)[6] = (val >> 48);
	((local unsigned char *)dst)[7] = (val >> 56);
}

kernel KERNEL_ATTRIB void 
single_noinit(constant uint2* restrict in,
                   constant uint*  restrict buf,
                   local    ulong* restrict hashes,
              	   global   uint*  restrict AES,
                   global   uint*  restrict mixtab,
                   global   uint*  restrict out,
                   global   uint*  restrict outcount,
                   global   uint*  restrict begin_nonce,
                   global   uint*  restrict target)
{
    size_t id = get_global_id(0);
    size_t lid = get_local_id(0);
    uint nonce = (uint)id + *begin_nonce;
    // temp hash pointer
    local ulong* hash = hashes+(8*lid);
    // Copy global lookup table into local memory
    local uint shavite_lookup0[256];
    local uint shavite_lookup1[256];
    local uint shavite_lookup2[256];
    local uint shavite_lookup3[256];
    //local uint local_mixtab[1024];
    local uint local_mixtab0[256];
    local uint local_mixtab1[256];
    local uint local_mixtab2[256];
    local uint local_mixtab3[256];

	event_t eshavite;
	eshavite = async_work_group_copy(shavite_lookup0, AES, 256, 0);
	eshavite = async_work_group_copy(shavite_lookup1, AES+256, 256, eshavite);
	eshavite = async_work_group_copy(shavite_lookup2, AES+512, 256, eshavite);
	eshavite = async_work_group_copy(shavite_lookup3, AES+768, 256, eshavite);
    event_t efugue;
    efugue = async_work_group_copy (local_mixtab0, mixtab, 256, 0);
    efugue = async_work_group_copy (local_mixtab1, mixtab+256, 256, efugue);
    efugue = async_work_group_copy (local_mixtab2, mixtab+512, 256, efugue);
    efugue = async_work_group_copy (local_mixtab3, mixtab+768, 256, efugue);


    // keccak

    uint2 ARGS_25(state);

    state0 = in[0];
    	state1 = in[1];
    	state2 = in[2];
    	state3 = in[3];
    	state4 = in[4];
    	state5 = in[5];
    	state6 = in[6];
    	state7 = in[7];
    	state8 = in[8];
    	state9 = (uint2)(in[18].x,nonce);
    	state10 = (uint2)(1,0);
    	state11 = 0;
    	state12 = 0;
    	state13 = 0;
    	state14 = 0;
    	state15 = 0;
    	state16 = (uint2)(0,0x80000000U);
    	state17 = 0;
    	state18 = 0;
    	state19 = 0;
    	state20 = 0;
    	state21 = 0;
    	state22 = 0;
    	state23 = 0;
    	state24 = 0;

    	keccak_block_noabsorb(ARGS_25(&state));
    	/* Finalize the "lane complement" */
    	state1 = ~state1;
    	state2 = ~state2;
    	state8 = ~state8;
    	state12 = ~state12;
    	state17 = ~state17;
    	state20 = ~state20;
    	sph_enc64le_aligned(hash+0, state0);
    	sph_enc64le_aligned(hash+1, state1);
    	sph_enc64le_aligned(hash+2, state2);
    	sph_enc64le_aligned(hash+3, state3);
    	sph_enc64le_aligned(hash+4, state4);
    	sph_enc64le_aligned(hash+5, state5);
    	sph_enc64le_aligned(hash+6, state6);
    	sph_enc64le_aligned(hash+7, state7);



    wait_group_events(1, &eshavite);
    shavite((local uint *)hash,
            shavite_lookup0,
            shavite_lookup1,
            shavite_lookup2,
            shavite_lookup3);
    wait_group_events(1, &efugue);
    metis((local uint *)hash,
          local_mixtab0,
          local_mixtab1,
          local_mixtab2,
          local_mixtab3);

    // for debug
#ifdef VALIDATE_ALGORITHMS
    for (int i = 0; i < 8; i++) {
            in[(id * 8)+i] = hash[i];
    }
#endif

    if( *(local uint*)((local uchar*)hash + 28) <= *target )
    {
        out[atomic_inc(outcount)] = nonce;
    }

}

