/*
 * Copyright 2009 Colin Percival, 2011 ArtForz, 2011-2013 pooler
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

#include "cpuminer-config.h"
#include "miner.h"
#include "sph_keccak.h"
#include "sph_metis.h"
#include "sph_shavite.h"
#include "metiscoinminerC.h"
#include "aes_helper.h"

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifndef NO_OPENCL
int scanhash_metis_gpu(int device, enum sha256_algos algo, int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done) {

	int i;
	int ret;
	uint32_t data[20];
	// do not swap the "nonce" or else i get lost in count
	for (i = 0; i < 19; i++) {
		data[i] = swab32(pdata[i]);
	}
	data[19] = pdata[19];

	ret = scanhash_metis_opencl(device, algo, thr_id, data, ptarget, max_nonce, hashes_done);
	pdata[19] = swab32(data[19]);
	if (ret) {
		// validates
		printf("validating share...\n");
		return validate(data, ptarget);
	}

	return ret;

}
#endif

int validate(const uint32_t *pdata,
	const uint32_t *ptarget)
{
	const uint32_t Htarg = ptarget[7];
	uint64_t hash0[8];
	uint64_t hash1[8];
	uint64_t hash2[8];

	sph_keccak224_context kctx;
	sph_shavite512_context sctx;
	sph_metis512_context mctx;

	sph_keccak512_init(&kctx);
	sph_keccak512(&kctx, pdata, 80);
	sph_keccak512_close(&kctx, hash0);

	sph_shavite512_init(&sctx);
	sph_shavite512(&sctx, hash0, 64);
	sph_shavite512_close(&sctx, hash1);

	sph_metis512_init(&mctx);
	sph_metis512(&mctx, hash1, 64);
	sph_metis512_close(&mctx, hash2);

	if( *(uint32_t*)((uint8_t*)hash2 + 28) <= Htarg && fulltest(hash2, ptarget) )
	{
		return 1;
	}

	return 0;
}

void OLD_HASH_ALGO(const uint32_t *pdata,
	const uint64_t *out)
{
	uint64_t hash0[8];
	uint64_t hash1[8];

	sph_keccak224_context kctx;
	sph_shavite512_context sctx;
	sph_metis512_context mctx;

	sph_keccak512_init(&kctx);
	sph_keccak512(&kctx, pdata, 80);
	sph_keccak512_close(&kctx, hash0);

	sph_shavite512_init(&sctx);
	sph_shavite512(&sctx, hash0, 64);
	sph_shavite512_close(&sctx, hash1);

	sph_metis512_init(&mctx);
	sph_metis512(&mctx, hash1, 64);
	sph_metis512_close(&mctx, out);

}


#if defined(_MSC_VER)
#define _ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__)
#define _ALIGNED(x) __attribute__ ((aligned(x)))
#else
#define _ALIGNED(x) (x)
#endif

#define GROUPED_HASHES (512)

int scanhash_metis_cpu(int thr_id, uint32_t *pdata,	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done) {

	int i, n;
	int ret;
	uint32_t data[20];
	uint32_t first_nonce;
	const uint32_t Htarg = ptarget[7];
	sph_keccak224_context kctx;
	sph_shavite512_context sctx;
	sph_metis512_context mctx;
	_ALIGNED(32) uint32_t metisPartData[36*GROUPED_HASHES];
	_ALIGNED(32) uint64_t hash0[8*GROUPED_HASHES];
	uint64_t keccakPre[25];

	first_nonce = pdata[19];
	for (i = 0; i < 20; i++) {
		data[i] = swab32(pdata[i]);
	}

	sph_keccak512_init(&kctx);
	keccak_core_prepare(&kctx, data, keccakPre);

	for (n = first_nonce; n < max_nonce; n+= GROUPED_HASHES) {

		// todo: Generate multiple hashes for multiple nonces at once
#pragma unroll
		for(uint32_t i=0; i<GROUPED_HASHES; i++)
		{
			data[19] = swab32(n+i);
			keccak_core_opt(&kctx, keccakPre, *(uint64_t*)(data+18), hash0+i*8);
		}
#pragma unroll
		for(uint32_t i=0; i<GROUPED_HASHES; i++)
		{
			shavite_big_core_opt(hash0+i*8, hash0+i*8);
		}
#pragma unroll
		for(uint32_t i=0; i<GROUPED_HASHES; i++)
		{
			metis4_core_opt_p1((unsigned int *)(hash0+i*8), metisPartData+i*36);
		}
#pragma unroll
		for(int i=0; i<GROUPED_HASHES; i++)
		{
#ifdef VALIDATE_ALGORITHMS
			uint32_t tmp = metis4_core_opt_p2(metisPartData+i*36);
			data[19] = swab32(n+i);
			OLD_HASH_ALGO(data, hash0);
			if (((uint32_t*)hash0)[7] !=  tmp) {
				printf("ERROR: %X != %X\n", ((uint32_t*)hash0)[7],  tmp);
			}
#endif
			if( metis4_core_opt_p2(metisPartData+i*36) <= Htarg )
			{
				pdata[19] = n+i;
				*hashes_done = n+i-first_nonce;
				// validate
				data[19] = swab32(pdata[19]);
				return validate(data, ptarget);
			}
		}


	}

	pdata[19] = n;
	*hashes_done = n-first_nonce;
	return 0;

}
