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

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

int scanhash_metis(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done) {

	int i;
	int ret;

	ret = scanhash_metis_opencl(thr_id, pdata, ptarget, max_nonce, hashes_done);
	if (ret) {
		// validates
		printf("validating share...\n");
		return validate(pdata, ptarget);
	}

	return ret;

}


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
