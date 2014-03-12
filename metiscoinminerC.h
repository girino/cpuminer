/*
 * metiscoinminerC.h
 *
 *  Created on: 03/03/2014
 *      Author: girino
 */

#ifndef METISCOINMINERC_H_
#define METISCOINMINERC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "miner.h"

void list_devices();
void init_opencl_miner(int device, enum sha256_algos algo, int thr_id);
int scanhash_metis_opencl(int device, enum sha256_algos algo, int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

// selects the best GPU algorithm
enum sha256_algos benchmark(int device);

#ifdef __cplusplus
}
#endif

#endif /* METISCOINMINERC_H_ */
