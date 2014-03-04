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

void init_opencl_miner();
int scanhash_metis_opencl(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

#ifdef __cplusplus
}
#endif

#endif /* METISCOINMINERC_H_ */
