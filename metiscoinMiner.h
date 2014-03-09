#ifndef __METISCOIN_MINER_H__
#define __METISCOIN_MINER_H__

// utils
int log2(size_t value);


class MetiscoinOpenCL {
public:
	MetiscoinOpenCL(int _device_num, uint32_t _step_size);
	virtual ~MetiscoinOpenCL() { };
	virtual int metiscoin_process(int thr_id, uint32_t *pdata,
			const uint32_t *ptarget,
			uint32_t max_nonce, unsigned long *hashes_done) = 0;
protected:
	int device_num;
	uint32_t STEP_SIZE;
	uint32_t NUM_STEPS;

	OpenCLBuffer* u;
	OpenCLBuffer* buff;
	OpenCLBuffer* hashes;
	OpenCLBuffer* out;
	OpenCLBuffer* out_count;
	OpenCLBuffer* begin_nonce;
	OpenCLBuffer* target;
	OpenCLCommandQueue * q;
	cl_uint out_tmp[255];
	cl_uint tmp_out_count;
	cl_uint tmp_begin_nonce;
	cl_uint tmp_target;

	size_t wg_size;
};


class MetiscoinOpenCLConstant : public MetiscoinOpenCL {
public:

	MetiscoinOpenCLConstant(int device_num, uint32_t _step_size);
	int metiscoin_process(int thr_id, uint32_t *pdata,
			const uint32_t *ptarget,
			uint32_t max_nonce, unsigned long *hashes_done);
private:
	OpenCLKernel* kernel_all;
	OpenCLKernel* kernel_keccak_noinit;
	OpenCLKernel* kernel_shavite;
	OpenCLKernel* kernel_metis;
};

class MetiscoinOpenCLGlobal : public MetiscoinOpenCL {
public:

	MetiscoinOpenCLGlobal(int _device_num, uint32_t _step_size);
	int metiscoin_process(int thr_id, uint32_t *pdata,
			const uint32_t *ptarget,
			uint32_t max_nonce, unsigned long *hashes_done);
private:

	OpenCLKernel* kernel_all;
	OpenCLKernel* kernel_keccak_noinit;
	OpenCLKernel* kernel_shavite;
	OpenCLKernel* kernel_metis;

	OpenCLBuffer* shavite_lookup;
	OpenCLBuffer* fugue_lookup;
};

class MetiscoinOpenCLSingle : public MetiscoinOpenCL {
public:

	MetiscoinOpenCLSingle(int _device_num, uint32_t _step_size);
	int metiscoin_process(int thr_id, uint32_t *pdata,
			const uint32_t *ptarget,
			uint32_t max_nonce, unsigned long *hashes_done);
private:

	OpenCLKernel* kernel_single_noinit;

	OpenCLBuffer* shavite_lookup;
	OpenCLBuffer* fugue_lookup;
};

//class MetiscoinOpenCLSingle1ghKeccak : public MetiscoinOpenCL {
//public:
//
//	MetiscoinOpenCLSingle1ghKeccak(int _device_num, uint32_t _step_size);
//	int metiscoin_process(int thr_id, uint32_t *pdata,
//			const uint32_t *ptarget,
//			uint32_t max_nonce, unsigned long *hashes_done);
//private:
//
//	OpenCLKernel* kernel_single_noinit;
//
//	OpenCLBuffer* shavite_lookup;
//	OpenCLBuffer* fugue_lookup;
//};

#endif
