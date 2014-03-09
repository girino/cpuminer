#include "OpenCLObjects.h"
#include "metiscoinMiner.h"
#include "tables_single.h"
#include "sph_keccak.h"
#include "sph_shavite.h"
#include "sph_metis.h"
#include "metiscoinminerC.h"
#include "miner.h"
#include <stdio.h>

int log2(size_t value) {
	int ret = 0;
	while (value > 1) {
		ret++;
		value = value>>1;
	}
	return ret;
}

MetiscoinOpenCL::MetiscoinOpenCL(int _device_num, uint32_t _step_size) {
	this->device_num = _device_num;
	this->STEP_SIZE = _step_size;
	this->NUM_STEPS = (uint32_t)(0x10000000L/STEP_SIZE);
	printf("Initializing GPU %d\n", device_num);
	OpenCLMain &main = OpenCLMain::getInstance();
	OpenCLDevice* device = main.getDevice(device_num);
	printf("Initializing Device: %s\n", device->getName().c_str());

	q = device->getContext()->createCommandQueue(device);

	u = device->getContext()->createBuffer(25*sizeof(cl_ulong), CL_MEM_READ_ONLY, NULL);
	buff = device->getContext()->createBuffer(4, CL_MEM_READ_ONLY, NULL);

	hashes = device->getContext()->createBuffer(64 * STEP_SIZE, CL_MEM_READ_WRITE, NULL);
	out = device->getContext()->createBuffer(sizeof(cl_uint) * 255, CL_MEM_WRITE_ONLY, NULL);
	out_count = device->getContext()->createBuffer(sizeof(cl_uint), CL_MEM_READ_WRITE, NULL);
	begin_nonce = device->getContext()->createBuffer(sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	target = device->getContext()->createBuffer(sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
}


MetiscoinOpenCLConstant::MetiscoinOpenCLConstant(int _device_num, uint32_t _step_size) : MetiscoinOpenCL(_device_num, _step_size) {

	printf ("Initing algo with constant memspace...\n");

	OpenCLMain &main = OpenCLMain::getInstance();
	OpenCLDevice* device = main.getDevice(device_num);
	std::vector<std::string> files_keccak;
	files_keccak.push_back("opencl/common.cl");
	files_keccak.push_back("opencl/keccak.cl");
	files_keccak.push_back("opencl/shavite_NVidia.cl"); // way faster on NVidia, not much slower on AMD
	//files_keccak.push_back("opencl/shavite_AMD.cl");
	files_keccak.push_back("opencl/metis.cl");
	files_keccak.push_back("opencl/tables.cl");
	files_keccak.push_back("opencl/miner_constant.cl");
#ifdef VALIDATE_ALGORITHMS
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak, "-DVALIDATE_ALGORITHMS");
#else
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak);
#endif
	kernel_keccak_noinit = program->getKernel("keccak_step_noinit");
	kernel_shavite = program->getKernel("shavite_step");
	kernel_metis = program->getKernel("metis_step");

	// params
	//kernel void keccak_step_noinit(constant const ulong* u, constant const char* buff, global ulong* out, uint begin_nonce)
	kernel_keccak_noinit->resetArgs();
	kernel_keccak_noinit->addGlobalArg(u);
	kernel_keccak_noinit->addGlobalArg(buff);
	kernel_keccak_noinit->addGlobalArg(hashes);
	kernel_keccak_noinit->addGlobalArg(begin_nonce);

	// shavite
	kernel_shavite->resetArgs();
	kernel_shavite->addGlobalArg(hashes);

	// metis_step(global ulong* in, global uint* out, global uint* outcount, uint begin_nonce, uint target) {
	kernel_metis->resetArgs();
	kernel_metis->addGlobalArg(hashes);
	kernel_metis->addGlobalArg(out);
	kernel_metis->addGlobalArg(out_count);
	kernel_metis->addGlobalArg(begin_nonce);
	kernel_metis->addGlobalArg(target);

	// work group sizes
	wg_size = kernel_keccak_noinit->getWorkGroupSize(device);
	size_t wgs_tmp = kernel_shavite->getWorkGroupSize(device);
	if (wgs_tmp < wg_size) wg_size = wgs_tmp;
	wgs_tmp = kernel_metis->getWorkGroupSize(device);
	if (wgs_tmp < wg_size) wg_size = wgs_tmp;
#ifdef DEBUG_WORKGROUP_SIZE
	printf ("wg_size = %d => %d\n", wg_size, 1 << log2(wg_size));
#endif
	wg_size = 1 << log2(wg_size); // guarantees to be a power of 2

}

int MetiscoinOpenCLConstant::metiscoin_process(int thr_id, uint32_t *pdata,
		const uint32_t *ptarget,
		uint32_t max_nonce, unsigned long *hashes_done)
{

	tmp_begin_nonce = pdata[19];
	tmp_target = ptarget[7];
	OpenCLDevice* device = OpenCLMain::getInstance().getDevice(device_num);
	//printf("processing block with Device: %s\n", device->getName().c_str());

	sph_keccak512_context	 ctx_keccak;
	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, pdata, 80);

	q->enqueueWriteBuffer(u, ctx_keccak.u.narrow, 25*sizeof(cl_ulong));
	q->enqueueWriteBuffer(buff, ctx_keccak.buf, 4);
	q->enqueueWriteBuffer(target, &tmp_target, sizeof(cl_uint));

	NUM_STEPS = (max_nonce - tmp_begin_nonce) / STEP_SIZE;
	if (NUM_STEPS < 1) NUM_STEPS = 1;

	for (uint32_t n = 0; n < NUM_STEPS; n++)
	{
		tmp_begin_nonce = (n * STEP_SIZE) + pdata[19];

		//keccak
		q->enqueueWriteBuffer(begin_nonce, &tmp_begin_nonce, sizeof(cl_uint));

		q->enqueueKernel1D(kernel_keccak_noinit, STEP_SIZE, wg_size);
		// shavite
		q->enqueueKernel1D(kernel_shavite, STEP_SIZE, wg_size);
		tmp_out_count = 0;
		q->enqueueWriteBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->enqueueKernel1D(kernel_metis, STEP_SIZE, wg_size);
		q->enqueueReadBuffer(out, out_tmp, sizeof(cl_uint) * 255);
		q->enqueueReadBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->finish();

		if (tmp_out_count > 0) {
			*hashes_done = n * STEP_SIZE;
			pdata[19] = out_tmp[0];
			return 1;
		}
	}
	*hashes_done = (NUM_STEPS*STEP_SIZE);
	pdata[19] = pdata[19] + *hashes_done;
	return 0;
}

MetiscoinOpenCLGlobal::MetiscoinOpenCLGlobal(int _device_num, uint32_t _step_size) : MetiscoinOpenCL(_device_num, _step_size) {

	printf ("Initing algo with global memspace...\n");

	OpenCLMain &main = OpenCLMain::getInstance();
	OpenCLDevice* device = main.getDevice(device_num);
	std::vector<std::string> files_keccak;
	files_keccak.push_back("opencl/common.cl");
	files_keccak.push_back("opencl/keccak.cl");
	files_keccak.push_back("opencl/shavite_NVidia.cl"); // way faster on NVidia, not much slower on AMD
	//files_keccak.push_back("opencl/shavite_AMD.cl");
	files_keccak.push_back("opencl/metis.cl");
	files_keccak.push_back("opencl/miner_global.cl");
#ifdef VALIDATE_ALGORITHMS
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak, "-DVALIDATE_ALGORITHMS");
#else
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak);
#endif
	kernel_keccak_noinit = program->getKernel("keccak_step_noinit");
	kernel_shavite = program->getKernel("shavite_step");
	kernel_metis = program->getKernel("metis_step");

	// allocs lookup tables
	fugue_lookup = device->getContext()->createBuffer(1024*sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	shavite_lookup = device->getContext()->createBuffer(1024*sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	// enqueue write tables
	q->enqueueWriteBuffer(shavite_lookup, (void*)AES_c, 1024*sizeof(cl_uint));
	q->enqueueWriteBuffer(fugue_lookup, (void*)mixtab_c, 1024*sizeof(cl_uint));

	// params
	//kernel void keccak_step_noinit(constant const ulong* u, constant const char* buff, global ulong* out, uint begin_nonce)
	kernel_keccak_noinit->resetArgs();
	kernel_keccak_noinit->addGlobalArg(u);
	kernel_keccak_noinit->addGlobalArg(buff);
	kernel_keccak_noinit->addGlobalArg(hashes);
	kernel_keccak_noinit->addGlobalArg(begin_nonce);

	// shavite
	kernel_shavite->resetArgs();
	kernel_shavite->addGlobalArg(hashes);
	kernel_shavite->addGlobalArg(shavite_lookup);

	// metis_step(global ulong* in, global uint* out, global uint* outcount, uint begin_nonce, uint target) {
	kernel_metis->resetArgs();
	kernel_metis->addGlobalArg(hashes);
	kernel_metis->addGlobalArg(out);
	kernel_metis->addGlobalArg(out_count);
	kernel_metis->addGlobalArg(begin_nonce);
	kernel_metis->addGlobalArg(target);
	kernel_metis->addGlobalArg(fugue_lookup);

	// work group sizes
	wg_size = kernel_keccak_noinit->getWorkGroupSize(device);
	size_t wgs_tmp = kernel_shavite->getWorkGroupSize(device);
	if (wgs_tmp < wg_size) wg_size = wgs_tmp;
	wgs_tmp = kernel_metis->getWorkGroupSize(device);
	if (wgs_tmp < wg_size) wg_size = wgs_tmp;
#ifdef DEBUG_WORKGROUP_SIZE
	printf ("wg_size = %d => %d\n", wg_size, 1 << log2(wg_size));
#endif
	wg_size = 1 << log2(wg_size); // guarantees to be a power of 2

}

int MetiscoinOpenCLGlobal::metiscoin_process(int thr_id, uint32_t *pdata,
		const uint32_t *ptarget,
		uint32_t max_nonce, unsigned long *hashes_done)
{

	tmp_begin_nonce = pdata[19];
	tmp_target = ptarget[7];
	OpenCLDevice* device = OpenCLMain::getInstance().getDevice(device_num);
	//printf("processing block with Device: %s\n", device->getName().c_str());


	sph_keccak512_context	 ctx_keccak;
	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, pdata, 80);

	q->enqueueWriteBuffer(u, ctx_keccak.u.narrow, 25*sizeof(cl_ulong));
	q->enqueueWriteBuffer(buff, ctx_keccak.buf, 4);
	q->enqueueWriteBuffer(target, &tmp_target, sizeof(cl_uint));

	NUM_STEPS = (max_nonce - tmp_begin_nonce) / STEP_SIZE;
	if (NUM_STEPS < 1) NUM_STEPS = 1;

	for (uint32_t n = 0; n < NUM_STEPS; n++)
	{
		tmp_begin_nonce = (n * STEP_SIZE) + pdata[19];

		q->enqueueWriteBuffer(begin_nonce, &tmp_begin_nonce, sizeof(cl_uint));

		q->enqueueKernel1D(kernel_keccak_noinit, STEP_SIZE,
				wg_size);
		// shavite
		q->enqueueKernel1D(kernel_shavite, STEP_SIZE, wg_size);
		// metis
		tmp_out_count = 0;
		q->enqueueWriteBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->enqueueKernel1D(kernel_metis, STEP_SIZE, wg_size);
		q->enqueueReadBuffer(out, out_tmp, sizeof(cl_uint) * 255);
		q->enqueueReadBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->finish();

		if (tmp_out_count > 0) {
			*hashes_done = n * STEP_SIZE;
			pdata[19] = out_tmp[0];
			return 1;
		}
	}
	*hashes_done = (NUM_STEPS*STEP_SIZE);
	pdata[19] = pdata[19] + *hashes_done;
	return 0;
}



MetiscoinOpenCLSingle::MetiscoinOpenCLSingle(int _device_num, uint32_t _step_size) : MetiscoinOpenCL(_device_num, _step_size) {

	printf ("Initing algo with global memspace and single kernel using local memory...\n");

	OpenCLMain &main = OpenCLMain::getInstance();
	OpenCLDevice* device = main.getDevice(device_num);
	std::vector<std::string> files_keccak;
	files_keccak.push_back("opencl/common.cl");
	files_keccak.push_back("opencl/keccak.cl");
	files_keccak.push_back("opencl/shavite_NVidia.cl"); // way faster on NVidia, not much slower on AMD
	//files_keccak.push_back("opencl/shavite_AMD.cl");
	files_keccak.push_back("opencl/metis.cl");
	files_keccak.push_back("opencl/miner_single.cl");
#ifdef VALIDATE_ALGORITHMS
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak, "-DLOCAL_HASHES -DVALIDATE_ALGORITHMS");
#else
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak, "-DLOCAL_HASHES");
#endif
	kernel_single_noinit = program->getKernel("single_noinit");

	// allocs lookup tables
	fugue_lookup = device->getContext()->createBuffer(1024*sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	shavite_lookup = device->getContext()->createBuffer(1024*sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	// enqueue write tables
	q->enqueueWriteBuffer(shavite_lookup, (void*)AES_c, 1024*sizeof(cl_uint));
	q->enqueueWriteBuffer(fugue_lookup, (void*)mixtab_c, 1024*sizeof(cl_uint));


	// work group sizes
	wg_size = kernel_single_noinit->getWorkGroupSize(device);
	// checks if there's enough local mem for the kernel
	size_t max_mem = device->getMaxMemAllocSize();
	size_t local_mem = device->getLocalMemSize();
	size_t reserved_mem = 8 * 256 * sizeof(cl_uint);
	if ((max_mem-reserved_mem)/(8*sizeof(cl_ulong)) < wg_size) wg_size = (max_mem-reserved_mem)/(8*sizeof(cl_ulong));
	if ((local_mem-reserved_mem)/(8*sizeof(cl_ulong)) < wg_size) wg_size = (local_mem-reserved_mem)/(8*sizeof(cl_ulong));
#ifdef DEBUG_WORKGROUP_SIZE
	printf ("wg_size = %d => %d\n", wg_size, 1 << log2(wg_size));
#endif
	if (STEP_SIZE % wg_size != 0) wg_size = 1 << log2(wg_size); // guarantees to be a multiple

	// params
	//kernel void keccak_step_noinit(constant const ulong* u, constant const char* buff, global ulong* out, uint begin_nonce)
	kernel_single_noinit->resetArgs();
	kernel_single_noinit->addGlobalArg(u);
	kernel_single_noinit->addGlobalArg(buff);
	kernel_single_noinit->addLocalArg(8*wg_size*sizeof(cl_ulong));
	kernel_single_noinit->addGlobalArg(shavite_lookup);
	kernel_single_noinit->addGlobalArg(fugue_lookup);
	kernel_single_noinit->addGlobalArg(out);
	kernel_single_noinit->addGlobalArg(out_count);
	kernel_single_noinit->addGlobalArg(begin_nonce);
	kernel_single_noinit->addGlobalArg(target);

}

int MetiscoinOpenCLSingle::metiscoin_process(int thr_id, uint32_t *pdata,
		const uint32_t *ptarget,
		uint32_t max_nonce, unsigned long *hashes_done)
{


 	tmp_begin_nonce = pdata[19];
	tmp_target = ptarget[7];
	OpenCLDevice* device = OpenCLMain::getInstance().getDevice(device_num);
#ifdef DEBUG_DATA
	printf("Begin nonce: %X\n", tmp_begin_nonce);
	printf("Max nonce: %X\n", max_nonce);
	printf("Data: %08X %08X %08X %08X\n", pdata[0], pdata[1], pdata[2], pdata[3]);
	printf("      %08X %08X %08X %08X\n", pdata[4], pdata[5], pdata[6], pdata[7]);
	printf("      %08X %08X %08X %08X\n", pdata[8], pdata[9], pdata[10], pdata[11]);
	printf("      %08X %08X %08X %08X\n", pdata[12], pdata[13], pdata[14], pdata[15]);
	printf("      %08X %08X %08X %08X\n", pdata[16], pdata[17], pdata[18], pdata[19]);
#endif

	sph_keccak512_context	 ctx_keccak;
	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, pdata, 80);

	q->enqueueWriteBuffer(u, ctx_keccak.u.narrow, 25*sizeof(cl_ulong));
	q->enqueueWriteBuffer(buff, ctx_keccak.buf, 4);
	q->enqueueWriteBuffer(target, &tmp_target, sizeof(cl_uint));

	if (tmp_begin_nonce >= max_nonce) {
		NUM_STEPS = 1;
	} else {
		NUM_STEPS = (max_nonce - tmp_begin_nonce) / STEP_SIZE;
		if (NUM_STEPS < 1) NUM_STEPS = 1;
	}

	for (uint32_t n = 0; n < NUM_STEPS; n++)
	{
		tmp_begin_nonce = (n * STEP_SIZE) + pdata[19];

		q->enqueueWriteBuffer(begin_nonce, &tmp_begin_nonce, sizeof(cl_uint));
		tmp_out_count = 0;
		q->enqueueWriteBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->enqueueKernel1D(kernel_single_noinit, STEP_SIZE, wg_size);
		q->enqueueReadBuffer(out, out_tmp, sizeof(cl_uint) * 255);
		q->enqueueReadBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->finish();

		if (tmp_out_count > 0) {
			*hashes_done = n * STEP_SIZE;
			pdata[19] = out_tmp[0];
			return 1;
		}
	}
	*hashes_done = (NUM_STEPS*STEP_SIZE);
	pdata[19] = pdata[19] + *hashes_done;
#ifdef DEBUG_DATA
	printf("End nonce: %X\n", pdata[19]);
#endif

	return 0;
}

MetiscoinOpenCLSingle1ghKeccak::MetiscoinOpenCLSingle1ghKeccak(int _device_num, uint32_t _step_size) : MetiscoinOpenCL(_device_num, _step_size) {

	printf ("Initing algo with global memspace and single kernel using local memory...\n");

	OpenCLMain &main = OpenCLMain::getInstance();
	OpenCLDevice* device = main.getDevice(device_num);
	std::vector<std::string> files_keccak;
	files_keccak.push_back("opencl/common.cl");
	files_keccak.push_back("opencl/keccak1gh.cl");
	files_keccak.push_back("opencl/shavite_NVidia.cl"); // way faster on NVidia, not much slower on AMD
	//files_keccak.push_back("opencl/shavite_AMD.cl");
	files_keccak.push_back("opencl/metis.cl");
	files_keccak.push_back("opencl/miner_single1gh.cl");
#ifdef VALIDATE_ALGORITHMS
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak, "-DLOCAL_HASHES -DVALIDATE_ALGORITHMS");
#else
	OpenCLProgram* program = device->getContext()->loadProgramFromFiles(files_keccak, "-DLOCAL_HASHES");
#endif
	kernel_single_noinit = program->getKernel("single_noinit");

	// allocs lookup tables
	fugue_lookup = device->getContext()->createBuffer(1024*sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	shavite_lookup = device->getContext()->createBuffer(1024*sizeof(cl_uint), CL_MEM_READ_ONLY, NULL);
	// enqueue write tables
	q->enqueueWriteBuffer(shavite_lookup, (void*)AES_c, 1024*sizeof(cl_uint));
	q->enqueueWriteBuffer(fugue_lookup, (void*)mixtab_c, 1024*sizeof(cl_uint));


	// work group sizes
	wg_size = kernel_single_noinit->getWorkGroupSize(device);
	// checks if there's enough local mem for the kernel
	size_t max_mem = device->getMaxMemAllocSize();
	size_t local_mem = device->getLocalMemSize();
	size_t reserved_mem = 8 * 256 * sizeof(cl_uint);
	if ((max_mem-reserved_mem)/(8*sizeof(cl_ulong)) < wg_size) wg_size = (max_mem-reserved_mem)/(8*sizeof(cl_ulong));
	if ((local_mem-reserved_mem)/(8*sizeof(cl_ulong)) < wg_size) wg_size = (local_mem-reserved_mem)/(8*sizeof(cl_ulong));
#ifdef DEBUG_WORKGROUP_SIZE
	printf ("wg_size = %d => %d\n", wg_size, 1 << log2(wg_size));
#endif
	if (STEP_SIZE % wg_size != 0) wg_size = 1 << log2(wg_size); // guarantees to be a multiple

	// params
	//kernel void keccak_step_noinit(constant const ulong* u, constant const char* buff, global ulong* out, uint begin_nonce)
	kernel_single_noinit->resetArgs();
	kernel_single_noinit->addGlobalArg(u);
	kernel_single_noinit->addGlobalArg(buff);
	kernel_single_noinit->addLocalArg(8*wg_size*sizeof(cl_ulong));
	kernel_single_noinit->addGlobalArg(shavite_lookup);
	kernel_single_noinit->addGlobalArg(fugue_lookup);
	kernel_single_noinit->addGlobalArg(out);
	kernel_single_noinit->addGlobalArg(out_count);
	kernel_single_noinit->addGlobalArg(begin_nonce);
	kernel_single_noinit->addGlobalArg(target);

}

int MetiscoinOpenCLSingle1ghKeccak::metiscoin_process(int thr_id, uint32_t *pdata,
		const uint32_t *ptarget,
		uint32_t max_nonce, unsigned long *hashes_done)
{


 	tmp_begin_nonce = pdata[19];
	tmp_target = ptarget[7];
	OpenCLDevice* device = OpenCLMain::getInstance().getDevice(device_num);
#ifdef DEBUG_DATA
	printf("Begin nonce: %X\n", tmp_begin_nonce);
	printf("Max nonce: %X\n", max_nonce);
	printf("Data: %08X %08X %08X %08X\n", pdata[0], pdata[1], pdata[2], pdata[3]);
	printf("      %08X %08X %08X %08X\n", pdata[4], pdata[5], pdata[6], pdata[7]);
	printf("      %08X %08X %08X %08X\n", pdata[8], pdata[9], pdata[10], pdata[11]);
	printf("      %08X %08X %08X %08X\n", pdata[12], pdata[13], pdata[14], pdata[15]);
	printf("      %08X %08X %08X %08X\n", pdata[16], pdata[17], pdata[18], pdata[19]);
#endif

	sph_keccak512_context	 ctx_keccak;
	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, pdata, 80);

	q->enqueueWriteBuffer(u, ctx_keccak.u.narrow, 25*sizeof(cl_ulong));
	q->enqueueWriteBuffer(buff, ctx_keccak.buf, 4);
	q->enqueueWriteBuffer(target, &tmp_target, sizeof(cl_uint));

	if (tmp_begin_nonce >= max_nonce) {
		NUM_STEPS = 1;
	} else {
		NUM_STEPS = (max_nonce - tmp_begin_nonce) / STEP_SIZE;
		if (NUM_STEPS < 1) NUM_STEPS = 1;
	}

	for (uint32_t n = 0; n < NUM_STEPS; n++)
	{
		tmp_begin_nonce = (n * STEP_SIZE) + pdata[19];

		q->enqueueWriteBuffer(begin_nonce, &tmp_begin_nonce, sizeof(cl_uint));
		tmp_out_count = 0;
		q->enqueueWriteBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->enqueueKernel1D(kernel_single_noinit, STEP_SIZE, wg_size);
		q->enqueueReadBuffer(out, out_tmp, sizeof(cl_uint) * 255);
		q->enqueueReadBuffer(out_count, &tmp_out_count, sizeof(cl_uint));
		q->finish();

		if (tmp_out_count > 0) {
			*hashes_done = n * STEP_SIZE;
			pdata[19] = out_tmp[0];
			return 1;
		}
	}
	*hashes_done = (NUM_STEPS*STEP_SIZE);
	pdata[19] = pdata[19] + *hashes_done;
#ifdef DEBUG_DATA
	printf("End nonce: %X\n", pdata[19]);
#endif

	return 0;
}

extern "C" {


void list_devices() {
	OpenCLMain::getInstance().listDevices();
}

MetiscoinOpenCL* processors[255];
int is_processors_inited = 0;
void init_opencl_miner(int device, enum sha256_algos algo, int thr_id) {
	switch(algo) {
	case ALGO_METIS_GPU_1: processors[thr_id] = new MetiscoinOpenCLConstant(device, opt_step_size); break;
	case ALGO_METIS_GPU_2: processors[thr_id] = new MetiscoinOpenCLGlobal(device, opt_step_size); break;
	case ALGO_METIS_GPU_3: processors[thr_id] = new MetiscoinOpenCLSingle1ghKeccak(device, opt_step_size); break;

	}
}
int scanhash_metis_opencl(int device, enum sha256_algos algo, int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done) {

	int i;

	if (!is_processors_inited) {
		is_processors_inited = 1;
		for (i = 0; i < 255; i++) processors[i] = NULL;
	}
	if (processors[thr_id] == NULL) {
		init_opencl_miner(device, algo, thr_id);
	}

	return processors[thr_id]->metiscoin_process(thr_id, pdata,	ptarget, max_nonce, hashes_done);

}
}
