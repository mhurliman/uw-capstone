
#include <fstream>
#include <iostream>
#include <vector>

#include <fftw3.h>

bool LoadFile(const char* filename, std::vector<float>& xvals, std::vector<float>& yvals)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
		return false;

	uint32_t count = 0;
	file.read((char*)&count, 4);

	xvals.resize(count);
	yvals.resize(count);

	for (int i = 0; i < count; ++i)
	{	
		file.read((char*)&xvals[i], 4);
	}
	for (int i = 0; i < count; ++i)
	{	
		file.read((char*)&yvals[i], 4);
	}
	file.close();
	
	return true;
}

int main(int argc, const char** argv)
{
	if (argc < 2)
		return 1;

	std::vector<float> xvals, yvals;
	if (!LoadFile(argv[1], xvals, yvals))
	{
		return 1;
	}

	int N = xvals.size();
	int OutN = N / 2 + 1;

	fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * OutN);

	for (int i = 0; i < N; ++i)
	{
		in[i][0] = yvals[i];
		in[i][1] = 0.0;
	}

	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(p);

	for (int i = 0; i < OutN; ++i)
	{
		float real = out[i][0] < 1e-5 ? 0 : out[i][0];
		float imag = out[i][1] < 1e-5 ? 0 : out[i][1];
		std::cout << i << ": " << real << ", " << imag << "\n";
	}

	std::cout.flush();

	fftw_destroy_plan(p);
	fftw_free(in); fftw_free(out);

	return 0;
}
