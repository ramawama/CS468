#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t input_size, uint16_t bin_size);

/* Include below the function headers of any other functions that you implement */
void opt_2dhisto_setup(uint32_t *input, int h, int w);

void opt_2dhisto_teardown(uint8_t *bins);

#endif
