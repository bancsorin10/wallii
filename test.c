
#include <stdio.h>

#define TEST 0xffd8


int main() {
    printf("%x\n", TEST);
    printf("%x\n", TEST >> 8);
    printf("%x\n", (unsigned char)(TEST >> 8) == 0xff);
    printf("%x\n", TEST == (0xff << 8 | 0xd8));

    printf("-------------------------\n");
    printf("%.5f\n", (double)'a');

    printf("-------------------------\n");
    printf("%.10f\n", 1e-7);
    return 0;
}