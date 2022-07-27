
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdlib.h>

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

    printf("-------------------------\n");

    struct dirent **namelist;
    int n;
    n = scandir("inputs/", &namelist, NULL, alphasort);

           while (n--) {
               printf("%s\n", namelist[n]->d_name);
               free(namelist[n]);
           }
           free(namelist);

    return 0;
}
