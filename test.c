
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdlib.h>
#include <pthread.h>

#define TEST 0xffd8

void *print_ptid()
{
    pthread_t ptid;
    ptid = pthread_self();
    printf("pthread id: %lu\n", ptid);
    return NULL;
}


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



    printf("----------------------------\n");
    unsigned int i,j;
    pthread_t *ptid;
    ptid = (pthread_t *)malloc(sizeof(pthread_t)*8);
    for (i = 0; i < 5; ++i)
    {
        for (j = 0; j < 8; ++j)
        {
            pthread_create(&ptid[i], NULL, &print_ptid, NULL);
        }
        for (j = 0; j < 8; ++j)
        {
            pthread_join(ptid[j], NULL);
        }
    }

    return 0;
}
