
all:
	rm a.out
	gcc -pthread -lm -ggdb -Wall -Wextra main.c decode_image.c utils.c nn_functions.c construct.c
