
all:
	rm a.out
	gcc -lm -ggdb -Wall -Wextra main.c decode_image.c utils.c
