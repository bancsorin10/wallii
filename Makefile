
all:
	rm train model || true
	gcc -pthread -lm -ggdb -Wall -Wextra train.c decode_image.c utils.c nn_functions.c construct.c -o train
	gcc -lm -Wall -Wextra validate.c nn_functions.c -o model
