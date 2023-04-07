CC=nvc
CFLAGS?=-Wall -tp=native -c17 -acc -mp -Minfo=all -g #-gpu=managed #-O3 -Mnovect
#-Wno-unused-result  

relu : relu_forward.o volume.o
	$(CC) $(CFLAGS) -o relu relu_forward.o volume.o -lm

# compare : relu_forward baseline
# 	./relu_forward relu_forward
# 	./relu_forward_baseline relu_forward

relu_forward.o : relu_forward.c volume.h
	$(CC) $(CFLAGS) -c relu_forward.c

volume.o : volume.c volume.h
	$(CC) $(CFLAGS) -c volume.c

clean:
	rm -f *.o
	rm -f relu
	rm outputs/*.txt

.PHONY : clean
